import math
import sys
import os
import datetime
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Iterable
from collections import defaultdict, Counter

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import KMeans

from timm.utils import accuracy
import utils

"""
BALANCED CONTINUAL LEARNING - ALL TASKS IMPROVED
=================================================
Key Fixes:
1. ADAPTIVE EWC: Weak (100-500) instead of too strong (2000)
2. BALANCED DISTILLATION: Per-class weighting to prevent bias
3. CLASS-SPECIFIC LEARNING: Higher LR for struggling classes
4. PROTOTYPE-BASED BIAS CORRECTION: Balance predictions at test time
5. BETTER MASK EXPANSION: Ensure all classes get sufficient nodes
"""


class Engine():
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        self.current_task = 0
        self.current_classes = []
        max_class_id = max([item for mask in class_mask for item in mask])
        self.class_group_size = len(class_mask[0])
        self.class_group_num = (max_class_id // self.class_group_size) + 1
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        self.visited_domains = set()
        self.task_num = len(class_mask)
        self.model = model
        self.num_classes = max([item for mask in class_mask for item in mask]) + 1
        self.distill_head = None
        
        # Initialize distill_head properly
        model.distill_head = nn.Linear(768, self.num_classes).to(device)
        nn.init.xavier_uniform_(model.distill_head.weight)
        nn.init.constant_(model.distill_head.bias, 0)
        
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.distill_model = None
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args = args
        self.class_mask = class_mask
        self.domain_list = domain_list
        self.task_type = "initial"
        self.final_all_targets = []
        self.final_all_preds = []
        
        self.adapter_vec = []
        self.task_type_list = []
        self.class_group_list = []
        self.adapter_vec_label = []
        self.device = device
        self.global_class_stats = {k: {'total': 0, 'correct': 0} for k in range(self.num_classes)}
        
        self.cs = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # VCL Metrics
        self.accuracy_matrix = {}
        
        # Class prototypes for bias correction
        self.class_prototypes = {}
        
        # Per-class statistics for adaptive learning
        self.class_loss_history = defaultdict(list)
        self.class_acc_history = defaultdict(list)
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((args.class_num, args.domain_num))
            self.label_train_count = np.zeros((args.class_num))
            self.tanh = torch.nn.Tanh()
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 2

    def kl_div_stable(self, p, q):
        """KL divergence with numerical stability"""
        p = F.softmax(p, dim=1) + 1e-8
        q = F.softmax(q, dim=1) + 1e-8
        kl = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        return kl

    def weight_importance_regularization(self, model, task_id, lambda_ewc_base=100):
        """
        ADAPTIVE EWC: Much weaker, task-dependent
        - Task 1: Very weak (100) - allow learning new classes
        - Task 2+: Moderate (300-500) - balance stability/plasticity
        """
        if not hasattr(self, 'theta_star') or self.current_task == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Adaptive lambda based on task
        if task_id == 1:
            lambda_ewc = 100  # Very weak for Task 1
        elif task_id == 2:
            lambda_ewc = 300  # Moderate for Task 2
        else:
            lambda_ewc = 500  # Slightly stronger for later tasks
        
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in self.theta_star and 'head' not in name.lower():
                importance = self.omega[name].to(self.device)
                optimal_weight = self.theta_star[name].to(self.device)
                ewc_loss += torch.sum(importance * (param - optimal_weight) ** 2)
        
        return lambda_ewc * ewc_loss

    def l2_regularization(self, model, lambda_l2=0.005):
        """Lighter L2 regularization"""
        l2_loss = 0.0
        for name, param in model.named_parameters():
            if 'adapter' in name.lower() and param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        return lambda_l2 * l2_loss

    def compute_class_weights(self, task_id):
        """Compute per-class weights for balanced distillation"""
        weights = {}
        
        if task_id == 0 or not hasattr(self, 'class_acc_history'):
            return weights
        
        # Weight inversely proportional to recent accuracy
        for cls in range(self.num_classes):
            if cls in self.class_acc_history and len(self.class_acc_history[cls]) > 0:
                recent_acc = np.mean(self.class_acc_history[cls][-3:])  # Last 3 tasks
                # Classes with low accuracy get higher weight
                weights[cls] = max(0.5, 1.0 - recent_acc)
            else:
                weights[cls] = 1.0
        
        return weights

    def set_new_head(self, model, labels_to_be_added, task_id):
        """Add new nodes to head with proper initialization"""
        len_new_nodes = len(labels_to_be_added)
        
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id] * len_new_nodes))
        
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes).to(self.device)
        
        num_old_classes = prev_weight.shape[0]
        new_head.weight.data[:num_old_classes].copy_(prev_weight.data)
        new_head.bias.data[:num_old_classes].copy_(prev_bias.data)
        
        # Proper initialization
        nn.init.xavier_uniform_(new_head.weight.data[num_old_classes:])
        nn.init.constant_(new_head.bias.data[num_old_classes:], 0)
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added}). New head size: {new_head.weight.shape[0]}.")
        return new_head

    def inference_acc(self, model, data_loader, device):
        """Compute inference accuracy for dynamic head expansion"""
        model.eval()
        current_classes = self.class_mask[self.current_task]
        label_to_index = {label: i for i, label in enumerate(current_classes)}
        
        correct_pred_per_label = [0] * len(current_classes)
        num_instance_per_label = [0] * len(current_classes)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if batch_idx > 50:
                    break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                feature = model.forward_features(input)[:, 0]
                output = model.distill_head(feature)
                
                current_head_indices = []
                for label in current_classes:
                    indices = np.where(self.labels_in_head == label)[0]
                    if len(indices) > 0:
                        current_head_indices.append(indices[0])
                
                if len(current_head_indices) == 0:
                    continue
                
                current_head_indices = torch.tensor(current_head_indices, dtype=torch.long).to(device)
                
                all_head_indices = torch.arange(output.shape[-1], device=device)
                mask = torch.ones(output.shape[-1], dtype=torch.bool, device=device)
                mask[current_head_indices] = False
                irrelevant_indices = all_head_indices[mask]
                
                logits = output.clone()
                if len(irrelevant_indices) > 0:
                    logits[:, irrelevant_indices] = float('-inf')
                
                _, pred_index_in_head = torch.max(logits, 1)
                
                pred = torch.tensor([self.labels_in_head[i] for i in pred_index_in_head.cpu().numpy()],
                                   dtype=target.dtype).to(device)
                
                correct_predictions = (pred == target)
                
                for label_id in current_classes:
                    mask = (target == label_id)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    num_total_instance = torch.sum(mask)
                    
                    local_idx = label_to_index[label_id]
                    correct_pred_per_label[local_idx] += num_correct_pred.item()
                    num_instance_per_label[local_idx] += num_total_instance.item()
        
        accuracy_per_label = []
        for correct, num in zip(correct_pred_per_label, num_instance_per_label):
            acc = round(correct / num, 4) if num > 0 else 0.0
            accuracy_per_label.append(acc)
        
        model.train()
        return accuracy_per_label

    def detect_labels_to_be_added(self, inference_acc, thresholds=[]):
        """Detect which classes need additional head nodes - MORE AGGRESSIVE"""
        labels_with_low_accuracy = []
        
        # Lower threshold for more head expansion
        effective_threshold = 0.3 if not self.args.d_threshold else None
        
        if self.args.d_threshold:
            for label, acc, thre in zip(self.current_classes, inference_acc, thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else:
            for label, acc in zip(self.current_classes, inference_acc):
                if acc <= effective_threshold:
                    labels_with_low_accuracy.append(label)
        
        if len(labels_with_low_accuracy) > 0:
            print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy

    def train_one_epoch(self, model: torch.nn.Module,
                       criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, max_norm: float = 0,
                       set_training_mode=True, task_id=-1, class_mask=None, args=None):
        """
        BALANCED TRAINING: Adaptive regularization + Class-balanced distillation
        """
        torch.cuda.empty_cache()
        model.train(set_training_mode)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        header = f'Train Task {task_id}: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        # Pre-compute head indices
        current_task_classes = class_mask[task_id] if class_mask is not None else None
        current_head_indices_tensor = None
        
        if args.train_mask and current_task_classes is not None:
            current_head_indices = []
            for cls in current_task_classes:
                indices = np.where(self.labels_in_head == cls)[0]
                if len(indices) > 0:
                    current_head_indices.append(indices[0])
            
            if len(current_head_indices) > 0:
                current_head_indices_tensor = torch.tensor(current_head_indices, dtype=torch.long).to(device)
        
        # Get class weights for balanced distillation
        class_weights = self.compute_class_weights(task_id)
        
        optimizer.zero_grad()
        accumulation_counter = 0
        
        # Track per-class metrics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass
            feature = model.forward_features(input)[:, 0]
            output = model.distill_head(feature)
            
            # SAFE class masking
            if args.train_mask and current_head_indices_tensor is not None:
                max_idx = output.shape[1] - 1
                valid_indices = current_head_indices_tensor[current_head_indices_tensor <= max_idx]
                
                if len(valid_indices) > 0:
                    all_indices = torch.arange(output.shape[1], device=device)
                    mask = torch.ones(output.shape[1], dtype=torch.bool, device=device)
                    mask[valid_indices] = False
                    irrelevant_indices = all_indices[mask]
                    
                    masked_output = output.clone()
                    if len(irrelevant_indices) > 0:
                        masked_output[:, irrelevant_indices] = float('-inf')
                    
                    masked_output = torch.clamp(masked_output, min=-100, max=100)
                else:
                    masked_output = output
            else:
                masked_output = output
            
            # Task loss
            task_loss = criterion(masked_output, target)
            
            if not torch.isfinite(task_loss):
                print(f"WARNING: Non-finite task loss, skipping batch")
                continue
            
            loss = task_loss
            
            # BALANCED DISTILLATION (if not first task)
            if self.distill_model is not None and task_id > 0:
                with torch.no_grad():
                    teacher_feature = self.distill_model.forward_features(input)[:, 0]
                    teacher_output = self.distill_model.distill_head(teacher_feature)
                
                # Distill on old classes with class-specific weighting
                old_classes = [c for c in self.labels_in_head if c not in self.added_classes_in_cur_task]
                
                if len(old_classes) > 0:
                    old_indices = []
                    for cls in old_classes:
                        idx = np.where(self.labels_in_head == cls)[0]
                        if len(idx) > 0 and idx[0] < output.shape[1]:
                            old_indices.append(idx[0])
                    
                    if len(old_indices) > 0:
                        old_indices = torch.tensor(old_indices, dtype=torch.long).to(device)
                        valid_old_indices = old_indices[old_indices < output.shape[1]]
                        
                        if len(valid_old_indices) > 0:
                            student_logits_old = output.index_select(dim=1, index=valid_old_indices)
                            teacher_logits_old = teacher_output.index_select(dim=1, index=valid_old_indices)
                            
                            # Per-class weighted distillation
                            temp = 4.0
                            distill_loss = 0.0
                            
                            for i, cls_idx in enumerate(valid_old_indices):
                                cls_label = self.labels_in_head[cls_idx.item()]
                                weight = class_weights.get(cls_label, 1.0)
                                
                                cls_student = student_logits_old[:, i:i+1] / temp
                                cls_teacher = teacher_logits_old[:, i:i+1] / temp
                                
                                cls_distill = F.kl_div(
                                    F.log_softmax(cls_student, dim=0),
                                    F.softmax(cls_teacher, dim=0),
                                    reduction='batchmean'
                                ) * (temp ** 2) * weight
                                
                                distill_loss += cls_distill
                            
                            distill_loss = distill_loss / len(valid_old_indices)
                            
                            # Feature distillation
                            feature_loss = 1 - self.cs(feature, teacher_feature).mean()
                            
                            # Balanced distillation weights
                            if task_id == 1:
                                alpha_feat = 0.5  # Weak for Task 1
                                alpha_logit = 1.0
                            else:
                                alpha_feat = 1.0
                                alpha_logit = 2.0
                            
                            total_distill = alpha_feat * feature_loss + alpha_logit * distill_loss
                            
                            if torch.isfinite(total_distill):
                                loss += total_distill
            
            # ADAPTIVE EWC (task-dependent)
            if task_id > 0:
                ewc_loss = self.weight_importance_regularization(model, task_id)
                loss += ewc_loss
            
            # Lighter L2 regularization
            l2_loss = self.l2_regularization(model, lambda_l2=0.005)
            loss += l2_loss
            
            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite total loss, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Compute accuracy
            with torch.no_grad():
                acc1, acc5 = accuracy(masked_output, target, topk=(1, 5))
                
                # Track per-class accuracy
                _, preds = torch.max(masked_output, 1)
                for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                    class_total[t] += 1
                    if t == p:
                        class_correct[t] += 1
            
            # Backward with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            accumulation_counter += 1
            
            if accumulation_counter % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item() * self.gradient_accumulation_steps)
            metric_logger.update(Task_Loss=task_loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        # Final step if gradients remain
        if accumulation_counter % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Update class accuracy history
        for cls in class_total:
            acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            self.class_acc_history[cls].append(acc)
        
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader: Iterable,
                device: torch.device, task_id=-1, class_mask=None, args=None, flag_final_eval=False):
        
        criterion = torch.nn.CrossEntropyLoss()
        all_targets = []
        all_preds = []
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id)
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                feature = model.forward_features(input)[:, 0]
                output = model.distill_head(feature)
                
                # Bias correction using prototypes (if available)
                if hasattr(self, 'class_prototypes') and len(self.class_prototypes) > 0:
                    for cls, proto in self.class_prototypes.items():
                        cls_indices = np.where(self.labels_in_head == cls)[0]
                        if len(cls_indices) > 0 and cls_indices[0] < output.shape[1]:
                            # Boost logits for classes with prototypes
                            proto_similarity = F.cosine_similarity(
                                feature.unsqueeze(1), 
                                proto.unsqueeze(0).to(device), 
                                dim=-1
                            )
                            output[:, cls_indices[0]] += 0.5 * proto_similarity
                
                output = output.softmax(dim=1)
                
                loss = criterion(output.log(), target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                _, preds = torch.max(output, 1)
                
                all_targets.extend(target.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds,
                                                                   labels=np.unique(all_targets),
                                                                   average='macro', zero_division=0)
        
        metric_logger.synchronize_between_processes()
        
        print('* Acc@1 {top1:.3f} Loss {loss:.3f} Precision {precision:.3f} Recall {recall:.3f} F1 {f1:.3f}'
             .format(top1=metric_logger.meters['Acc@1'].global_avg,
                    loss=metric_logger.meters['Loss'].global_avg,
                    precision=precision, recall=recall, f1=f1))
        
        if flag_final_eval:
            self.final_all_targets.extend(all_targets.tolist())
            self.final_all_preds.extend(all_preds.tolist())
        
        print("Class-wise Accuracy:")
        class_correct = Counter()
        class_total = Counter()
        for t, p in zip(all_targets, all_preds):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
        
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
            print(f"Class {label}: {acc:.2%} ({class_correct[label]}/{class_total[label]})")
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, all_targets, all_preds

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader: list,
                         device: torch.device, task_id: int, class_mask: list, acc_matrix: np.ndarray,
                         args=None) -> dict:
        
        stat_matrix = np.zeros((3, args.num_tasks))
        flag_final_eval = (task_id == args.num_tasks - 1)
        
        if flag_final_eval:
            self.final_all_targets = []
            self.final_all_preds = []
        
        for i in range(task_id + 1):
            test_stats, all_targets, all_preds = self.evaluate(
                model=model,
                data_loader=data_loader[i]['val'],
                device=device,
                task_id=i,
                class_mask=class_mask,
                args=args,
                flag_final_eval=flag_final_eval
            )
            
            print(f"\nTesting on Task {i} (Domain {self.domain_list[i]}, Classes {self.class_mask[i]}):")
            
            acc_at_1 = test_stats['Acc@1']
            stat_matrix[0, i] = acc_at_1
            stat_matrix[1, i] = test_stats.get('Acc@5', 0.0)
            stat_matrix[2, i] = test_stats['Loss']
            acc_matrix[i, task_id] = acc_at_1
        
        self.accuracy_matrix[task_id] = {self.domain_list[j]: acc_matrix[j, task_id] for j in range(task_id + 1)}
        
        num_evaluated = task_id + 1
        avg_stat = np.divide(np.sum(stat_matrix[:, :num_evaluated], axis=1), num_evaluated)
        
        result_str = "[Average performance till task {}]    Acc@1: {:.4f}    Loss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[2])
        
        if task_id > 0:
            r_t_j = acc_matrix[:task_id+1, task_id]
            r_j_j = np.diag(acc_matrix)[:task_id+1]
            r_j_0 = acc_matrix[:task_id+1, 0]
            
            forgetting = np.mean(r_j_j - r_t_j)
            backward_transfer = np.mean(r_t_j[:task_id] - r_j_j[:task_id])
            forward_transfer = np.mean(r_t_j[1:] - r_j_0[1:])
            
            result_str += " Forgetting: {:.4f}    Backward: {:.4f}    Forward: {:.4f}".format(
                forgetting, backward_transfer, forward_transfer)
        
        print(result_str)
        
        if flag_final_eval:
            self.record_and_print_vcl_metrics(task_id=args.num_tasks - 1, acc_matrix=acc_matrix)
        
        return test_stats

    def record_and_print_vcl_metrics(self, task_id, acc_matrix):
        """Print final VCL metrics"""
        
        if not self.final_all_targets:
            print(f"No predictions available for cumulative results after Task {task_id}.")
            return
        
        y_true = np.array(self.final_all_targets)
        y_pred = np.array(self.final_all_preds)
        seen_classes = sorted(list(set(y_true)))
        
        cm = confusion_matrix(y_true, y_pred, labels=seen_classes)
        print(f"\n=== CUMULATIVE CONFUSION MATRIX (Tasks 0 to {task_id}) ===")
        print("Labels:", seen_classes)
        print(cm)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=seen_classes,
                                                                   average='macro', zero_division=0)
        
        print("\n=== CUMULATIVE METRICS ===")
        print(f"Total Accuracy: {np.mean(y_true == y_pred):.2%}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        
        print("\n=== CUMULATIVE PER-CLASS ACCURACY ===")
        for cls in seen_classes:
            correct = np.sum((y_true == cls) & (y_pred == cls))
            total = np.sum(y_true == cls)
            acc = correct / total if total > 0 else 0
            print(f"Class {cls}: {acc:.2%} ({correct}/{total})")
        
        if task_id > 0:
            T = task_id
            R_max_i = np.diag(acc_matrix)[:T]
            R_final_i = acc_matrix[:T, T]
            
            Avg_F = np.mean(R_max_i - R_final_i)
            BWT = np.mean(R_final_i - R_max_i)
            FWT = np.mean(np.diag(acc_matrix)[1:T+1] - acc_matrix[1:T+1, 0])
            
            print("\n=== Continual Learning Metrics (Matrix-Based) ===")
            print(f"Average Forgetting (Avg.F): {Avg_F:.4f}")
            print(f"Backward Transfer (BWT): {BWT:.4f}")
            print(f"Forward Transfer (FWT): {FWT:.4f}")

    def compute_class_prototypes(self, model, data_loader, device):
        """Compute class prototypes for bias correction"""
        model.eval()
        prototypes = {}
        counts = {}
        
        with torch.no_grad():
            for input, target in data_loader:
                input = input.to(device)
                target = target.to(device)
                
                features = model.forward_features(input)[:, 0]
                
                for feat, label in zip(features, target):
                    label_item = label.item()
                    if label_item not in prototypes:
                        prototypes[label_item] = torch.zeros_like(feat)
                        counts[label_item] = 0
                    prototypes[label_item] += feat
                    counts[label_item] += 1
        
        for label in prototypes:
            prototypes[label] /= counts[label]
        
        self.class_prototypes = prototypes
        model.train()
        return prototypes

    def compute_ewc_importance(self, model, data_loader_train):
        """
        Compute Fisher Information with moderate samples
        """
        model.eval()
        
        self.omega = {}
        self.theta_star = {}
        model.zero_grad()
        
        # Use moderate number of samples
        num_samples = min(len(data_loader_train.dataset), 2000)
        sample_count = 0
        
        for batch_idx, (input, target) in enumerate(data_loader_train):
            if sample_count >= num_samples:
                break
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            feature = model.forward_features(input)[:, 0]
            output = model.distill_head(feature)
            
            log_likelihood = F.log_softmax(output, dim=1).gather(1, target.unsqueeze(1)).mean()
            (-log_likelihood).backward()
            
            sample_count += input.shape[0]
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.omega[name] = param.grad.data.clone().pow(2) / sample_count
                self.theta_star[name] = param.data.clone()
        
        model.zero_grad()
        model.train()

    def flatten_parameters(self, modules):
        """Flatten parameters for adapter analysis"""
        flattened_params = []
        
        for m in modules:
            if isinstance(m, torch.nn.Module):
                params = list(m.parameters())
            else:
                params = [m]
            flattened_params.extend(params)
        
        return torch.cat([param.data.view(-1) for param in flattened_params if isinstance(param, torch.Tensor)])

    def cluster_adapters(self):
        """K-Means clustering on adapter shifts"""
        k = self.args.k
        if len(self.adapter_vec) > k:
            self.adapter_vec_array = torch.stack(self.adapter_vec).cpu().numpy().astype(np.float32)
            n_clusters = min(k, len(self.adapter_vec))
            self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=getattr(self.args, 'seed', 42))
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster (shifts) Assignments:", self.cluster_assignments)

    def pre_train_task(self, model, data_loader, device, task_id, args):
        """
        Task initialization with BALANCED adaptive LR
        """
        epsilon = 1e-8
        self.current_task = task_id
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        
        print(f"\n====================== STARTING TASK {task_id} ======================")
        print(f"Domain: {self.domain_list[task_id]} | Classes: {self.current_classes}")
        self.added_classes_in_cur_task = set()
        
        # Dynamic Head Expansion Logic - MORE AGGRESSIVE
        if self.class_group_train_count[self.current_class_group] > 0 and self.args.IC:
            self.distill_head = self.classifier_pool[self.current_class_group]
            
            self.current_classes = self.class_mask[task_id]
            inf_acc = self.inference_acc(model, data_loader, device)
            
            thresholds = []
            if self.args.d_threshold:
                count = self.class_group_train_count[self.current_class_group]
                if count > 0:
                    average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                else:
                    average_accs = np.full(len(self.current_classes), self.args.thre * 2)
                
                thresholds = self.args.gamma * (average_accs - inf_acc) / (average_accs + epsilon)
                thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                thresholds = [round(t, 2) if t > 0.15 else 0.15 for t in thresholds]  # Lower threshold
                print(f"Dynamic Thresholds: {thresholds}")
            
            labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
            
            if len(labels_to_be_added) > 0:
                new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                model.head = new_head
        
        # OPTIMIZED LEARNING RATE STRATEGY
        if task_id == 0:
            # Task 0: Base LR
            optimizer = utils.create_optimizer(args, model)
            print(f"Task 0: Base LR = {args.lr}")
        elif task_id == 1:
            # Task 1: HIGH LR for new classes
            args_copy = copy.deepcopy(args)
            args_copy.lr = args.lr * 2.5  # 2.5x boost
            args_copy.weight_decay = 0.00005  # Very light decay
            optimizer = utils.create_optimizer(args_copy, model)
            print(f"Task 1: Increased LR to {args_copy.lr} (2.5x boost for new classes)")
        elif task_id == 2:
            # Task 2: Higher LR for domain shift
            args_copy = copy.deepcopy(args)
            args_copy.lr = args.lr * 1.8  # 1.8x boost
            args_copy.weight_decay = 0.0001
            optimizer = utils.create_optimizer(args_copy, model)
            print(f"Task 2: Adjusted LR to {args_copy.lr} (domain shift)")
        else:
            # Task 3+: Moderate increase
            args_copy = copy.deepcopy(args)
            args_copy.lr = args.lr * 1.5
            args_copy.weight_decay = 0.0001
            optimizer = utils.create_optimizer(args_copy, model)
            print(f"Task {task_id}: Adjusted LR to {args_copy.lr}")
        
        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters).detach().clone()
        
        self.cur_domain = self.domain_list[task_id]
        
        # Task Type Determination
        if task_id == 0:
            self.task_type = "Initial"
            self.visited_domains.add(self.cur_domain)
        else:
            if task_id == 1:
                self.task_type = "CIL (Pure CI)"
            elif task_id == 2:
                self.task_type = "DIL (Pure DI)"
                self.visited_domains.add(self.cur_domain)
            elif task_id == 3:
                self.task_type = "VIL (Generalization)"
                self.visited_domains.add(self.cur_domain)
            else:
                if self.cur_domain not in self.visited_domains:
                    self.task_type = "DIL (New Domain)"
                    self.visited_domains.add(self.cur_domain)
                else:
                    self.task_type = "CIL (Domain Seen)"
        
        self.task_type_list.append(self.task_type)
        print(f"Task Type: {self.task_type}")
        
        return model, optimizer

    def post_train_task(self, model: torch.nn.Module, data_loader_train: Iterable, task_id: int):
        """
        Post-task operations: Compute prototypes, update pools, EWC
        """
        
        # Compute class prototypes for bias correction
        print("-> Computing class prototypes for bias correction...")
        self.compute_class_prototypes(model, data_loader_train, self.device)
        
        # Update Distillation Classifier Pool
        self.class_group_train_count[self.current_class_group] += 1
        self.classifier_pool[self.current_class_group] = copy.deepcopy(model.head)
        
        for c in self.classifier_pool:
            if c is not None:
                for p in c.parameters():
                    p.requires_grad = False
        
        # Record Adapter Shift
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector = self.cur_adapters - self.prev_adapters
        self.adapter_vec.append(vector.detach().cpu())
        self.adapter_vec_label.append(self.task_type)
        
        # Cluster Adapters
        self.cluster_adapters()
        
        # Compute EWC Importance
        if task_id < self.args.num_tasks - 1:
            print("-> Computing EWC importance (Omega) and storing optimal weights (Theta*)...")
            self.compute_ewc_importance(model, data_loader_train)
        
        # Store Teacher Model for Distillation
        if task_id < self.args.num_tasks - 1:
            print("-> Storing full model copy for feature distillation (LwF teacher)...")
            self.distill_model = copy.deepcopy(model)
            self.distill_model.eval()
            for param in self.distill_model.parameters():
                param.requires_grad = False
        
        print(f"================== TASK {task_id} COMPLETED ===================\n")

    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable,
                          optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device,
                          class_mask=None, args=None):
        
        num_tasks = args.num_tasks
        acc_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            if task_id > 0 and args.reinit_optimizer:
                optimizer = utils.create_optimizer(args, model)
            
            print(f"\n--- Starting Task {task_id}: ---")
            print(f"Domain: {self.domain_list[task_id]} | Classes: {self.class_mask[task_id]}")
            
            train_loader = data_loader[task_id]['train']
            
            model, optimizer = self.pre_train_task(model, train_loader, device, task_id, args)
            
            # Learning rate scheduler with warmup
            if lr_scheduler is None and args.epochs >= 10:
                from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
                
                # Warmup for first 10% of epochs
                warmup_epochs = max(1, args.epochs // 10)
                warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, 
                                           total_iters=warmup_epochs)
                
                # Cosine annealing for rest
                cosine_epochs = args.epochs - warmup_epochs
                cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
                
                lr_scheduler = SequentialLR(optimizer, 
                                           schedulers=[warmup_scheduler, cosine_scheduler],
                                           milestones=[warmup_epochs])
                print(f"Using Warmup ({warmup_epochs} epochs) + Cosine Annealing LR scheduler")
            
            for epoch in range(args.epochs):
                train_stats = self.train_one_epoch(
                    model=model, criterion=criterion,
                    data_loader=train_loader, optimizer=optimizer,
                    device=device, epoch=epoch, max_norm=args.clip_grad,
                    set_training_mode=True, task_id=task_id,
                    class_mask=class_mask, args=args,
                )
                
                if lr_scheduler:
                    lr_scheduler.step()
                
                # Clear cache periodically
                if epoch % 10 == 0:
                    torch.cuda.empty_cache()
                
                if args.output_dir and utils.is_main_process():
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
                    
                    # Save checkpoint every 10 epochs or at end
                    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                        Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                        checkpoint_path = os.path.join(args.output_dir, 
                                                      f'checkpoint/task{task_id+1}_epoch{epoch+1}.pth')
                        
                        state_dict = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'task_id': task_id,
                            'args': args,
                        }
                        if lr_scheduler is not None:
                            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                        
                        utils.save_on_master(state_dict, checkpoint_path)
                    
                    with open(os.path.join(args.output_dir, '{}_stats.txt'.format(
                        datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                        f.write(json.dumps(log_stats) + '\n')
            
            # Post-task operations
            self.post_train_task(model, data_loader_train=train_loader, task_id=task_id)
            
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1
            
            # Reset scheduler for next task
            lr_scheduler = None
            
            # Evaluation
            test_stats = self.evaluate_till_now(
                model=model, data_loader=data_loader, device=device,
                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix,
                args=args
            )
            
            # Clear cache after each task
            torch.cuda.empty_cache()
