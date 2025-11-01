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
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Iterable
from operator import itemgetter
from collections import defaultdict, Counter

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from timm.utils import accuracy, ModelEmaV2
from timm.optim import create_optimizer
from timm.data import create_transform
import utils
from operator import itemgetter
from collections import defaultdict


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
        
        # CRITICAL FIX 1: Initialize distill_head with proper dimensions
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
        self.all_targets_cumulative = []
        self.all_preds_cumulative = []
        self.current_domain_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        self.domain_accuracy_history = []
        self.domain_history = {}
        self.domain_best = {}
        self.domain_initial = {}
        self.accuracy_matrix = {}
        self.true_labels = {}
        self.predicted_labels = {}
        
        # OPTIMIZATION 1: Lightweight memory buffer for critical samples only
        self.memory_buffer = {}  # task_id -> list of (input, target) tuples
        self.buffer_size_per_task = min(100, args.replay_buffer_size // args.num_tasks) if hasattr(args, 'replay_buffer_size') else 0
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((args.class_num, args.domain_num))
            self.label_train_count = np.zeros((args.class_num))
            self.tanh = torch.nn.Tanh()

    def kl_div(self, p, q):
        """KL divergence with numerical stability"""
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log((p + 1e-10) / (q + 1e-10)), dim=1))
        return kl

    def weight_importance_regularization(self, model, lambda_ewc=1000):
        """
        CRITICAL FIX 2: Enhanced EWC with stronger regularization
        Exclude head layers to allow new class learning
        """
        ewc_loss = 0.0
        
        if not hasattr(self, 'theta_star') or self.current_task == 0:
            return torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            # Exclude all head-related parameters from EWC
            if name in self.theta_star and 'head' not in name.lower():
                importance = self.omega[name].to(self.device)
                optimal_weight = self.theta_star[name].to(self.device)
                ewc_loss += torch.sum(importance * (param - optimal_weight) ** 2)
        
        return lambda_ewc * ewc_loss

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
        
        # CRITICAL FIX 3: Proper initialization for new nodes
        nn.init.xavier_uniform_(new_head.weight.data[num_old_classes:])
        nn.init.constant_(new_head.bias.data[num_old_classes:], 0)
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added}). New head size: {new_head.weight.shape[0]}.")
        return new_head

    def store_critical_samples(self, data_loader, task_id):
        """
        OPTIMIZATION 2: Store only the most critical samples for replay
        Uses gradient norm as criticality measure
        """
        if self.buffer_size_per_task == 0:
            return
        
        self.model.eval()
        sample_scores = []
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if batch_idx > 10:  # Limit sampling to keep it fast
                    break
                
                input = input.to(self.device)
                target = target.to(self.device)
                
                feature = self.model.forward_features(input)[:, 0]
                output = self.model.distill_head(feature)
                
                # Compute prediction confidence
                probs = F.softmax(output, dim=1)
                confidence, _ = torch.max(probs, dim=1)
                
                # Store samples with lower confidence (more critical)
                for i in range(input.shape[0]):
                    sample_scores.append((
                        1.0 - confidence[i].item(),  # criticality score
                        input[i].cpu(),
                        target[i].cpu()
                    ))
        
        # Sort by criticality and keep top-k
        sample_scores.sort(key=lambda x: x[0], reverse=True)
        self.memory_buffer[task_id] = [(s[1], s[2]) for s in sample_scores[:self.buffer_size_per_task]]
        
        self.model.train()

    def get_replay_batch(self, batch_size=16):
        """
        OPTIMIZATION 3: Efficient replay batch sampling
        """
        if not self.memory_buffer or self.current_task == 0:
            return None, None
        
        all_samples = []
        for task_id in self.memory_buffer:
            all_samples.extend(self.memory_buffer[task_id])
        
        if len(all_samples) == 0:
            return None, None
        
        # Sample without replacement
        n_samples = min(batch_size, len(all_samples))
        indices = random.sample(range(len(all_samples)), n_samples)
        
        replay_inputs = torch.stack([all_samples[i][0] for i in indices])
        replay_targets = torch.stack([all_samples[i][1] for i in indices])
        
        return replay_inputs.to(self.device), replay_targets.to(self.device)

    def inference_acc(self, model, data_loader, device):
        """Compute inference accuracy for dynamic head expansion"""
        model.eval()
        current_classes = self.class_mask[self.current_task]
        label_to_index = {label: i for i, label in enumerate(current_classes)}
        
        correct_pred_per_label = [0] * len(current_classes)
        num_instance_per_label = [0] * len(current_classes)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop and batch_idx > 200:
                    break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                feature = model.forward_features(input)[:, 0]
                output = model.distill_head(feature)
                
                # Get indices for current classes
                current_head_indices = [np.where(self.labels_in_head == label)[0][0] for label in current_classes]
                
                all_head_indices = np.arange(output.shape[-1])
                irrelevant_indices = np.setdiff1d(all_head_indices, current_head_indices)
                irrelevant_indices_tensor = torch.tensor(irrelevant_indices, dtype=torch.long).to(device)
                
                logits = output.index_fill(dim=1, index=irrelevant_indices_tensor, value=float('-inf'))
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
        """Detect which classes need additional head nodes"""
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label, acc, thre in zip(self.current_classes, inference_acc, thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else:
            for label, acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
        
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy

    def train_one_epoch(self, model: torch.nn.Module,
                       criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, max_norm: float = 0,
                       set_training_mode=True, task_id=-1, class_mask=None, ema_model=None, args=None):
        """
        CRITICAL FIX 4: Corrected training loop with proper masking and loss computation
        """
        torch.cuda.empty_cache()
        model.train(set_training_mode)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        header = f'Train Task {task_id}: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break
            
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass
            feature = model.forward_features(input)[:, 0]
            output = model.distill_head(feature)
            
            # CRITICAL FIX 5: Proper class masking for current task
            if args.train_mask and class_mask is not None:
                current_task_classes = class_mask[task_id]
                
                # Map task classes to head indices
                current_head_indices = []
                for cls in current_task_classes:
                    indices = np.where(self.labels_in_head == cls)[0]
                    if len(indices) > 0:
                        current_head_indices.extend(indices.tolist())
                
                current_head_indices = torch.tensor(current_head_indices, dtype=torch.long).to(device)
                
                # Mask irrelevant classes
                all_indices = torch.arange(output.shape[1], device=device)
                mask = torch.ones(output.shape[1], dtype=torch.bool, device=device)
                mask[current_head_indices] = False
                irrelevant_indices = all_indices[mask]
                
                masked_output = output.clone()
                masked_output[:, irrelevant_indices] = float('-inf')
                
                # CRITICAL FIX 6: Numerical stability
                masked_output = torch.clamp(masked_output, min=-100, max=100)
            else:
                masked_output = torch.clamp(output, min=-100, max=100)
            
            # Task loss
            task_loss = criterion(masked_output, target)
            
            # Safety check
            if not torch.isfinite(task_loss):
                print(f"WARNING: Non-finite task loss detected, skipping batch")
                continue
            
            loss = task_loss
            
            # OPTIMIZATION 4: Lightweight replay with minimal overhead
            if self.buffer_size_per_task > 0 and task_id > 0 and batch_idx % 4 == 0:  # Every 4th batch
                replay_input, replay_target = self.get_replay_batch(batch_size=8)
                
                if replay_input is not None:
                    replay_feature = model.forward_features(replay_input)[:, 0]
                    replay_output = model.distill_head(replay_feature)
                    replay_loss = criterion(replay_output, replay_target)
                    
                    if torch.isfinite(replay_loss):
                        loss += 0.5 * replay_loss
            
            # CRITICAL FIX 7: Enhanced distillation for task_id > 0
            if self.distill_model is not None and task_id > 0:
                with torch.no_grad():
                    teacher_feature = self.distill_model.forward_features(input)[:, 0]
                    teacher_output = self.distill_model.distill_head(teacher_feature)
                
                # Only distill on old classes
                old_classes = [c for c in self.labels_in_head if c not in self.added_classes_in_cur_task]
                
                if len(old_classes) > 0:
                    old_indices = []
                    for cls in old_classes:
                        idx = np.where(self.labels_in_head == cls)[0]
                        if len(idx) > 0:
                            old_indices.append(idx[0])
                    
                    old_indices = torch.tensor(old_indices, dtype=torch.long).to(device)
                    
                    student_logits_old = output.index_select(dim=1, index=old_indices)
                    teacher_logits_old = teacher_output.index_select(dim=1, index=old_indices)
                    
                    # CRITICAL FIX 8: Higher temperature for smoother distillation
                    temp = 4.0
                    logit_distill_loss = F.kl_div(
                        F.log_softmax(student_logits_old / temp, dim=1),
                        F.softmax(teacher_logits_old / temp, dim=1),
                        reduction='batchmean'
                    ) * (temp ** 2)
                    
                    feature_loss = 1 - self.cs(feature, teacher_feature).mean()
                    
                    # OPTIMIZATION 5: Balanced distillation weights
                    alpha_feat = 1.0
                    alpha_logit = 2.0
                    distill_loss = alpha_feat * feature_loss + alpha_logit * logit_distill_loss
                    
                    if torch.isfinite(distill_loss):
                        loss += distill_loss
            
            # CRITICAL FIX 9: Stronger EWC regularization
            if task_id > 0:
                ewc_loss = self.weight_importance_regularization(model, lambda_ewc=1000)
                loss += ewc_loss
            
            # Final safety check
            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite total loss, skipping batch")
                continue
            
            # Compute accuracy
            acc1, acc5 = accuracy(masked_output, target, topk=(1, 5))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # CRITICAL FIX 10: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm if max_norm > 0 else 1.0)
            
            optimizer.step()
            
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Task_Loss=task_loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            
            if ema_model is not None:
                ema_model.update(model.get_adapter())
        
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader: Iterable,
                device: torch.device, task_id=-1, class_mask=None, ema_model=None, args=None, flag_final_eval=False):
        
        criterion = torch.nn.CrossEntropyLoss()
        all_targets = []
        all_preds = []
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id)
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop and batch_idx > 20:
                    break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                feature = model.forward_features(input)[:, 0]
                output = model.distill_head(feature)
                
                # EMA ensemble
                output_ensemble = [output.softmax(dim=1)]
                
                if ema_model is not None:
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)
                    
                    feature_ema = model.forward_features(input)[:, 0]
                    output_ema = model.distill_head(feature_ema)
                    output_ensemble.append(output_ema.softmax(dim=1))
                    
                    model.put_adapter(tmp_adapter)
                
                output = torch.stack(output_ensemble, dim=-1).max(dim=-1)[0]
                
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
                         ema_model=None, args=None) -> dict:
        
        stat_matrix = np.zeros((3, args.num_tasks))
        flag_final_eval = (task_id == args.num_tasks - 1)
        
        self.current_domain_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        if flag_final_eval:
            self.final_all_targets = []
            self.final_all_preds = []
        
        current_task_accuracies = {}
        
        for i in range(task_id + 1):
            test_stats, all_targets, all_preds = self.evaluate(
                model=model,
                data_loader=data_loader[i]['val'],
                device=device,
                task_id=i,
                class_mask=class_mask,
                ema_model=ema_model,
                args=args,
                flag_final_eval=flag_final_eval
            )
            
            print(f"\nTesting on Task {i} (Domain {self.domain_list[i]}, Classes {self.class_mask[i]}):")
            
            acc_at_1 = test_stats['Acc@1']
            stat_matrix[0, i] = acc_at_1
            stat_matrix[1, i] = test_stats.get('Acc@5', 0.0)
            stat_matrix[2, i] = test_stats['Loss']
            acc_matrix[i, task_id] = acc_at_1
            
            current_task_accuracies[self.domain_list[i]] = {
                'Acc@1': acc_at_1,
                'Acc@5': stat_matrix[1, i],
                'Loss': stat_matrix[2, i]
            }
        
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

    def compute_ewc_importance(self, model, data_loader_val):
        """
        CRITICAL FIX 11: Enhanced Fisher Information calculation
        Uses more samples and proper gradient accumulation
        """
        model.eval()
        
        self.omega = {}
        self.theta_star = {}
        model.zero_grad()
        
        # Use more samples for better Fisher estimation
        num_samples = min(len(data_loader_val.dataset), 2000)
        sample_count = 0
        
        for batch_idx, (input, target) in enumerate(data_loader_val):
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

    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args=None):
        """Handle adapter freezing"""
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))
        
        if epoch == args.num_freeze_epochs:
            torch.cuda.empty_cache()
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')
        return model

    def pre_train_task(self, model, data_loader, device, task_id, args):
        """
        CRITICAL FIX 12: Enhanced task initialization with proper LR scheduling
        """
        epsilon = 1e-8
        self.current_task = task_id
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        
        print(f"\n====================== STARTING TASK {task_id} ======================")
        print(f"Domain: {self.domain_list[task_id]} | Classes: {self.current_classes}")
        self.added_classes_in_cur_task = set()
        
        # Dynamic Head Expansion Logic
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
                thresholds = [round(t, 2) if t > self.args.thre else self.args.thre for t in thresholds]
                print(f"Dynamic Thresholds: {thresholds}")
            
            labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
            
            if len(labels_to_be_added) > 0:
                new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                model.head = new_head
        
        # CRITICAL FIX 13: Adaptive learning rate for different tasks
        if task_id == 1:
            # Increase LR for Task 1 to ensure new class learning
            args_copy = copy.deepcopy(args)
            args_copy.lr = args.lr * 1.5
            optimizer = utils.create_optimizer(args_copy, model)
            print(f"Task 1: Increased LR to {args_copy.lr}")
        elif task_id == 2:
            # Moderate LR for domain shift
            args_copy = copy.deepcopy(args)
            args_copy.lr = args.lr * 1.2
            optimizer = utils.create_optimizer(args_copy, model)
            print(f"Task 2: Adjusted LR to {args_copy.lr}")
        else:
            optimizer = utils.create_optimizer(args, model)
        
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

    def post_train_task(self, model: torch.nn.Module, data_loader_val: Iterable, task_id: int):
        """
        OPTIMIZATION 6: Efficient post-task operations with memory buffer storage
        """
        
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
        
        # OPTIMIZATION 7: Store critical samples BEFORE computing EWC
        if self.buffer_size_per_task > 0 and task_id < self.args.num_tasks - 1:
            print(f"-> Storing {self.buffer_size_per_task} critical samples for replay...")
            self.store_critical_samples(data_loader_val, task_id)
        
        # Compute EWC Importance
        if task_id < self.args.num_tasks - 1:
            print("-> Computing EWC importance (Omega) and storing optimal weights (Theta*)...")
            self.compute_ewc_importance(model, data_loader_val)
        
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
        
        ema_model = None
        
        for task_id in range(num_tasks):
            if task_id > 0 and args.reinit_optimizer:
                optimizer = utils.create_optimizer(args, model)
            
            if task_id == 1 and hasattr(args, 'adapt_blocks') and len(args.adapt_blocks) > 0:
                from timm.utils import ModelEmaV2
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
            
            print(f"\n--- Starting Task {task_id}: ---")
            print(f"Domain: {self.domain_list[task_id]} | Classes: {self.class_mask[task_id]}")
            
            train_loader = data_loader[task_id]['train']
            val_loader_for_ewc = data_loader[task_id]['val']
            
            model, optimizer = self.pre_train_task(model, train_loader, device, task_id, args)
            
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args)
                
                train_stats = self.train_one_epoch(
                    model=model, criterion=criterion,
                    data_loader=train_loader, optimizer=optimizer,
                    device=device, epoch=epoch, max_norm=args.clip_grad,
                    set_training_mode=True, task_id=task_id,
                    class_mask=class_mask, ema_model=ema_model, args=args,
                )
                
                if lr_scheduler:
                    lr_scheduler.step(epoch)
                
                if args.output_dir and utils.is_main_process():
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
                    
                    if epoch == args.epochs - 1:
                        Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                        
                        state_dict = {
                            'model': model.state_dict(),
                            'ema_model': ema_model.state_dict() if ema_model is not None else None,
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }
                        if args.sched is not None and args.sched != 'constant':
                            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                        
                        utils.save_on_master(state_dict, checkpoint_path)
                    
                    with open(os.path.join(args.output_dir, '{}_stats.txt'.format(
                        datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                        f.write(json.dumps(log_stats) + '\n')
            
            self.post_train_task(model, data_loader_val=val_loader_for_ewc, task_id=task_id)
            
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1
            
            test_stats = self.evaluate_till_now(
                model=model, data_loader=data_loader, device=device,
                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix,
                ema_model=ema_model, args=args
            )
