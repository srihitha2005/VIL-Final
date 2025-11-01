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
# from sklearn.manifold import TSNE # Kept optional, uncomment if needed

from timm.utils import accuracy, ModelEmaV2
from timm.optim import create_optimizer
from timm.data import create_transform # Assuming needed for build_transform
# Assuming 'utils' is a local module containing MetricLogger, save_on_master, etc.
import utils
import random
from operator  import itemgetter
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
        
        # Initialize distill_head properly
        model.distill_head = nn.Linear(768, self.num_classes).to(device)
        
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
        
        # CRITICAL: Initialize replay buffer
        self.buffer_size = getattr(args, 'buffer_size', 500)
        self.replay_buffer = defaultdict(list)  # {class_id: [(score, (input, target, domain))]}
        self.seen_classes = set()
        self.buffer_per_class = self.buffer_size // self.num_classes
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((args.class_num, args.domain_num))
            self.label_train_count = np.zeros((args.class_num))
            self.tanh = torch.nn.Tanh()

    def _update_buffer_quota(self):
        """Update buffer quota per class based on seen classes"""
        num_seen = len(self.seen_classes)
        if num_seen > 0:
            self.buffer_per_class = self.buffer_size // num_seen
        print(f"Buffer quota per class updated: {self.buffer_per_class} (seen classes: {num_seen})")

    def _collect_buffer_samples(self, class_id, domain_id, scored_samples):
        """Collect samples into buffer with utility-based prioritization"""
        for score, sample_data in scored_samples:
            self.replay_buffer[class_id].append((score, sample_data))
        
        # Sort by score (descending) and keep top samples
        self.replay_buffer[class_id].sort(key=lambda x: x[0], reverse=True)
        self.replay_buffer[class_id] = self.replay_buffer[class_id][:self.buffer_per_class]

    def _get_replay_batch(self, replay_size):
        """Sample from replay buffer"""
        if not self.seen_classes or replay_size == 0:
            return None, None
        
        replay_inputs = []
        replay_targets = []
        
        samples_per_class = max(1, replay_size // len(self.seen_classes))
        
        for class_id in self.seen_classes:
            if class_id not in self.replay_buffer or len(self.replay_buffer[class_id]) == 0:
                continue
            
            # Sample from top-k buffer samples
            num_samples = min(samples_per_class, len(self.replay_buffer[class_id]))
            sampled_items = np.random.choice(len(self.replay_buffer[class_id]), num_samples, replace=False)
            
            for idx in sampled_items:
                _, (inp, tgt, _) = self.replay_buffer[class_id][idx]
                replay_inputs.append(inp)
                replay_targets.append(tgt)
        
        if len(replay_inputs) == 0:
            return None, None
        
        replay_inputs = torch.stack(replay_inputs).to(self.device)
        replay_targets = torch.stack(replay_targets).to(self.device)
        
        return replay_inputs, replay_targets

    @torch.no_grad()
    def compute_sample_score(self, model, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        
        with torch.no_grad():
            output = model(input.unsqueeze(0))
            prob = torch.softmax(output, dim=1)
        
        loss = F.cross_entropy(output, target.unsqueeze(0), reduction='none')
        max_confidence = torch.max(prob, dim=1).values
        
        sample_score = (loss.item() + (1.0 - max_confidence.item()))
        return sample_score

    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=1))
        return kl

    def weight_importance_regularization(self, model, lambda_ewc=200):
        ewc_loss = 0.0
        
        if not hasattr(self, 'theta_star') or self.current_task == 0:
            return torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            if name in self.theta_star:
                importance = self.omega[name].to(self.device)
                optimal_weight = self.theta_star[name].to(self.device)
                ewc_loss += torch.sum(importance * (param - optimal_weight) ** 2)
        
        return lambda_ewc * ewc_loss

    def set_new_head(self, model, labels_to_be_added, task_id):
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
        new_head.bias.data[num_old_classes:].zero_()
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added}). New head size: {new_head.weight.shape[0]}.")
        return new_head

    def inference_acc(self, model, data_loader, device):
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
                
                output = model(input)
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
        
        torch.cuda.empty_cache()
        model.train(set_training_mode)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        header = f'Train Task {task_id}: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        current_domain_id = self.domain_list[task_id] if task_id < len(self.domain_list) else None
        
        # CRITICAL: Determine replay size
        replay_ratio = getattr(args, 'replay_ratio', 0.5)
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break
            
            original_batch_size = input.size(0)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # CRITICAL: Add replay samples
            if task_id > 0:
                replay_size = int(original_batch_size * replay_ratio)
                replay_inputs, replay_targets = self._get_replay_batch(replay_size)
                
                if replay_inputs is not None:
                    input = torch.cat([input, replay_inputs], dim=0)
                    target = torch.cat([target, replay_targets], dim=0)
            
            # Forward Pass
            output = model(input)
            distill_loss = 0
            
            # Distillation Loss
            if self.distill_model is not None and task_id > 0:
                with torch.no_grad():
                    feature = model.forward_features(input)[:, 0]
                    teacher_feature = self.distill_model.forward_features(input)[:, 0]
                    
                    # Get teacher logits from distill_head
                    teacher_logits = self.distill_model.distill_head(teacher_feature)
                
                student_logits = model.distill_head(feature)
                
                # Only use old classes for distillation
                old_labels = [label for label in self.labels_in_head if label not in self.added_classes_in_cur_task]
                
                if len(old_labels) > 0:
                    old_label_indices = torch.tensor([np.where(self.labels_in_head == label)[0][0]
                                                     for label in old_labels], dtype=torch.long).to(device)
                    
                    student_logits_old = student_logits.index_select(dim=1, index=old_label_indices)
                    teacher_logits_old = teacher_logits.index_select(dim=1, index=old_label_indices)
                    
                    logit_distill_loss = F.kl_div(
                        F.log_softmax(student_logits_old / args.distill_temp, dim=1),
                        F.softmax(teacher_logits_old / args.distill_temp, dim=1),
                        reduction='batchmean'
                    ) * (args.distill_temp ** 2)
                    
                    feature_loss = 1 - self.cs(feature, teacher_feature).mean()
                    distill_loss = args.alpha_feat * feature_loss + args.alpha_logit * logit_distill_loss
            
            # Task Loss with proper masking
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask_tensor = torch.tensor(not_mask, dtype=torch.long).to(device)
                
                # Use distill_head for loss calculation
                logits = model.distill_head(model.forward_features(input)[:, 0])
                logits = logits.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))
            else:
                logits = model.distill_head(model.forward_features(input)[:, 0])
            
            task_loss = criterion(logits, target)
            loss = task_loss
            
            # Add distillation loss
            if self.args.IC and distill_loss > 0:
                loss += distill_loss
            
            # EWC Regularization
            lambda_ewc = 200
            ewc_loss = self.weight_importance_regularization(model, lambda_ewc)
            loss += ewc_loss
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            # CRITICAL: Buffer collection (only for new samples)
            if current_domain_id is not None:
                for i in range(original_batch_size):
                    class_id = target[i].item()
                    score = self.compute_sample_score(model, input[i], target[i])
                    
                    sample_data = (input[i].detach().cpu(), target[i].detach().cpu(), current_domain_id)
                    scored_sample = (score, sample_data)
                    
                    if class_id not in self.seen_classes:
                        self.seen_classes.add(class_id)
                        self._update_buffer_quota()
                    
                    self._collect_buffer_samples(class_id, current_domain_id, [scored_sample])
            
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

    def aggregate_dynamic_head(self, output, slice_output=True):
        """Aggregate dynamic head logits back to fixed num_classes"""
        aggregated_output = torch.full((output.size(0), self.num_classes), float('-inf'), device=output.device)
        
        for label in range(self.num_classes):
            label_nodes = np.where(self.labels_in_head == label)[0]
            
            if len(label_nodes) > 0:
                class_logits = output.index_select(dim=1, index=torch.tensor(label_nodes, device=output.device))
                max_logits, _ = torch.max(class_logits, dim=1)
                aggregated_output[:, label] = max_logits
        
        return aggregated_output

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
                
                # Use distill_head for evaluation
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
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
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
        """Calculate Fisher Information Matrix for EWC"""
        model.eval()
        
        self.omega = {}
        self.theta_star = {}
        model.zero_grad()
        
        num_samples = min(len(data_loader_val.dataset), 1000)
        
        for batch_idx, (input, target) in enumerate(data_loader_val):
            if batch_idx * data_loader_val.batch_size > num_samples:
                break
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            feature = model.forward_features(input)[:, 0]
            output = model.distill_head(feature)
            
            log_likelihood = F.log_softmax(output, dim=1).gather(1, target.unsqueeze(1)).mean()
            (-log_likelihood).backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.omega[name] = param.grad.data.clone().pow(2)
                self.theta_star[name] = param.data.clone()
        
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
        """Initialize task variables and setup"""
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
        """Finalize task: update pools, compute EWC, cluster adapters"""
        
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
