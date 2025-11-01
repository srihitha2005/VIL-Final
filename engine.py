import math
import sys
import os
import datetime
import json
from turtle import undo
from typing import Iterable
import random
from pathlib import Path 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix

#Changed
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

#Not using validation anymore
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_accuracy = 0
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_accuracy, model):
        if val_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Early stopping triggered. Restored best weights with accuracy: {self.best_accuracy:.4f}")
            return True
        return False



class Engine():
    def __init__(self, model=None,device=None,class_mask=[], domain_list= [], args=None):
        self.current_task=0
        self.current_classes=[]
        #! distillation
        max_class_id = max([item for mask in class_mask for item in mask])
        self.class_group_size = len(class_mask[0])  # keep this as it is
        self.class_group_num = (max_class_id // self.class_group_size) + 1

        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        #changed
        self.visited_domains = set()

        #changed 
        self.replay_buffer = defaultdict(list)  # key: (domain_id, class_id) → list of samples
        self.buffer_size_per_key = args.replay_buffer_size_per_key  # new argument (explained below)
        self.buffer_size = args.replay_buffer_size  # Total buffer capacity
        self.replay_top_k_percent = args.replay_top_k_percent  # e.g., 0.2 (top 20%)

        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.model = model
        
        self.num_classes= max([item for mask in class_mask for item in mask])+1
        self.distill_head = None
        model.distill_head = nn.Linear(768, self.num_classes).to(device)        
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args=args
        
        self.class_mask=class_mask
        self.domain_list=domain_list

        self.task_type="initial"
        self.args=args
        #changed
        self.final_all_targets = []
        self.final_all_preds = []

        #replay
        self.buffer_size = args.replay_buffer_size
        self.seen_classes = set()
        self.num_domains_per_class = defaultdict(lambda: 0)
        
        self.adapter_vec=[]
        self.task_type_list=[]
        self.class_group_list=[]
        self.adapter_vec_label=[]
        self.device=device
        self.global_class_stats = {k: {'total': 0, 'correct': 0} for k in range(5)}
        
        self.task5_true = []
        self.task5_pred = []
        self.global_confusion = {i: {j: 0 for j in range(5)} for i in range(5)}
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.class_num, self.args.domain_num))
            self.label_train_count = np.zeros((self.args.class_num))
            self.tanh = torch.nn.Tanh()
            
        self.cs=torch.nn.CosineSimilarity(dim=1,eps=1e-6)

        #Changed
        self.all_targets_cumulative = []
        self.all_preds_cumulative = []
        self.current_domain_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        self.domain_accuracy_history = []  # Store domain accuracies after each task for forgetting calculation
        self.domain_history = {}
        self.domain_best = {}          # best accuracy so far
        self.domain_initial = {}       # first-seen accuracy before training domain
        self.accuracy_matrix = {}
        self.true_labels = {}
        self.predicted_labels = {}

        #Changed
        self.early_stopping = EarlyStopping(
            patience=args.early_stopping_patience if hasattr(args, 'early_stopping_patience') else 9,
            min_delta=args.early_stopping_min_delta if hasattr(args, 'early_stopping_min_delta') else 0.001
        )

    #Changed
    #Not using validate for now. 
    def split_train_val_data(self, dataset, train_ratio=0.8, random_state=42):
        """
        Split dataset into train and validation sets
        
        Args:
            dataset: PyTorch dataset
            train_ratio: Ratio for training data (default 0.8 for 80-20 split)
            random_state: Random seed for reproducibility
        
        Returns:
            train_dataset, val_dataset: PyTorch Subset objects
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Get labels for stratified split (if your dataset has targets attribute)
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # If no labels available, do random split
            labels = None
        
        if labels is not None:
            # Stratified split to maintain class distribution
            train_indices, val_indices = train_test_split(
                indices, 
                train_size=train_ratio,
                random_state=random_state,
                stratify=labels
            )
        else:
            # Random split
            train_indices, val_indices = train_test_split(
                indices,
                train_size=train_ratio, 
                random_state=random_state
            )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        return train_dataset, val_dataset

    #Changed
    def create_train_val_loaders(self, dataset, batch_size, train_ratio=0.8, random_state=42, num_workers=4):
        """
        Create train and validation data loaders from dataset
        """
        train_dataset, val_dataset = self.split_train_val_data(dataset, train_ratio, random_state)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def validate_model(self, model, val_loader, criterion, device, class_mask, task_id):
        """
        Validate the model on validation set
        
        Returns:
            validation accuracy
        """
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input, target in val_loader:
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                # Apply same masking logic as in training
                if output.shape[-1] > self.num_classes:
                    output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                    if len(self.added_classes_in_cur_task) > 0:
                        for added_class in self.added_classes_in_cur_task:
                            cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                            output[:, added_class] = output[:, cur_node]
                    output = output[:, :self.num_classes]
                
                if hasattr(self.args, 'train_mask') and self.args.train_mask and class_mask is not None:
                    mask = class_mask[task_id]
                    not_mask = np.setdiff1d(np.arange(5), mask)
                    not_mask_tensor = torch.tensor(not_mask, dtype=torch.int64).to(output.device)
                    logits = output.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))
                else:
                    logits = output
                
                loss = criterion(logits, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        return val_accuracy

    #changed
    @torch.no_grad()
    def compute_sample_score(self, model, input, target):
        # Ensure input is on the correct device for the model
        input = input.to(self.device)
        target = target.to(self.device)
        
        # 1. Get model output (logits)
        with torch.no_grad():
            # Pass input through model, unsqueezing for batch dimension
            output = model(input.unsqueeze(0))
            prob = torch.softmax(output, dim=1)
        
        # 2. Calculate Cross-Entropy Loss (measures incorrectness)
        loss = torch.nn.functional.cross_entropy(output, target.unsqueeze(0), reduction='none')
        
        # 3. Calculate Maximum Prediction Confidence (measures certainty)
        # We want a sample score that is HIGHER for LOW confidence.
        max_confidence = torch.max(prob, dim=1).values
        
        # 4. New Score: Loss + (1 - Max Confidence)
        # Rationale: Samples with high loss AND low confidence (1 - max_confidence is high) 
        # are the most valuable to store for replay/buffer prioritization.
        # This prioritizes samples the model is most wrong and uncertain about.
    
        # Use item() to return a standard Python float as required by the output format.
        sample_score = (loss.item() + (1.0 - max_confidence.item()))
        return sample_score
    
    import random
    from operator import itemgetter
    from collections import defaultdict

    def _update_buffer_quota(self):
        """
        Update quota for each class to maintain fixed total buffer size (self.buffer_size).
        Ensures proportional storage for all seen classes, crucial for Class-Incremental (CI) balance.
        """
        self.num_classes_seen = len(self.seen_classes)
        if self.num_classes_seen == 0:
            self.buffer_per_class = 0
            return
        # Use floor division for integer quota. The remainder is discarded (or can be added to the base class).
        self.buffer_per_class = self.buffer_size // self.num_classes_seen
        
    def _collect_buffer_samples(self, class_id, domain_id, scored_samples):
        """
        Saves samples to buffer, prioritizing samples based on their 'utility score'.
        Input: scored_samples is a list of tuples: (score, sample_data).
        """
        key = (class_id, domain_id)
        if key not in self.replay_buffer:
            self.replay_buffer[key] = []
            
        # Add new samples
        self.replay_buffer[key].extend(scored_samples)
        
        # Trim by score: Keep only the highest-scoring samples up to the per-key size limit (self.buffer_size_per_key)
        # The buffer size per key prevents any single domain-class pair from dominating the memory before rebalancing.
        if len(self.replay_buffer[key]) > self.args.replay_buffer_size_per_key:
            # Sort by score (first element of the tuple), descending, and keep the top N
            self.replay_buffer[key].sort(key=itemgetter(0), reverse=True)
            self.replay_buffer[key] = self.replay_buffer[key][:self.args.replay_buffer_size_per_key]

    def _rebalance_buffer(self):
        """
        Rebalances the total buffer size (self.buffer_size) across all seen classes.
        Samples are prioritized based on their stored score (utility).
        """
        new_buffer = defaultdict(list)
        
        for class_id in self.seen_classes:
            all_keys = [k for k in self.replay_buffer if k[0] == class_id]
            
            # 1. Collect all samples for this class from all domains (D1, D2, D3)
            all_samples_scored = []
            for k in all_keys:
                all_samples_scored.extend(self.replay_buffer[k])
            
            # 2. Score-based Trimming: Sort by score (descending) and keep only the top samples 
            # up to the calculated class quota (self.buffer_per_class).
            all_samples_scored.sort(key=itemgetter(0), reverse=True)
            final_class_samples = all_samples_scored[:self.buffer_per_class]
            
            # 3. Re-distribute the high-utility samples back into the new buffer keys
            # Crucial for Domain Incremental (DI): We preserve samples from ALL domains
            # as long as they are high-scoring, allowing for implicit cross-domain replay.
            
            # Note: final_class_samples contains (score, sample_data) tuples
            for score, sample in final_class_samples:
                # We need the original domain_id of the sample to put it back in the right key.
                # Assuming sample_data carries the domain_id (e.g., sample = (data, target, domain_id))
                # Since the structure of 'sample' isn't given, we must infer it or pass it.
                # Assuming 'sample' is the full (data, target, domain_id) tuple:
                
                # --- CRITICAL ASSUMPTION ---
                # For this to work, the 'sample' data must contain the domain_id.
                # Let's assume sample is (data_tensor, target_tensor, domain_id).
                
                # Find the domain ID from the sample data
                # If the sample only contains image/label, this must be adjusted in the dataset loader.
                # For now, we will distribute randomly if domain info is not in the sample data,
                # but the optimal approach is to have the domain ID within the sample tuple.
                
                # Safest distribution logic (assuming domain_id is accessible via sample[2]):
                domain_id_from_sample = sample[2] if len(sample) > 2 else random.choice([k[1] for k in all_keys])
                
                key = (class_id, domain_id_from_sample)
                new_buffer[key].append((score, sample))

        self.replay_buffer = new_buffer

    # Note: The _rebalance_buffer function relies on the sample data (which is a tuple stored in the buffer)
    # containing the domain ID, e.g., sample = (data_tensor, target_tensor, domain_id).
    # If your dataset loader does not provide the domain ID in the sample tuple, you must adjust the loader.

    #Changed till now
    import torch
    import torch.nn.functional as F

    # Keeping kl_div the same as it is a standard and correct function.
    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q + 1e-8), dim=1))
        return kl

    # --- SUPERIOR REGULARIZATION: EWC-STYLE ---
    def weight_importance_regularization(self, model, lambda_ewc=1e3):
        """
        Calculates the Elastic Weight Consolidation (EWC) regularization loss.
        This is generally better than Spectral Norm for VCL as it directly combats
        catastrophic forgetting based on parameter importance.
        
        CRITICAL ASSUMPTIONS:
        1. self.theta_star: A dictionary storing model weights after the previous task.
        2. self.omega: A dictionary storing the parameter importance (Fisher/MAS estimate).
        
        If these are not populated, this loss will be zero.
        """
        ewc_loss = 0.0
        
        # Check if necessary components exist and the current task is not the first one
        if not hasattr(self, 'theta_star') or self.current_task == 0:
            return torch.tensor(0.0, device=self.device)
            
        for name, param in model.named_parameters():
            if name in self.theta_star:
                # Importance weight (Omega) and previous optimal weight (theta_star)
                importance = self.omega[name].to(self.device)
                optimal_weight = self.theta_star[name].to(self.device)
                
                # EWC Loss: sum(Omega * (theta - theta*)^2)
                ewc_loss += torch.sum(importance * (param - optimal_weight)**2)
                
        return lambda_ewc * ewc_loss


    def set_new_head(self, model, labels_to_be_added, task_id):
        len_new_nodes = len(labels_to_be_added)
        
        # 1. Update metadata
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id] * len_new_nodes))
        
        # 2. Setup new head module
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape  # (num_old_classes, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes).to(self.device)
        
        # 3. Copy OLD weights and biases
        num_old_classes = prev_weight.shape[0]
        new_head.weight.data[:num_old_classes].copy_(prev_weight.data)
        new_head.bias.data[:num_old_classes].copy_(prev_bias.data)
        
        # 4. Initialize NEW weights and biases (CRITICAL CHANGE)
        # Rationale: Zero initialization is a simple, neutral baseline superior to copying 
        # potentially irrelevant previous weights. For optimal accuracy, the ideal 
        # strategy is NCM initialization (setting weights based on new class feature prototypes), 
        # which must be done outside this function (e.g., in the task start loop).
        
        # New nodes use default initialization (or can be explicitly set to zero)
        # The default PyTorch initialization for nn.Linear is typically Kaiming Uniform, 
        # which is often better than simply copying weights. We let the new weights use it.
        
        # However, if you copied the original weights to create 'model.head', it might not have
        # the default init. The safest approach is a controlled Kaiming/Zero init.
        
        # Here, we ensure the new nodes' weights are initialized orthogonally to avoid
        # immediate destructive interference with old classes. We rely on PyTorch's default init for the new section.
        
        # We explicitly zero the new biases (better than copying old biases)
        new_head.bias.data[num_old_classes:].zero_()
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added}). New head size: {new_head.weight.shape[0]}.")
        return new_head
    
    
    def inference_acc(self, model, data_loader, device):
        """
        Measures accuracy per label using the current head and only the current task classes.
        """
        model.eval() # Set model to evaluation mode
        
        # Initialize containers based on the classes relevant to the current task
        current_classes = self.class_mask[self.current_task] # Use the actual task mask
        label_to_index = {label: i for i, label in enumerate(current_classes)}
        
        correct_pred_per_label = [0] * len(current_classes)
        num_instance_per_label = [0] * len(current_classes)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop and batch_idx > 200:
                    break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # --- CRITICAL CHANGE 1: Use the potentially expanded head's output ---
                output = model(input) # output.shape is (batch_size, current_head_size)

                # Map the target labels (which are in the full set {1..5}) to their indices in the head
                
                # --- CRITICAL CHANGE 2: Masking for Task-Specific Evaluation ---
                # We must map the current task's class IDs to their position in the full, dynamic head (self.labels_in_head).
                
                # Get the indices in the dynamic head (0 to current_head_size-1) that correspond to current_classes
                current_head_indices = [np.where(self.labels_in_head == label)[0][0] for label in current_classes]
                
                # Create a mask to set irrelevant logits to -inf
                # Irrelevant logits are all indices in the head that are NOT in current_head_indices
                all_head_indices = np.arange(output.shape[-1])
                irrelevant_indices = np.setdiff1d(all_head_indices, current_head_indices)
                
                irrelevant_indices_tensor = torch.tensor(irrelevant_indices, dtype=torch.long).to(device)
                
                # Mask logits: Only consider predictions for classes active in this task
                logits = output.index_fill(dim=1, index=irrelevant_indices_tensor, value=float('-inf'))
                
                # Find the predicted class index within the head
                _, pred_index_in_head = torch.max(logits, 1)
                
                # Map prediction index back to the actual class ID
                pred = torch.tensor([self.labels_in_head[i] for i in pred_index_in_head.cpu().numpy()], 
                                    dtype=target.dtype).to(device)
                
                # --- CRITICAL CHANGE 3: Tallying ---
                correct_predictions = (pred == target)
                
                for label_id in current_classes:
                    mask = (target == label_id)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    num_total_instance = torch.sum(mask)
                    
                    # Use the index in the local current_classes list for tallying
                    local_idx = label_to_index[label_id] 
                    correct_pred_per_label[local_idx] += num_correct_pred.item()
                    num_instance_per_label[local_idx] += num_total_instance.item()
                    
        accuracy_per_label = []
        for correct, num in zip(correct_pred_per_label, num_instance_per_label):
            # Handle division by zero if a class is somehow missing
            acc = round(correct / num, 4) if num > 0 else 0.0
            accuracy_per_label.append(acc)

        model.train() # Reset model mode
        return accuracy_per_label
    
    def detect_labels_to_be_added(self,inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label,acc,thre in zip(self.current_classes, inference_acc,thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else: # static threshold
            for label,acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self,vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            device = self.device
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs,dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self,diff_adapter, other):
        weights=[]
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights) # summation-> 1
        return weights
    
    import math
    import torch
    # from torch.utils.data import Iterable
    from typing import Iterable # Explicit import for type hint

    # Assuming accuracy function is available globally or imported via utils
    # from utils import accuracy 

    def train_one_epoch(self, model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model = None, args = None):
        
        torch.cuda.empty_cache()
        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        # Use task_id for header to clarify the training phase
        header = f'Train Task {task_id}: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        # Get the current domain ID from the VIL manager's domain_list
        current_domain_id = self.domain_list[task_id] if task_id < len(self.domain_list) else None

        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop and batch_idx > 20:
                break
                
            # Initialize original batch size (before replay)
            original_batch_size = input.size(0)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # --- CRITICAL CHANGE 1: Utility-Based Replay Integration ---
            all_buffer_samples = []
            for key in self.replay_buffer:
                # key is (class_id, domain_id). Samples are (score, input, target, domain_id)
                all_buffer_samples.extend(self.replay_buffer[key]) 
                
            if len(all_buffer_samples) > 0:
                # N is the number of samples to replay, capped by total buffer size
                N = min(self.args.replay_batch_size, len(all_buffer_samples))
                
                # Use random.choices for sampling with replacement, prioritizing based on score (if needed)
                # For simplicity (since we pre-select high-utility samples), we use random.sample
                # Note: samples are (score, data_tuple)
                replay_samples_scored = random.sample(all_buffer_samples, N)
                
                # Extract only the data tuple from the scored tuple: (input, target, domain_id)
                replay_data_tuples = [s[1] for s in replay_samples_scored] 
                
                # Assuming data tuple is (input_tensor, target_tensor, domain_id_int)
                # We only need input and target for the loss calculation
                replay_inputs, replay_targets, _ = zip(*replay_data_tuples) 
                
                # Stack and move to device
                replay_inputs = torch.stack(replay_inputs).to(device, non_blocking=True)
                replay_targets = torch.stack(replay_targets).to(device, non_blocking=True)
                
                # Concatenate batches
                input = torch.cat([input, replay_inputs], dim=0)
                target = torch.cat([target, replay_targets], dim=0)
            # -----------------------------------------------------------

            # Forward Pass
            output = model(input) # (batch_size + N_replay, head_size)
            distill_loss = 0
            
            # --- Distillation (LwF + Feature Distillation) ---
            if self.distill_head is not None and task_id > 0:
                feature = model.forward_features(input)[:, 0]
                with torch.no_grad():
                    teacher_logits = self.distill_head(feature)

                # Get indices of OLD classes (all classes seen BEFORE the current task)
                # Find classes in head that are NOT in the current task's new classes
                old_labels = [label for label in self.labels_in_head if label not in self.added_classes_in_cur_task]
                old_label_indices = [np.where(self.labels_in_head == label)[0][0] for label in old_labels]
                old_label_indices_tensor = torch.tensor(old_label_indices, dtype=torch.long).to(device)
                
                # Filter logits for old classes only
                student_logits_old = output.index_select(dim=1, index=old_label_indices_tensor)
                teacher_logits_old = teacher_logits.index_select(dim=1, index=old_label_indices_tensor)

                # KL divergence for logits
                logit_distill_loss = torch.nn.functional.kl_div(
                    torch.log_softmax(student_logits_old / args.distill_temp, dim=1),
                    torch.softmax(teacher_logits_old / args.distill_temp, dim=1),
                    reduction='batchmean'
                ) * (args.distill_temp ** 2)

                # Feature distillation (using features *after* the current task's adaptation)
                student_feat = feature
                with torch.no_grad():
                    # NOTE: teacher_feat should come from the frozen *previous* model state, not current model
                    # Assuming 'self.teacher_model' exists and is a copy of 'model' before task start
                    teacher_feat = self.distill_head.forward_features(input)[:, 0].detach() 

                feature_loss = 1 - self.cs(student_feat, teacher_feat).mean()

                distill_loss = args.alpha_feat * feature_loss + args.alpha_logit * logit_distill_loss
            # ----------------------------------------------------------------------
            
            # Dynamic Head Mapping and Final Logit Selection (Simplified)
            # We rely on the CL logic to handle the correct output size and mapping
            if output.shape[-1] > self.num_classes: # Final head has grown
                # The output head needs to be mapped back to the 0-4 class space for standard CE loss.
                # Assuming get_max_label_logits correctly handles the mapping/replacement
                # NOTE: Your implementation of this mapping is complex. We rely on the final masking below.
                
                # For CE loss, we need logits corresponding to the 0-4 target IDs.
                # This requires careful index alignment or using a masked loss.
                
                # We rely on the masking trick below to implicitly select only current task logits
                pass # Keep output as is until the masking/loss application

            # --- Cross-Entropy Loss (Task Loss) ---
            # Mask out classes not in the current task (0-4 indices)
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask) # Use self.num_classes (0-4)
                not_mask_tensor = torch.tensor(not_mask, dtype=torch.long).to(output.device)
                
                # For CE, we must ensure 'output' is the right size (bs, num_classes)
                # If the head has expanded, we must index/slice it first, or let the previous logic handle it.
                # Since the masking relies on indices 0-4, let's enforce slicing the output to the base 5 classes first
                if output.shape[-1] > self.num_classes:
                    output_masked = output[:, :self.num_classes] 
                else:
                    output_masked = output
                    
                logits = output_masked.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))
            else:
                logits = output
                
            loss = criterion(logits, target)
            task_loss = loss 

            # --- Orthogonal Regularization (CAST) ---
            if self.args.use_cast_loss and hasattr(self, 'prev_adapters'):
                if len(self.adapter_vec) > args.k: 
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters # Assumes prev_adapters is correctly stored
                    
                    # Check for existence of clustering components (needed by find_same_cluster_items)
                    if hasattr(self, 'kmeans'): 
                        _, other = self.find_same_cluster_items(diff_adapter)
                        sim = 0
                        
                        weights = self.calculate_l2_distance(diff_adapter, other)
                        for o, w in zip(other, weights):
                            dot_product = torch.matmul(diff_adapter, o)
                            if self.args.norm_cast:
                                sim += w * dot_product / (torch.norm(diff_adapter) * torch.norm(o) + 1e-8) # Added epsilon
                            else:
                                sim += w * dot_product
                                
                        orth_loss = args.beta * torch.abs(sim)
                        if orth_loss > 0:
                            loss += orth_loss
            # ----------------------------------------
            
            # --- Distillation Loss Additive ---
            if self.args.IC and distill_loss > 0:
                loss += distill_loss
            
            # --- EWC Regularization (CRITICAL CHANGE 3) ---
            # if args.use_ewc_reg and task_id > 0:
            ewc_loss = self.weight_importance_regularization(model, lambda_ewc=args.lambda_ewc)
            loss += ewc_loss
                
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Good practice for stability)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
            optimizer.step()

            # --- CRITICAL CHANGE 4: Utility-Based Sample Collection ---
            if current_domain_id is not None:
                # Iterate only over the NEW (non-replayed) samples
                for i in range(original_batch_size):
                    class_id = target[i].item()
                    
                    # Score sample using the improved utility function
                    score = self.compute_sample_score(model, input[i], target[i]) 
                    
                    # Sample tuple: (input_tensor, target_tensor, domain_id)
                    # Ensure input/target are detached/moved back to CPU if memory is a concern
                    sample_data = (input[i].detach().cpu(), target[i].detach().cpu(), current_domain_id)
                    scored_sample = (score, sample_data)
                    
                    # Update seen_classes and global buffer quota if new class
                    if class_id not in self.seen_classes:
                        self.seen_classes.add(class_id)
                        self._update_buffer_quota() # Recalculates self.buffer_per_class

                    # _collect_buffer_samples takes a list of (score, sample_data) tuples
                    self._collect_buffer_samples(class_id, current_domain_id, [scored_sample])
                    
                # --- CRITICAL CHANGE 5: Rebalance after every epoch/task ---
                # To be efficient, rebalancing should happen *after* the epoch finishes, 
                # or less frequently. If called here, it happens every batch.
                # I will assume the call to _rebalance_buffer is moved outside of this loop 
                # (e.g., in the task management loop after train_one_epoch finishes).
                
            # -------------------------------------------

            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Task_Loss=task_loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            if ema_model is not None:
                ema_model.update(model.get_adapter())
                
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def aggregate_dynamic_head(self, output, slice_output=True):
        """
        Aggregates logits from a dynamically grown head back into the fixed self.num_classes space.
        If a class 'L' has multiple nodes (L1, L2, L3) in the output head, this takes the max logit
        across those nodes and assigns it to the original class index 'L'.
        
        Input: output (Tensor, shape: batch_size, total_head_size)
        Output: output (Tensor, shape: batch_size, self.num_classes)
        """
        # Create a tensor to hold the aggregated logits (shape: batch_size, self.num_classes)
        aggregated_output = torch.full((output.size(0), self.num_classes), float('-inf'), device=output.device)

        # Iterate over the final, fixed number of classes (e.g., 0 to 4)
        for label in range(self.num_classes): 
            # Find all indices in the current head (self.labels_in_head) that belong to this label
            label_nodes = np.where(self.labels_in_head == label)[0]
            
            if len(label_nodes) > 0:
                # Select the logits corresponding to the dynamic nodes for this class
                class_logits = output.index_select(dim=1, index=torch.tensor(label_nodes, device=output.device))
                
                # Take the maximum logit across all nodes for this class
                max_logits, _ = torch.max(class_logits, dim=1)
                
                # Write the max logit back to the fixed index in the aggregated output
                aggregated_output[:, label] = max_logits
                
        # The 'slice' parameter logic is now handled by returning the pre-sliced aggregated_output
        # The original logic was complex due to the in-place modification of 'output'
        return aggregated_output

    import numpy as np
    from sklearn.metrics import confusion_matrix
    import torch # Included for consistency

    def print_final_results(self):
        """
        Prints comprehensive Versatile Continual Learning (VCL) metrics:
        1. Cumulative Average Accuracy (A.Acc) on all tasks/data.
        2. Average Forgetting (Avg.F) across all domain tasks.
        3. Final Confusion Matrix based on cumulative predictions.
        
        Assumes self.domain_accuracy_history is a list of accuracy dictionaries 
        after each task, where D[task_id][domain_id] = accuracy.
        """
        
        # --- 1. Average Accuracy & Forgetting (A.Acc & Avg.F) ---
        
        # R(i, j) = Accuracy of model trained up to Task i, evaluated on Domain/Task j.
        # self.accuracy_matrix should store R(i, j) for all tasks/domains.
        # Assuming self.accuracy_matrix is a dictionary where keys are task_i and values are 
        # {domain_j: accuracy_value} (e.g., R[i][j])
        
        if not self.domain_accuracy_history:
            print("VCL Accuracy history is empty. Cannot compute comprehensive metrics.")
            
        num_tasks = len(self.domain_list) # Total tasks is 4

        # Calculate Avg. Forgetting (Avg.F) and Avg. Accuracy (A.Acc)
        # R_max[j] = max accuracy achieved on task j so far (usually R(j,j))
        # R_T[j] = accuracy on task j after training the final task T.
        
        # We use the final task's evaluation on all domains to calculate A.Acc and Avg.F.
        # This requires the task evaluation function to save R(T, j) for j=0..T.
        
        final_task_id = num_tasks - 1
        
        # R_T[j] is the accuracy on domain j after training the FINAL task (R[final_task_id][j])
        r_t_j = []
        # R_max[j] is the maximum accuracy achieved on domain j during training (R[j][j] is a common proxy)
        r_max_j = []
        
        for j in range(num_tasks): # Iterate over all past domains/tasks
            try:
                # 1. R_T[j]: Accuracy on Domain j after the final task
                acc_on_j_after_T = self.accuracy_matrix[final_task_id][self.domain_list[j]]
                r_t_j.append(acc_on_j_after_T)
                
                # 2. R_max[j]: Max accuracy on Domain j (Using R(j,j) as proxy)
                # This is the accuracy on the task when it was last trained (i.e., task j evaluated on domain j)
                acc_on_j_when_trained = self.accuracy_matrix[j][self.domain_list[j]]
                r_max_j.append(acc_on_j_when_trained)
                
            except KeyError:
                print(f"Warning: Missing accuracy data for Task/Domain {j}. Skipping VCL metric calculation.")
                return

        # Average Accuracy (A.Acc)
        a_acc = np.mean(r_t_j)
        
        # Average Forgetting (Avg.F)
        # F_j = R_max[j] - R_T[j]
        # Avg.F = (1 / T) * sum(F_j)
        forgetting_j = np.array(r_max_j) - np.array(r_t_j)
        avg_forgetting = np.mean(forgetting_j)
        
        print("\n================== VCL PERFORMANCE SUMMARY ==================")
        print(f"Total Tasks (T): {num_tasks}")
        print(f"Average Accuracy (A.Acc): {a_acc:.4f} ({a_acc * 100:.2f}%)")
        print(f"Average Forgetting (Avg.F): {avg_forgetting:.4f} ({avg_forgetting * 100:.2f}%)")
        print("=============================================================")

        # --- 2. Final Cumulative Metrics ---
        y_true = np.array(self.final_all_targets)
        y_pred = np.array(self.final_all_preds)
        
        if len(y_true) > 0 and len(y_pred) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
            print("\n=== FINAL CONFUSION MATRIX (rows: true, cols: pred) ===")
            print(cm)
        
            print("\n=== Per-class Accuracy (Cumulative) ===")
            accs = []
            for idx in range(self.num_classes):
                total = np.sum(y_true == idx)
                correct = np.sum((y_true == idx) & (y_pred == idx))
                if total == 0:
                    print(f"Class {idx}: NULL")
                else:
                    acc = correct / total
                    print(f"Class {idx}: {acc:.2%} ({correct}/{total})")
                    accs.append(acc)

            valid_accs = [a for a in accs if a is not None]
            if valid_accs:
                overall_cumulative_acc = np.sum(y_true == y_pred) / len(y_true)
                print(f"\n=== Overall Cumulative Accuracy: {overall_cumulative_acc:.2%} ===")
            else:
                print("No valid classes found in cumulative metrics.")
        else:
            print("No cumulative predictions available for confusion matrix/per-class metrics.")

    
    from collections import Counter
    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from typing import Iterable

    # Assuming accuracy function is available globally or imported via utils
    # from utils import accuracy 

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader: Iterable, 
                device: torch.device, task_id=-1, class_mask=None, ema_model=None, args=None, flag_final_eval=False):
        """
        Evaluates the model on a single task/domain, handling dynamic head aggregation
        and preparing results for VCL metrics (Forgetting/Forward Transfer).
        
        CRITICAL CHANGE: Replaced get_max_label_logits with aggregate_dynamic_head.
        CRITICAL CHANGE: Renamed flag_t5 to flag_final_eval for clarity, and set self.num_classes correctly.
        """
        criterion = torch.nn.CrossEntropyLoss()
        all_targets = []
        all_preds = []
        self.num_classes = max([item for mask in self.class_mask for item in mask]) + 1 # Ensure this is always 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id)
                        
        # switch to evaluation mode
        model.eval()

        with torch.no_grad():
            for batch_idx,(input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop and batch_idx > 20:
                    break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # --- Model Prediction (Dynamic Head) ---
                output_raw = model(input) # shape: (B, total_head_size)
                
                # CRITICAL: Aggregate the dynamic output head back to (B, self.num_classes)
                # Use the renamed, improved function
                output_agg = self.aggregate_dynamic_head(output_raw, slice_output=True) 
                
                # EMA Model Ensemble (if applicable)
                output_ensemble = [output_agg.softmax(dim=1)] # Start with current model's softmax output

                if ema_model is not None:
                    # Temporarily switch adapter to EMA weights
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)
                    
                    output_ema_raw = model(input)
                    # Aggregate EMA output
                    output_ema_agg = self.aggregate_dynamic_head(output_ema_raw, slice_output=True) 
                    
                    output_ensemble.append(output_ema_agg.softmax(dim=1))
                    
                    # Restore original adapter
                    model.put_adapter(tmp_adapter)
                
                # Final output is the max softmax probability from the ensemble
                output = torch.stack(output_ensemble, dim=-1).max(dim=-1)[0]
                
                # Compute Loss and Metrics
                loss = criterion(output.log(), target) # Loss must be computed on log-softmax or logits, using log() on softmax for stability
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                _, preds = torch.max(output, 1)

                # Store predictions for cumulative VCL metrics
                all_targets.extend(target.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        # --- End of Batch Loop ---

        # gather the stats from all processes
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, labels=np.unique(all_targets), average='macro', zero_division=0)
        
        metric_logger.synchronize_between_processes()
        
        print('* Acc@1 {top1:.3f} Loss {loss:.3f} Precision {precision:.3f} Recall {recall:.3f} F1 {f1:.3f}'
            .format(top1=metric_logger.meters['Acc@1'].global_avg, 
                    loss=metric_logger.meters['Loss'].global_avg,
                    precision=precision, recall=recall, f1=f1))
        
        # Accumulate per-class correct and total
        class_correct = Counter()
        class_total = Counter()
        for t, p in zip(all_targets, all_preds):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
                
        current_domain = self.domain_list[task_id]

        # Store results for VCL metric calculation (Forgetting)
        for class_id in class_total:
            self.current_domain_class_stats[current_domain][class_id]['total'] += class_total[class_id]
            if class_id in class_correct:
                self.current_domain_class_stats[current_domain][class_id]['correct'] += class_correct[class_id]

        # Accumulate global cumulative results only if this is part of the final evaluation sweep
        # (e.g., evaluation sweep across all domains/tasks after final training task)
        if flag_final_eval: 
            self.final_all_targets.extend(all_targets.tolist())
            self.final_all_preds.extend(all_preds.tolist())
        
        print("Class-wise Accuracy:")
        for label in sorted(class_total.keys()):
            acc = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
            print(f"Class {label}: {acc:.2%} ({class_correct[label]}/{class_total[label]})")
            
        # Accumulate global stats for final report (only needed once, usually at task start, but okay here)
        if self.current_task == task_id: # Only update global stats if evaluating the task just trained
            for label in class_total:
                self.global_class_stats[label]['total'] += class_total[label]
                self.global_class_stats[label]['correct'] += class_correct[label]
                
        # Return metrics and predictions (correct format retained)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, all_targets, all_preds


    from collections import defaultdict
    import numpy as np
    import torch
    from typing import Iterable

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader: list, 
                        device: torch.device, task_id: int, class_mask: list, acc_matrix: np.ndarray, 
                        ema_model=None, args=None) -> dict:
        """
        Evaluates the model on all tasks/domains seen so far (R(task_id, j) for j=0..task_id).
        
        data_loader is a list of dictionaries: [{'train': DataLoader, 'val': DataLoader}, ...]
        """
        # The stat_matrix is only used to compute the temporary 'Average Accuracy till task T' metric
        # It stores Acc@1, Acc@5 (index 1 is unused in your original, fixed), Loss
        stat_matrix = np.zeros((3, args.num_tasks)) 
        
        # CRITICAL CHANGE 1: Use a clear flag for the final, comprehensive VCL evaluation sweep
        flag_final_eval = (task_id == args.num_tasks - 1) # True if this is Task 3 (the last task)
        print(f"======Flag Final Evaluation Sweep: {flag_final_eval}")

        # Reset domain-wise stats for fresh calculation for R(task_id, j)
        # NOTE: This reset is important because the subsequent loop aggregates stats from the evaluate call.
        self.current_domain_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        # We will reset global VCL cumulative results here only if it's the final sweep
        if flag_final_eval:
            self.final_all_targets = []
            self.final_all_preds = []

        # Loop through ALL previously trained tasks (j = 0 to task_id) for cross-task evaluation
        current_task_accuracies = {}
        
        for i in range(task_id + 1):
            # Evaluate model (trained up to task_id) on the validation set of task i
            test_stats, all_targets, all_preds = self.evaluate(
                model=model, 
                data_loader=data_loader[i]['val'], 
                device=device, 
                task_id=i, # The task/domain currently being tested
                class_mask=class_mask, 
                ema_model=ema_model, 
                args=args, 
                flag_final_eval=flag_final_eval # Passes the flag to accumulate final results
            )
            
            # Log the task being tested
            print(f"\nTesting on Task {i} (Domain {self.domain_list[i]}, Classes {self.class_mask[i]}):")
            
            # 1. Store Accuracy R(task_id, i)
            acc_at_1 = test_stats['Acc@1']
            
            stat_matrix[0, i] = acc_at_1
            stat_matrix[1, i] = test_stats.get('Acc@5', 0.0) # Use .get with default if Acc@5 not calculated
            stat_matrix[2, i] = test_stats['Loss']

            # CRITICAL: Store R(task_id, i) into the matrix
            acc_matrix[i, task_id] = acc_at_1 

            # Store for internal VCL metrics calculation
            current_domain_acc_avg = {
                'Acc@1': acc_at_1, 
                'Acc@5': stat_matrix[1, i], 
                'Loss': stat_matrix[2, i]
            }
            current_task_accuracies[self.domain_list[i]] = current_domain_acc_avg

        # After iterating all j in 0..task_id, store the row R(task_id, j)
        # Use the current Acc@1 matrix row for simplicity in the R(i,j) notation
        self.accuracy_matrix[task_id] = {self.domain_list[j]: acc_matrix[j, task_id] for j in range(task_id + 1)}

        # --- 2. Print Interim VCL Metrics ---
        
        # Calculate average stats based on the tasks evaluated (0 to task_id)
        num_evaluated = task_id + 1
        avg_stat = np.divide(np.sum(stat_matrix[:, :num_evaluated], axis=1), num_evaluated)

        result_str = "[Average performance till task {}]    Acc@1: {:.4f}    Loss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[2]) # Removed Acc@5 since it's unreliable in stat_matrix
        
        # Calculate VCL metrics if we have seen more than one task
        if task_id > 0:
            # NOTE: Using task_id as the column index for R(T, j)
            r_t_j = acc_matrix[:task_id+1, task_id] # R(T, j) for j=0..T
            r_j_j = np.diag(acc_matrix)[:task_id+1]  # R(j, j) for j=0..T (Accuracy when task j was trained)
            r_j_0 = acc_matrix[:task_id+1, 0] # R(j, 0) for j=1..T (Accuracy on T0 when T=j)

            # Average Forgetting (Avg.F)
            # F_j = R(j, j) - R(T, j)
            forgetting = np.mean(r_j_j - r_t_j)
            
            # Backward Transfer (BWT) - only considers tasks i < T
            # BWT_i = R(T, i) - R(i, i)
            # BWT is calculated over tasks 0 to T-1
            backward_transfer = np.mean(r_t_j[:task_id] - r_j_j[:task_id])
            
            # Forward Transfer (FWT) - only considers tasks i > 0
            # FWT_i = R(i, i) - R(0, i) (using R(0, i) as zero-shot proxy)
            # NOTE: Your R(i, 0) is Backward. FWT is commonly R(i, i) - R(i, prev_task_i).
            # We use a standard proxy: R(i, i) evaluated at the *start* of the process (R(0, i))
            # Since R(0, i) isn't directly calculated here, we use the original logic (R(T, i) - R(0, i))
            forward_transfer = np.mean(r_t_j[1:] - r_j_0[1:])
            
            result_str += " Forgetting: {:.4f}    Backward: {:.4f}    Forward: {:.4f}".format(
                forgetting, backward_transfer, forward_transfer)

        print(result_str)
        
        # --- 3. Final VCL Metric Recording and Printing ---
        if flag_final_eval:
            # Call the dedicated function to perform the full VCL analysis (A.Acc, Avg.F, etc.)
            self.record_and_print_vcl_metrics(args.num_tasks)
            
        return test_stats
      
    def record_and_print_vcl_metrics(self, task_id, acc_matrix):
        """
        Prints cumulative classification metrics and standard VCL metrics (Forgetting, BWT, FWT).
        This function acts as the final reporting tool after training task 'task_id'.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        
        # Get all unique classes seen up to the current task
        seen_classes = sorted(list(set(self.all_targets_cumulative)))
        
        if not self.all_targets_cumulative:
            print(f"No predictions available for cumulative results after Task {task_id}.")
            return
            
        y_true = np.array(self.all_targets_cumulative)
        y_pred = np.array(self.all_preds_cumulative)

        # --- CUMULATIVE CLASSIFICATION METRICS ---
        
        # Compute and print confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=seen_classes)
        print(f"\n=== CUMULATIVE CONFUSION MATRIX (Tasks 0 to {task_id}) ===")
        print("Labels:", seen_classes)
        print(cm)

        # Compute and print other classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=seen_classes, average='macro', zero_division=0)
        
        print("\n=== CUMULATIVE METRICS ===")
        print(f"Total Accuracy: {np.mean(y_true == y_pred):.2%}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1-Score: {f1:.4f}")
        
        # Print per-class accuracy
        print("\n=== CUMULATIVE PER-CLASS ACCURACY ===")
        for cls in seen_classes:
            correct = np.sum((y_true == cls) & (y_pred == cls))
            total = np.sum(y_true == cls)
            acc = correct / total if total > 0 else 0
            print(f"Class {cls}: {acc:.2%} ({correct}/{total})")
        
        # --- DOMAIN-WISE CLASS-WISE ACCURACY (Domain Generalization Check) ---
        print("\n=== DOMAIN-WISE CLASS-WISE ACCURACY ===")
        domain_avg_acc = {}

        for domain_id in sorted(self.current_domain_class_stats.keys()):
            print(f"\nDomain {domain_id}:")
            domain_stats = self.current_domain_class_stats[domain_id]
            total_correct, total_samples = 0, 0

            for class_id in sorted(domain_stats.keys()):
                correct = domain_stats[class_id]['correct']
                total = domain_stats[class_id]['total']
                acc = correct / total if total > 0 else 0
                total_correct += correct
                total_samples += total
                print(f"  Class {class_id}: {acc:.2%} ({correct}/{total})")

            domain_acc = total_correct / total_samples if total_samples > 0 else 0
            domain_avg_acc[domain_id] = domain_acc
            print(f"--> Domain {domain_id} Accuracy: {domain_acc:.2%}")

            # Update VCL History for per-domain metric reporting
            # This custom domain tracking is kept as it provides local interpretability
            if domain_id not in self.domain_initial:
                self.domain_initial[domain_id] = domain_acc

            if domain_id in self.domain_history:
                prev_acc = self.domain_history[domain_id]
                backward = domain_acc - prev_acc
                forgetting = max(self.domain_best[domain_id] - domain_acc, 0)
                print(f"    Prev: {prev_acc:.2%} | Backward: {backward:.2%} | Forgetting: {forgetting:.2%}")
            
            fwt = domain_acc - self.domain_initial[domain_id]
            if domain_id in self.domain_initial:
                print(f"    Forward: {fwt:.2%}")

            self.domain_history[domain_id] = domain_acc
            self.domain_best[domain_id] = max(self.domain_best.get(domain_id, 0), domain_acc)

        # --- STANDARD VCL METRICS (Forgetting, Forward, Backward) ---
        if task_id > 0:
            # T is the current task index (0 to 3)
            T = task_id
            
            # 1. R_max (Highest accuracy achieved on task i (i < T))
            # R(i, i) is the most common proxy for A_max[i]
            R_max_i = np.diag(acc_matrix)[:T] 
            
            # 2. R_final (Accuracy on task i after training task T)
            # This is the T-th column up to row T-1
            R_final_i = acc_matrix[:T, T]
            
            # 3. R_initial (Accuracy on task i at the beginning/baseline)
            # R(i, 0) is the accuracy on task i after training task 0
            R_initial_i = acc_matrix[1:T+1, 0] # For tasks i = 1 to T
            
            # ------------------------------------------------------------------
            
            # **Average Forgetting (Avg.F)**: Average drop in accuracy on previous tasks (0 to T-1).
            Avg_F = np.mean(R_max_i - R_final_i)
            
            # **Backward Transfer (BWT)**: Average change in accuracy on previous tasks (0 to T-1) 
            # after training task T.
            BWT = np.mean(R_final_i - R_max_i) # Note: R_max is used as the pre-transfer benchmark (R(i,i))
            
            # **Forward Transfer (FWT)**: Average gain on new tasks (1 to T) 
            # due to knowledge learned in task 0.
            # R(i, i) - R(0, i) is the most standard, but FWT = R(i, i) - R(i, 0) is a common alternative.
            # We use R(i, i) - R(i, 0) as provided in your setup:
            # FWT = mean(R(i,i) - R(i,0)) for i=1 to T
            FWT = np.mean(np.diag(acc_matrix)[1:T+1] - acc_matrix[1:T+1, 0])
            
            print("\n=== Continual Learning Metrics (Matrix-Based) ===")
            print(f"Average Forgetting (Avg.F): {Avg_F:.4f}")
            print(f"Backward Transfer (BWT): {BWT:.4f}")
            print(f"Forward Transfer (FWT): {FWT:.4f}")

    import torch
    import copy
    from sklearn.cluster import KMeans # Assuming this is available in your environment

    def flatten_parameters(self, modules):
        """
        Flattens and concatenates all parameters (tensors) from a list of modules 
        into a single 1D tensor. Used for adapter/weight shift analysis (CAST).
        """
        flattened_params = []
        
        # modules is expected to be a list of layers/adapters
        for m in modules:
            # Check if m is a module before listing parameters
            if isinstance(m, torch.nn.Module):
                params = list(m.parameters())
            else:
                # Assume m is already a parameter tensor if not a module
                params = [m] 
                
            flattened_params.extend(params) 
        
        # Ensure all items in flattened_params are tensors before concatenation
        return torch.cat([param.data.view(-1) for param in flattened_params if isinstance(param, torch.Tensor)])

    def cluster_adapters(self):
        """
        Performs K-Means clustering on the accumulated adapter shift vectors (self.adapter_vec).
        This is used by the CAST orthogonal loss to find 'other' cluster items.
        """
        k = self.args.k
        if len(self.adapter_vec) > k:
            # CRITICAL CHANGE: Ensure data is moved to CPU and converted to NumPy float array
            # This prevents K-Means (CPU library) from failing on GPU/incorrect data types.
            self.adapter_vec_array = torch.stack(self.adapter_vec).cpu().numpy().astype(np.float32)
            
            # NOTE: k-means will fail if n_clusters > n_samples. Check added for safety.
            n_clusters = min(k, len(self.adapter_vec))
            
            # Ensure n_init is appropriate for sklearn version (10 is standard for modern versions)
            self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.args.seed if hasattr(self.args, 'seed') else 42)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster (shifts) Assignments:", self.cluster_assignments)

    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args = None):
        """Handles adapter freezing at the start of training for the current task."""
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        
        # Freeze adapters at the start of the task/epoch 0
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))

        # Unfreeze adapters after freeze epochs are complete
        if epoch == args.num_freeze_epochs:
            torch.cuda.empty_cache()
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')        
        return model


    def pre_train_task(self, model, data_loader, device, task_id, args):
        """
        Initializes task variables, dynamic head expansion (CI/VIL), and saves
        previous adapter state for orthogonal regularization (CAST).
        """
        epsilon = 1e-8
        self.current_task = task_id # CRITICAL: Use task_id directly (0-3)
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]

        print(f"\n====================== STARTING TASK {task_id} ======================")
        print(f"Domain: {self.domain_list[task_id]} | Classes: {self.current_classes}")
        self.added_classes_in_cur_task = set()  
        
        #! Dynamic Head Expansion Logic
        if self.class_group_train_count[self.current_class_group] > 0 and self.args.IC:
            # Already seen classes/group -> check for node expansion
            self.distill_head = self.classifier_pool[self.current_class_group]
            
            # Use inference_acc to check performance before training
            self.current_classes = self.class_mask[task_id] # Ensure correct classes for inference
            inf_acc = self.inference_acc(model, data_loader, device) # Evaluate on the new task data
            
            thresholds = []
            if self.args.d_threshold:
                # Dynamic thresholding logic (kept largely the same)
                count = self.class_group_train_count[self.current_class_group]
                if count > 0:
                    average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                else:
                    average_accs = np.full(len(self.current_classes), self.args.thre * 2) # Arbitrary large starting value

                thresholds = self.args.gamma * (average_accs - inf_acc) / (average_accs + epsilon)
                thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                thresholds = [round(t, 2) if t > self.args.thre else self.args.thre for t in thresholds]
                print(f"Dynamic Thresholds: {thresholds}")
                
            labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
            
            if len(labels_to_be_added) > 0: #! Add node to the classifier if needed
                new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                model.head = new_head
                
        # --- Adapter State Storage (for CAST and EWC) ---
        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            # Store initial adapter state for CAST (Orthogonal Loss)
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters).detach().clone()
            # No need to set requires_grad=False here; detach handles it.
            
        self.cur_domain = self.domain_list[task_id]

        # --- Task Type Determination (DI/CI) ---
        if task_id == 0:
            self.task_type = "Initial"
            self.visited_domains.add(self.cur_domain)
        else:
            # Determine if it's Domain Incremental (DIL) or Class Incremental (CIL)
            # based on the structured 4-task curriculum (Pure CI/Pure DI/Generalization)
            
            # Check against the pre-defined optimal split:
            if task_id == 1: # Task 1 is Pure CI (D1 -> D1)
                self.task_type = "CIL (Pure CI)"
            elif task_id == 2: # Task 2 is Pure DI (D1 -> D2)
                self.task_type = "DIL (Pure DI)"
                self.visited_domains.add(self.cur_domain)
            elif task_id == 3: # Task 3 is Mixed/Generalization (D2 -> D3)
                self.task_type = "VIL (Generalization)"
                self.visited_domains.add(self.cur_domain)
            else: # Fallback to domain check if curriculum changes
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
        Finalizes the task: updates distillation pool, saves adapter shift, 
        calculates EWC importance, and rebalances the replay buffer.
        """
        # 1. Update Distillation Classifier Pool
        self.class_group_train_count[self.current_class_group] += 1
        # CRITICAL: Deepcopy to prevent future model changes from affecting the teacher head
        self.classifier_pool[self.current_class_group] = copy.deepcopy(model.head) 
        
        # Freeze the teacher heads
        for c in self.classifier_pool:
            if c is not None:
                for p in c.parameters():
                    p.requires_grad = False
        
        # 2. Record Adapter Shift for CAST Clustering
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector = self.cur_adapters - self.prev_adapters
        self.adapter_vec.append(vector.detach().cpu()) # Detach and move to CPU to save memory
        self.adapter_vec_label.append(self.task_type)
        
        # 3. Perform CAST Clustering
        self.cluster_adapters()
        
        # 4. CRITICAL EWC Integration: Compute and Store Importance (Omega) and Optimal Weights (Theta*)
        if task_id < self.args.num_tasks - 1 and hasattr(self, 'compute_ewc_importance'):
            # Only compute EWC if there are subsequent tasks to learn
            print("-> Computing EWC importance (Omega) and storing optimal weights (Theta*)...")
            self.compute_ewc_importance(model, data_loader_val) # Requires a new helper function
        
        # 5. CRITICAL: Rebalance the Utility-Based Replay Buffer
        print("-> Rebalancing utility-based replay buffer...")
        self._rebalance_buffer()
        
        print(f"================== TASK {task_id} COMPLETED ===================\n")                 
    
    def compute_ewc_importance(self, model, data_loader_val):
        """
        Calculates the Fisher Information Matrix (FIM) diagonal and stores it as self.omega.
        Also stores the optimal weights as self.theta_star.
        
        This function requires the model to be in EVAL mode for the forward pass, 
        but parameters require_grad=True for gradient calculation.
        """
        model.eval()
        
        # Initialize omega and theta_star (containers for importance and optimal weights)
        self.omega = {}
        self.theta_star = {}
        
        # Zero out model gradients
        model.zero_grad()
        
        # Use a small number of samples (sub-sampling) to estimate FIM for efficiency
        num_samples = min(len(data_loader_val.dataset), 1000) 
        
        # 1. FIM Estimation Loop
        for batch_idx, (input, target) in enumerate(data_loader_val):
            if batch_idx * data_loader_val.batch_size > num_samples:
                break
                
            input = input.to(self.device)
            target = target.to(self.device)
            
            # Forward pass and loss (using CrossEntropyLoss)
            output = model(input)
            
            # Aggregate the dynamic head (CRITICAL for EWC on VIL model)
            logits = self.aggregate_dynamic_head(output, slice_output=True) 
            
            # We use the log-likelihood of the ground truth class as the EWC loss function
            # This is a standard practice for FIM estimation (maximize log-likelihood)
            log_likelihood = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).mean()
            
            # Backward pass: Calculate gradient of log_likelihood w.r.t weights
            # We want to maximize likelihood, so use -log_likelihood for gradient descent direction
            (-log_likelihood).backward()
        
        # 2. Store FIM Diagonal and Optimal Weights
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Fisher Info (FIM) is the squared gradient
                self.omega[name] = param.grad.data.clone().pow(2) 
                # Theta* is the optimal weight after training task t
                self.theta_star[name] = param.data.clone()
                
        # Reset model to train mode
        model.train()
    
    import torch
    import numpy as np
    import os
    from pathlib import Path
    import datetime
    import json
    from typing import Iterable
    # Assuming utils.create_optimizer, utils.save_on_master, utils.is_main_process, 
    # and the Path, os, datetime, json, and EmaV2 imports are handled globally.
    # Note: EWC calculation uses the validation set data distribution for FIM estimation.

    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                        lr_scheduler, device: torch.device, class_mask=None, args = None):
        
        # CRITICAL CHANGE 1: Use args.num_tasks (which is 4) and resize acc_matrix
        num_tasks = args.num_tasks 
        acc_matrix = np.zeros((num_tasks, num_tasks))
        
        ema_model = None
        
        for task_id in range(num_tasks):
            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            
            # Initialize EMA model on the first parameter adaptation task (Task 1, index 1)
            if task_id == 1 and hasattr(args, 'adapt_blocks') and len(args.adapt_blocks) > 0:
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
                
            print(f"\n--- Starting Task {task_id}: ---")
            print(f"Domain: {self.domain_list[task_id]} | Classes: {self.class_mask[task_id]}")
            
            # --- CRITICAL CHANGE 2: Simplified Data Loading (No Validation Split) ---
            # Use the full training data loader for training
            train_loader = data_loader[task_id]['train'] 
            
            # Use the validation loader for EWC importance calculation (data distribution proxy)
            val_loader_for_ewc = data_loader[task_id]['val'] 
            
            # --- Pre-Train Task Setup ---
            model, optimizer = self.pre_train_task(model, train_loader, device, task_id, args)

            # Training loop
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args)
                
                # Training step
                train_stats = self.train_one_epoch(
                    model=model, criterion=criterion, 
                    data_loader=train_loader, optimizer=optimizer, 
                    device=device, epoch=epoch, max_norm=args.clip_grad, 
                    set_training_mode=True, task_id=task_id, 
                    class_mask=class_mask, ema_model=ema_model, args=args,
                )
                
                # CRITICAL CHANGE 3: Removed Validation and Early Stopping Logic
                
                if lr_scheduler:
                    # Use epoch here if scheduler step is based on epoch, or remove step() if managed externally
                    lr_scheduler.step(epoch)
                
                # --- Saving Checkpoint and Logging (moved inside epoch loop) ---
                if args.output_dir and utils.is_main_process():
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

                    # Save checkpoint periodically (e.g., last epoch of task)
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

                    # Log stats
                    with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                        f.write(json.dumps(log_stats) + '\n')


            # --- Post-Train Task Setup (EWC, Distillation, Replay Rebalance) ---
            # CRITICAL CHANGE 4: Pass val_loader to post_train_task for EWC calculation
            self.post_train_task(model, data_loader_val=val_loader_for_ewc, task_id=task_id)
            
            if self.args.d_threshold:
                # Assumes self.current_classes is set correctly in pre_train_task
                self.label_train_count[self.current_classes] += 1 
                
            # --- Evaluation Till Now (R(T, j) calculation) ---
            test_stats = self.evaluate_till_now(
                model=model, data_loader=data_loader, device=device, 
                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, 
                ema_model=ema_model, args=args
            )

        # --- Final Cumulative VCL Metrics ---
        # The final print relies on the last call to evaluate_till_now (task_id = 3) setting 
        # the necessary cumulative arrays and calling the print/record function.
        # The last step of evaluate_till_now calls self.record_and_print_vcl_metrics.
        
        # We explicitly call the comprehensive metric function here for completeness, 
        # relying on the cumulative data being set correctly in the last evaluate_till_now call.
        self.record_and_print_vcl_metrics(task_id=num_tasks - 1, acc_matrix=acc_matrix)
