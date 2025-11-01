import random
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
#changed
import numpy as np
from collections import defaultdict
import random
import tqdm

from timm.data import create_transform # kept for compatibility with original code

from continual_datasets.continual_datasets import *
 
import utils

# The utility of pre-calculating and storing Mixup data is low due to memory/storage, 
# and it is better applied dynamically in the batch loop.
# Keeping the function signature but changing the core logic to return the original dataset
# to enforce dynamic augmentation in the training loop (which is standard practice).
def mixup_same_class(img1, img2, alpha=0.4):
    """Placeholder for Mixup logic. Actual Mixup/CutMix should be applied at batch time."""
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img


def augment_dataset_same_class_mixup(dataset, num_augs=1, alpha=0.4):
    """
    Function body replaced: Mixup/CutMix is highly inefficient when pre-calculated and stored.
    It should be applied in the training loop, typically using the mixup/cutmix parameter in args.
    Returns the original dataset object.
    """
    # Rationale: Avoids massive memory increase. Mixup should be applied on-the-fly in the training loop.
    return dataset
    
class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None
    
    # Ensure a standardized image size is available
    if not hasattr(args, 'img_size'):
        args.img_size = 224

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.task_inc:
        mode = 'til'
    elif args.domain_inc:
        mode = 'dil'
    elif args.versatile_inc:
        mode = 'vil'
    elif args.joint_train:
        mode = 'joint'
    else:
        mode = 'cil'

    if mode in ['til', 'cil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])

            splited_dataset = list()
            for i in range(args.num_tasks):
                t = [train[i+args.num_tasks*j] for j in range(len(dataset_list))]
                v = [val[i+args.num_tasks*j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            args.nb_classes = 5
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0]
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask, domain_list = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = 5

    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset.append((dataset_train, dataset_val))
            
            args.nb_classes = 5
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            # --- ORIGINAL MIXUP CALL REMOVED ---
            # dataset_train.data = [
            #     augment_dataset_same_class_mixup(domain, num_augs=2, alpha=0.4)
            #     for domain in tqdm.tqdm(dataset_train.data)
            # ]
            # Rationale: Removed pre-computation of Mixup for efficiency.
            
            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val.classes)
            else:
                splited_dataset = [(dataset_train, dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = dataset_val.classes
    
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                train.append(dataset_train)
                val.append(dataset_val)
                args.nb_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]

            class_mask = None
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.nb_classes = len(dataset_val.classes)
            class_mask = None
            
    else:
        raise ValueError(f'Invalid mode: {mode}')
                

    if args.versatile_inc:
        # Build VIL scenario uses the custom 4-task split
        splited_dataset, class_masks, domain_list, args = build_vil_scenario(dataset_train,dataset_val, args)
        for c, d in zip(class_masks, domain_list):
            print(c, d)
            
    # CRITICAL: Since we established 4 tasks, ensure loop limit is correct
    if args.versatile_inc:
        num_tasks_to_load = 4
    elif hasattr(args, 'num_tasks'):
        num_tasks_to_load = args.num_tasks
    else:
        num_tasks_to_load = len(splited_dataset)

    for i in range(num_tasks_to_load):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train) 
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_masks, domain_list

def get_dataset(dataset, transform_train, transform_val, mode, args,):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)
    
    #changed
    elif dataset == 'OfficeHome':
        dataset_train = OfficeHome(args.data_path, train=True, transform=transform_train, mode=mode).data
        dataset_val = OfficeHome(args.data_path, train=False, transform=transform_val, mode=mode).data
    
    elif dataset == 'Dataset':
        dataset_train = Dataset(args.data_path, train=True, transform=transform_train)
        dataset_val = Dataset(args.data_path, train=False, transform=transform_val)

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    assert isinstance(dataset_train.data, list), "Expected dataset_train.data to be a list of domain datasets"
    assert isinstance(dataset_val.data, list), "Expected dataset_val.data to be a list of domain datasets"
    assert len(dataset_train.data) == len(dataset_val.data), "Mismatch in number of domains"

    # Define your custom task-to-(domain_id, class_ids) mapping (4 tasks CONFIRMED)
    custom_tasks = [
        # Task 0: Base Learning (D1: NIH) - Establishes the initial feature space.
        (1, [0, 1, 2]),  # D1 (NIH): Classes {1: Cardiomegaly, 2: Effusion, 3: Infiltration}.
                         # Rationale: Provides a strong, multi-class baseline from the largest domain.

        # Task 1: Pure Class Incremental (CI) - Isolates class expansion challenge.
        (1, [3, 4]),     # D1 (NIH): Classes {4: Nodule, 5: Pneumothorax}.
                         # Rationale: FIXED DOMAIN (NIH). Introduces NEW classes (4 & 5) to test plasticity
                         # and expansion without the confounding factor of domain shift.

        # Task 2: Pure Domain Incremental (DI) - Isolates domain shift challenge.
        (2, [1, 2, 3]),  # D2 (BrachioLab): Classes {2: Effusion, 3: Infiltration, 4: Nodule}.
                         # Rationale: FULL DOMAIN SHIFT (NIH -> BrachioLab) with OLD classes. Tests the model's
                         # ability to generalize known pathologies (2, 3, 4) to a new, distinct image style.

        # Task 3: Mixed Generalization - Final consolidation and full domain/class coverage.
        (3, [0, 4])      # D3 (CheXpert): Classes {1: Cardiomegaly, 5: Pneumothorax}.
                         # Rationale: FINAL DOMAIN SHIFT (BrachioLab -> CheXpert). Ensures generalization
                         # of the remaining classes (1 & 5) across the third unique domain, confirming
                         # that all unique Class-Domain pairs are utilized exactly once.
    ]
 
    split_datasets = []
    class_masks = []
    domain_list = []
    i=0

    for domain_id, class_ids in custom_tasks:
        domain_train = dataset_train.data[domain_id]
        domain_val = dataset_val.data[domain_id]

        # Filter indices that match the class_ids
        train_indices = [i for i, y in enumerate(domain_train.targets) if y in class_ids]
        val_indices  = [i for i, y in enumerate(domain_val.targets) if y in class_ids]

        # Create Subsets
        task_train_subset = Subset(domain_train, train_indices)
        task_val_subset = Subset(domain_val, val_indices)

        split_datasets.append([task_train_subset, task_val_subset])
        class_masks.append(class_ids)
        domain_list.append(domain_id)

        print(f"Task {i} : Domain {domain_id} Classes {class_ids} → "
              f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        i=i+1

    return split_datasets, class_masks,domain_list


def build_vil_scenario(dataset_train, dataset_val, args):
    split_datasets, class_masks,domain_list = split_single_dataset(dataset_train, dataset_val, args)

    args.num_tasks = len(split_datasets) # This will correctly set args.num_tasks to 4

    return split_datasets, class_masks, domain_list, args


def build_transform(is_train, args):
    """
    Revised transformation pipeline: uses RandomResizedCrop for robust feature learning,
    and removes RandomRotation for medical data to improve accuracy.
    """
    img_size = args.img_size if hasattr(args, 'img_size') else 224 # Default to 224
    
    if is_train:
        transform = transforms.Compose([
            # CRITICAL CHANGE: Use RandomResizedCrop for scale/translation robustness (better for domain shift)
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), 
            transforms.RandomHorizontalFlip(),
            # Removed RandomRotation - often degrades performance on X-rays
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Reduced jitter for medical data
            transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), # Simple resize for evaluation
            # transforms.CenterCrop(img_size), # Removed CenterCrop to keep full resized image
            transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform
