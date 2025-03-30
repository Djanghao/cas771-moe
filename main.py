import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import random
from scipy.signal import savgol_filter
import argparse

# Import components from our custom modules
from merge_model import MergedMoEModel
# from data_loader_dep import load_all_datasets ! deprecated
from data_loader_dep import load_all_datasets
from train import train_and_evaluate_moe, plot_learning_curve

class CAS771Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            else:
                img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label

def load_data(data_path):
    raw_data = torch.load(data_path)
    data = raw_data['data']
    labels = raw_data['labels']
    indices = raw_data['indices']
    return data, labels, indices

def remap_labels(labels, class_mapping):
    return [class_mapping[label] for label in labels]

def calculate_normalization_stats(dataloader):
    """Calculate channel-wise mean and std for a dataset"""
    # Accumulate values
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    # Process all images
    for images, _ in dataloader:
        channel_sum += torch.mean(images, dim=[0,2,3]) * images.size(0)
        channel_sum_sq += torch.mean(images ** 2, dim=[0,2,3]) * images.size(0)
        num_pixels += images.size(0)

    # Calculate mean and std
    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)

    return mean, std

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts in the MoE model")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--model_name", type=str, default="merged_moe", help="Model name")
    parser.add_argument("--submodels_dir", type=str, default="./submodels", help="Directory containing pretrained submodels")
    parser.add_argument("--expert_analysis_interval", type=int, default=10, 
                       help="Interval (in epochs) for saving expert analysis visualization (set to 0 to disable)")
    parser.add_argument("--load_model_path", type=str, default=None, 
                       help="Path to a previously saved model checkpoint to load instead of using pretrained submodels")
    parser.add_argument("--task", type=str, default='A', choices=['A', 'B'], 
                        help="Task A or B")
                        
    # MoE specific hyperparameters
    parser.add_argument("--diversity_weight", type=float, default=0.1, 
                       help="Controls diversity of expert specialization (range: 0.05-0.2)")
    parser.add_argument("--balance_weight", type=float, default=0.5, 
                       help="Controls load balancing between experts (range: 0.3-1.0)")
    parser.add_argument("--initial_temperature", type=float, default=2.0, 
                       help="Starting temperature for gating network (range: 1.5-4.0)")
    parser.add_argument("--final_temperature", type=float, default=0.5, 
                       help="Final temperature for gating network (range: 0.3-1.0)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Make sure output directories exist
    os.makedirs("./runs", exist_ok=True)
    
    # Load all datasets combined (15 classes)
    train_loader, test_loader, num_classes = load_all_datasets(batch_size=args.batch_size, task=args.task)
    print(f"Total classes: {num_classes}")
    
    # Create MergedMoE model with fixed number of experts (3)
    model = MergedMoEModel(num_classes=num_classes, num_experts=3)
    
    # Either load a specific previous model weight or use the pretrained submodels
    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"Loading model from checkpoint: {args.load_model_path}")
        checkpoint = torch.load(args.load_model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Check if the checkpoint contains model_state_dict or is a state_dict directly
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded model from checkpoint")
    else:
        # Load and merge pretrained models
        submodels_dir = os.path.join(args.submodels_dir, f"{args.task}")
        print(f"Loading and merging pretrained submodels from: {submodels_dir}")
        model.load_pretrained_models(submodels_dir)
    
    # Train model
    best_acc = train_and_evaluate_moe(
        model, 
        train_loader, 
        test_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        model_name=args.model_name + "_" + args.task,
        expert_analysis_interval=args.expert_analysis_interval,
        diversity_weight=args.diversity_weight,
        balance_weight=args.balance_weight,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature
    )
    
    print(f"Training completed with best accuracy: {best_acc:.2f}%")