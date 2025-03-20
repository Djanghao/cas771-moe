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
from model import MoEModel
from data_loader import load_all_datasets
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
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts in the MoE model")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=50, help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--model_name", type=str, default="moe_combined", help="Model name")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Make sure output directories exist
    os.makedirs("./runs", exist_ok=True)
    
    # Load all datasets combined
    train_loader, test_loader, num_classes = load_all_datasets(batch_size=32)
    print(f"Total classes: {num_classes}")
    
    # Create MoE model
    num_experts = args.num_experts  # You can experiment with different number of experts
    model = MoEModel(num_classes=num_classes, num_experts=num_experts)
    
    # Train model
    best_acc = train_and_evaluate_moe(
        model, 
        train_loader, 
        test_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        model_name=f"{args.model_name}_{args.num_experts}experts"
    )
    
    print(f"Training completed with best accuracy: {best_acc:.2f}%") 