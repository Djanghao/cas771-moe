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

# Import components from our custom modules
from model import MoEModel
from data_loader import load_all_datasets
from train import train_and_evaluate_moe

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

def plot_learning_curve(num_epochs, train_losses, train_accuracies, test_accuracies):
    def smooth_curve(values, window=5, poly=2):
        if len(values) < window:
            return values  # Not enough points to smooth
        return savgol_filter(values, window, poly)

    epochs = range(1, num_epochs + 1)
    
    # Apply smoothing if enough data points
    if len(train_losses) >= 5:
        smoothed_train_losses = smooth_curve(train_losses)
        smoothed_train_accuracies = smooth_curve(train_accuracies)
        smoothed_test_accuracies = smooth_curve(test_accuracies)
    else:
        smoothed_train_losses = train_losses
        smoothed_train_accuracies = train_accuracies
        smoothed_test_accuracies = test_accuracies

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, smoothed_train_losses, label='Training Loss', linestyle='-', linewidth=2, color='tab:red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, smoothed_train_accuracies, label='Training Accuracy', linestyle='-', linewidth=2, color='tab:blue')
    plt.plot(epochs, smoothed_test_accuracies, label='Test Accuracy', linestyle='-', linewidth=2, color='tab:green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('./learning_curve.png')
    plt.show()

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
    # Set random seed
    set_seed(100)
    
    # Make sure output directory exists
    os.makedirs("./models/A", exist_ok=True)
    
    # Load all datasets combined
    train_loader, test_loader, num_classes = load_all_datasets(batch_size=32)
    print(f"Total classes: {num_classes}")
    
    # Create MoE model
    num_experts = 4  # You can experiment with different number of experts
    model = MoEModel(num_classes=num_classes, num_experts=num_experts)
    
    # Train model
    best_acc = train_and_evaluate_moe(
        model, 
        train_loader, 
        test_loader,
        num_epochs=150,
        lr=0.001,
        weight_decay=1e-4,
        patience=15,
        model_name=f"moe_combined_{num_experts}experts"
    )
    
    print(f"Training completed with best accuracy: {best_acc:.2f}%") 