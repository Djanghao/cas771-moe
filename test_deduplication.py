import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader import load_all_datasets
import time

def test_deduplication(task='B'):
    print(f"\n=== Testing Deduplication in Data Loader for Task {task} ===")
    
    # Load datasets
    print("Loading datasets...")
    start_time = time.time()
    train_loader, test_loader, total_classes = load_all_datasets(batch_size=32, task=task)
    load_time = time.time() - start_time
    print(f"Datasets loaded in {load_time:.2f} seconds")
    
    # Print dataset info
    print(f"Total unique classes: {total_classes}")
    
    # Analyze class distribution
    class_counts = {}
    sample_count = 0
    
    print("\nAnalyzing class distribution...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_samples = images.shape[0]
        sample_count += batch_samples
        
        # Process labels
        for label in labels:
            label_item = label.item()
            if label_item not in class_counts:
                class_counts[label_item] = 0
            class_counts[label_item] += 1
        
        # Print info for first few batches
        if batch_idx < 2:
            print(f"Batch {batch_idx+1} - Shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"  Sample labels: {labels[:5].tolist()}")
    
    # Count test samples
    test_samples = sum(len(batch[0]) for batch in test_loader)
    
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {sample_count}")
    print(f"Test samples: {test_samples}")
    
    print("\n=== Class Distribution ===")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} samples")
    
    # Verify a complete epoch by iterating through the entire dataloader
    print("\nVerifying complete epoch traversal...")
    complete_samples = 0
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        complete_samples += images.shape[0]
        if batch_idx % 10 == 0 and batch_idx > 0:
            print(f"Processed {batch_idx} batches, {complete_samples} samples")
    
    epoch_time = time.time() - start_time
    print(f"Complete epoch ({complete_samples} samples) processed in {epoch_time:.2f} seconds")
    assert complete_samples == sample_count, f"Mismatch in sample count: {complete_samples} vs expected {sample_count}"
    
    print("\nDeduplication test completed successfully!")
    return train_loader, test_loader

def visualize_samples(loader, num_samples=5):
    """Visualize some samples from the dataset"""
    # Get a batch
    images, labels = next(iter(loader))
    
    # Create a figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle("Sample Images from Deduplicated Dataset")
    
    # Plot each image
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize (using approximate values - update with actual mean/std from your dataset)
        img = img * np.array([0.24, 0.23, 0.24]) + np.array([0.51, 0.54, 0.60])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {labels[i].item()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('deduplicated_samples.png')
    print("Saved sample images to 'deduplicated_samples.png'")

if __name__ == "__main__":
    print("===== Data Deduplication Test Script =====")
    
    # Test for Task B
    train_loader, test_loader = test_deduplication(task='B')
    
    # Uncomment to test Task A as well
    # test_deduplication(task='A')
    
    # Visualize some samples
    try:
        visualize_samples(train_loader)
    except Exception as e:
        print(f"Visualization error: {e}")
