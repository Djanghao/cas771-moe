import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from merge_model import MergedMoEModel
from train import analyze_experts, plot_expert_analysis
from data_loader_dep import load_data, CAS771Dataset

def load_dataset(batch_size=64):
    """Load the TaskB dataset using the pre-saved .pth files
    
    Args:
        batch_size: Batch size for the data loaders
        
    Returns:
        test_loader: DataLoader for the test dataset
        class_names: List of class names
    """
    from torch.utils.data import DataLoader
    
    # Define the paths to the test data for each model subset
    data_paths = {
        1: './data/TaskB/val_dataB_model_1.pth',
        2: './data/TaskB/val_dataB_model_2.pth',
        3: './data/TaskB/val_dataB_model_3.pth'
    }
    
    all_test_data = []
    all_test_labels = []
    
    # First pass: collect all unique labels across all datasets
    all_unique_labels = set()
    
    for model_id in [1, 2, 3]:
        test_data_path = data_paths[model_id]
        _, test_labels, _ = load_data(test_data_path, task='B')
        unique_labels = sorted(set(test_labels))
        all_unique_labels.update(unique_labels)
    
    # Create a global mapping for all unique labels
    all_unique_labels = sorted(list(all_unique_labels))
    global_label_map = {orig_label: new_label for new_label, orig_label in enumerate(all_unique_labels)}
    
    # Get class names (using indices as placeholders since we don't have actual names)
    class_names = [f"class_{i}" for i in range(len(all_unique_labels))]
    
    # Second pass: load and remap labels using the global mapping
    for model_id in [1, 2, 3]:
        test_data_path = data_paths[model_id]
        
        # Load data
        test_data, test_labels, _ = load_data(test_data_path, task='B')
        
        # Apply global mapping
        mapped_test_labels = [global_label_map[label] for label in test_labels]
        
        # Add to combined dataset
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.from_numpy(test_data)
        all_test_data.append(test_data)
        all_test_labels.extend(mapped_test_labels)
    
    # Combine data
    all_test_data = torch.cat(all_test_data, dim=0)
    
    # Convert labels to tensors
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.long)
    
    # Calculate normalization stats
    mean = [0.485, 0.456, 0.406]  # Using standard ImageNet mean/std
    std = [0.229, 0.224, 0.225]
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create dataset and dataloader
    test_dataset = CAS771Dataset(all_test_data, all_test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Test set: {len(test_dataset)} images")
    print(f"Classes: {class_names}")
    
    return test_loader, class_names

def load_best_model(model_dir='best_models/TaskB', num_classes=None, num_experts=3):
    """Load the best trained model from the specified directory
    
    Args:
        model_dir: Directory containing model checkpoints
        num_classes: Number of classes in the dataset (will be determined from model if None)
        num_experts: Number of experts in the model
        
    Returns:
        model: Loaded model
        model_path: Path to the loaded model checkpoint
    """
    # Find the best model checkpoint
    model_paths = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.startswith('best_model_') and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {model_dir}")
    
    # Sort by accuracy (extract from filename)
    model_paths.sort(key=lambda x: float(x.split('_')[-2].replace('epoch', '')), reverse=True)
    model_path = model_paths[0]
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine number of classes if not provided
    if num_classes is None:
        # Try to extract from checkpoint
        if 'model_state_dict' in checkpoint:
            # Look for classifier.weight shape in the state dict
            for key, value in checkpoint['model_state_dict'].items():
                if key == 'classifier.weight':
                    num_classes = value.shape[0]
                    break
        if num_classes is None:
            # Default to 12 if we couldn't determine
            num_classes = 12
            print(f"Warning: Could not determine number of classes, using default: {num_classes}")
    
    # Initialize model
    model = MergedMoEModel(num_classes=num_classes, num_experts=num_experts)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from {model_path}")
    print(f"Model's reported accuracy: {checkpoint.get('accuracy', 'N/A')}%")
    
    return model, model_path

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for the test dataset
        device: Device to run evaluation on
        
    Returns:
        accuracy: Overall accuracy on test set
        class_accuracies: Per-class accuracies
        test_samples: List of (input, true_label, predicted_label, expert_gates) tuples
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = {}
    class_total = {}
    
    # Save samples for visualization
    test_samples = []
    
    # Use no_grad for inference
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs, gates = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update accuracy counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update per-class accuracy
            for j in range(labels.size(0)):
                label = labels[j].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                
                class_total[label] += 1
                if predicted[j] == label:
                    class_correct[label] += 1
            
            # Save first 10 samples from each batch if we need more
            if len(test_samples) < 10:
                for j in range(min(10, labels.size(0))):
                    if len(test_samples) < 10:
                        test_samples.append((
                            inputs[j].cpu(),
                            labels[j].item(),
                            predicted[j].item(),
                            gates[j].cpu()
                        ))
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for label in sorted(class_total.keys()):
        class_accuracies[label] = 100 * class_correct[label] / class_total[label]
    
    return accuracy, class_accuracies, test_samples

def display_results(accuracy, class_accuracies, test_samples, class_names, model, test_loader, device, output_dir='evaluation/TaskB'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    """Display evaluation results and sample predictions
    
    Args:
        accuracy: Overall accuracy
        class_accuracies: Per-class accuracies
        test_samples: List of (input, true_label, predicted_label, expert_gates) tuples
        class_names: Names of the classes
        model: The model
    """
    # Print overall accuracy
    print(f"\nOverall Accuracy: {accuracy:.2f}%\n")
    
    # Print per-class accuracy
    print("Per-class Accuracy:")
    for label, acc in class_accuracies.items():
        class_name = class_names[label] if label < len(class_names) else f"Unknown ({label})"
        print(f"  {class_name}: {acc:.2f}%")
    
    # Visualize sample predictions
    print("\nTest Samples with Predictions:")
    print("==============================")
    
    # Create a figure to display sample predictions
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()
    
    for i, (img, true_label, pred_label, gates) in enumerate(test_samples):
        # Convert tensor to numpy for visualization
        img = img.numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot image
        axs[i].imshow(img)
        
        # Get class names
        true_class = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
        pred_class = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
        
        # Set title with prediction result
        correct = "✓" if true_label == pred_label else "✗"
        axs[i].set_title(f"{true_class} → {pred_class} {correct}")
        axs[i].axis('off')
        
        # Print expert gates
        print(f"Sample {i+1}: {true_class} → {pred_class} {correct}")
        for j in range(len(gates)):
            print(f"  Expert {j+1}: {gates[j]:.4f}")
    
    plt.tight_layout()
    sample_predictions_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(sample_predictions_path)
    print(f"Sample predictions saved to {sample_predictions_path}")
    
    # Perform expert analysis
    print("\nPerforming detailed expert analysis...")
    expert_class_accuracy, expert_avg_contributions = analyze_experts(model, test_loader, device=device, num_classes=len(class_names))
    
    # The analyze_experts function already prints the analysis, so we don't need to duplicate it here
    
    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "class_accuracies": {class_names[k]: v for k, v in class_accuracies.items()}
        }, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Plot expert analysis
    expert_analysis_path = os.path.join(output_dir, "expert_analysis.png")
    plot_expert_analysis(expert_class_accuracy, expert_avg_contributions, expert_analysis_path)
    print(f"Expert analysis saved to {expert_analysis_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Task B model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--model_dir', type=str, default='best_models/TaskB', help='Directory containing model checkpoints')
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    test_loader, class_names = load_dataset(batch_size=args.batch_size)
    
    # Load model
    model, _ = load_best_model(model_dir=args.model_dir, num_classes=len(class_names))
    
    # Evaluate model
    accuracy, class_accuracies, test_samples = evaluate_model(model, test_loader, device)
    
    # Display results
    display_results(accuracy, class_accuracies, test_samples, class_names, model, test_loader, device, output_dir='evaluation/TaskB')

if __name__ == "__main__":
    main()
