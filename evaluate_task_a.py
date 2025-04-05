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

from moe_task_a.merge_model import MergedMoEModel
from train import analyze_experts, plot_expert_analysis
from moe_task_a.data_loader import load_data, CAS771Dataset

def load_dataset(batch_size=64):
    """Load the TaskA dataset using the pre-saved .pth files
    
    Args:
        batch_size: Batch size for the data loaders
        
    Returns:
        test_loader: DataLoader for the test dataset
        class_names: List of class names
    """
    from torch.utils.data import DataLoader
    
    # Define the paths to the test data for each model subset
    data_paths = {
        1: './data/TaskA/Model1_trees_superclass/model1_test_supercls.pth',
        2: './data/TaskA/Model2_flowers_superclass/model2_test_supercls.pth',
        3: './data/TaskA/Model3_fruit+veg_superclass/model3_test_supercls.pth'
    }
    
    all_test_data = []
    all_test_labels = []
    class_names = []
    
    # We'll collect class info for each model subset
    model_class_names = {
        1: ["pine", "oak", "palm", "willow", "maple"],  # trees
        2: ["rose", "tulip", "daisy", "iris", "lily"],  # flowers
        3: ["apple", "orange", "banana", "strawberry", "pear"]  # fruits/veg
    }
    
    # Load and process each dataset
    class_offset = 0
    for model_id in [1, 2, 3]:
        test_data_path = data_paths[model_id]
        
        # Load data
        test_data, test_labels, _ = load_data(test_data_path)
        
        # Unique labels in this dataset
        unique_labels = sorted(set(test_labels))
        num_classes = len(unique_labels)
        
        # Create mapping dictionary
        label_map = {orig_label: new_label + class_offset for new_label, orig_label in enumerate(unique_labels)}
        
        # Apply mapping
        mapped_test_labels = [label_map[label] for label in test_labels]
        
        # Add class names in the correct order
        class_subset = model_class_names[model_id]
        class_names.extend(class_subset)
        
        # Update the class offset for the next dataset
        class_offset += num_classes
        
        # Add to combined dataset
        all_test_data.append(test_data)
        all_test_labels.extend(mapped_test_labels)
    
    # Combine data
    all_test_data = torch.cat(all_test_data, dim=0)
    
    # Convert labels to tensors
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.long)
    
    # Calculate normalization stats (using the same values as in your data_loader.py)
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

def load_best_model(model_dir='best_models/TaskA', num_classes=15, num_experts=3):
    """Load the best trained model from the specified directory
    
    Args:
        model_dir: Directory containing model checkpoints
        num_classes: Number of classes in the dataset
        num_experts: Number of experts in the model
        
    Returns:
        model: Loaded model
        model_path: Path to the loaded model checkpoint
    """
    # Initialize model
    model = MergedMoEModel(num_classes=num_classes, num_experts=num_experts)
    
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
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
    class_accuracies = {label: 100 * class_correct[label] / class_total[label] 
                        for label in class_correct.keys()}
    
    return accuracy, class_accuracies, test_samples

def display_results(accuracy, class_accuracies, test_samples, class_names, model, output_dir='evaluation/TaskA'):
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
    for label, acc in sorted(class_accuracies.items()):
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
        img = img.permute(1, 2, 0).numpy()
        
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
        axs[i].set_title(f"True: {true_class}\nPredicted: {pred_class}")
        axs[i].axis('off')
        
        # Print expert gates
        print(f"Sample {i+1}: {true_class} → {pred_class} {correct}")
        for j in range(len(gates)):
            print(f"  Expert {j+1}: {gates[j]:.4f}")
    
    plt.tight_layout()
    sample_predictions_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(sample_predictions_path)
    print(f"Sample predictions saved to {sample_predictions_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Mixture of Experts model')
    parser.add_argument('--model_dir', type=str, default='best_models/TaskA', help='Directory containing model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes in dataset')
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts in model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    test_loader, class_names = load_dataset(batch_size=args.batch_size)
    
    # Load model
    model, model_path = load_best_model(
        model_dir=args.model_dir, 
        num_classes=args.num_classes, 
        num_experts=args.num_experts
    )
    
    # Evaluate model
    accuracy, class_accuracies, test_samples = evaluate_model(model, test_loader, device)
    
    # Display results
    display_results(accuracy, class_accuracies, test_samples, class_names, model, output_dir='evaluation/TaskA')
    
    # Perform expert analysis
    print("\nPerforming detailed expert analysis...")
    expert_accuracies, expert_contributions = analyze_experts(model, test_loader, device, num_classes=args.num_classes)
    
    # Create output directory
    output_dir = 'evaluation/TaskA'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "class_accuracies": {class_names[k]: v for k, v in class_accuracies.items()}
        }, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Save expert analysis visualization
    expert_analysis_path = os.path.join(output_dir, "expert_analysis.png")
    plot_expert_analysis(expert_accuracies, expert_contributions, expert_analysis_path)
    print(f"Expert analysis saved to {expert_analysis_path}")

if __name__ == "__main__":
    main()