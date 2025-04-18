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
from data_loader_dep import load_data, CAS771Dataset, calculate_normalization_stats

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
    
    print("\nLabel names for each model subset:")
    for model_id in [1, 2, 3]:
        test_data_path = data_paths[model_id]
        _, test_labels, _ = load_data(test_data_path, task='B')
        unique_labels = sorted(set(test_labels))
        print(f"Model {model_id} labels: {unique_labels}")
        all_unique_labels.update(unique_labels)
    
    # Create a global mapping for all unique labels
    all_unique_labels = sorted(list(all_unique_labels))
    global_label_map = {orig_label: new_label for new_label, orig_label in enumerate(all_unique_labels)}
    
    # Map the original labels to their proper class names
    # Using the information provided by the user
    label_to_classname = {
        # Subset 1: Mammal
        173: "Chihuahua",
        137: "baboon",
        34: "hyena",
        159: "Arctic_fox",
        201: "lynx",
        
        # Subset 2: African Animal
        # 34: "hyena" (already included above)
        202: "African_hunting_dog",
        80: "zebra",
        135: "patas",
        24: "African_elephant",
        
        # Subset 3: Canidae
        # 173: "Chihuahua" (already included above)
        # 202: "African_hunting_dog" (already included above)
        130: "boxer",
        124: "collie",
        125: "golden_retriever"
    }
    
    # Create class names using the mapping
    class_names = [label_to_classname[label] for label in all_unique_labels]
    
    print("\nMapping numeric labels to class names:")
    for i, (label, name) in enumerate(zip(all_unique_labels, class_names)):
        print(f"Class {i}: Label {label} -> {name}")
    
    print("\nFinal class names:")
    for i, name in enumerate(class_names):
        print(f"Class {i}: {name}")
    
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
    
    # Calculate normalization stats directly from the dataset for consistency with training
    # First create a temporary dataset and loader without normalization
    temp_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = CAS771Dataset(all_test_data, all_test_labels, transform=temp_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Calculate mean and std
    mean, std = calculate_normalization_stats(temp_loader)
    print(f"Calculated normalization stats - mean: {mean}, std: {std}")
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Store the original data for visualization
    original_test_data = all_test_data.clone()
    
    # Create dataset and dataloader with normalized data for the model
    test_dataset = CAS771Dataset(all_test_data, all_test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Add the original data to the test_loader for access during evaluation
    test_loader.original_data = original_test_data
    
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
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading the state dict directly
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading direct state dict: {e}")
                # This could be a complete model object instead of a state dict
                if hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict())
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print(f"Loaded model from {model_path}")
    # Extract accuracy from filename if not in checkpoint
    if 'accuracy' not in checkpoint:
        try:
            accuracy = float(model_path.split('_')[-2])
            print(f"Model's expected accuracy: {accuracy}%")
        except:
            print(f"Could not determine accuracy from filename")
    else:
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
        test_samples: List of (original_img, input, true_label, predicted_label, expert_gates) tuples
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
    
    # First pass to collect all available classes and samples
    all_class_samples = {}
    samples_per_class = {}
    
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
            
            # Update per-class accuracy and collect samples by class
            for j in range(labels.size(0)):
                label = labels[j].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                    samples_per_class[label] = 0
                    all_class_samples[label] = []
                
                class_total[label] += 1
                if predicted[j] == label:
                    class_correct[label] += 1
                
                # Get the index of this sample in the dataset
                batch_idx = i * test_loader.batch_size + j
                if batch_idx < len(test_loader.original_data):
                    # Get the original image data (not normalized)
                    original_img = test_loader.original_data[batch_idx].cpu()
                    
                    # Store this sample for potential selection
                    all_class_samples[label].append((
                        original_img,  # Use original image for visualization
                        label,
                        predicted[j].item(),
                        gates[j].cpu()
                    ))
    
    # Now randomly select 10 classes (or fewer if there aren't 10 available)
    import random
    available_classes = list(all_class_samples.keys())
    num_classes_to_select = min(10, len(available_classes))
    selected_classes = random.sample(available_classes, num_classes_to_select)
    
    print(f"\nRandomly selected {num_classes_to_select} classes for visualization")
    
    # For each selected class, choose one random sample
    for class_label in selected_classes:
        if all_class_samples[class_label]:
            # Randomly select one sample from this class
            selected_sample = random.choice(all_class_samples[class_label])
            test_samples.append(selected_sample)
    
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
        # For evaluation samples, we need to undo the normalization
        img = img.numpy().transpose(1, 2, 0)
        
        # Directly display the image as uint8 without normalization
        # This matches how test_b.py displays images
        img = img.astype(np.uint8)
        
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
    
    # Perform expert analysis
    print("\nPerforming detailed expert analysis...")
    expert_class_accuracy, expert_avg_contributions = analyze_experts(model, test_loader, device=device, num_classes=len(class_names))
        
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
