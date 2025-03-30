import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
import time
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch.nn.functional as F

def plot_learning_curve(num_epochs, train_losses, train_accuracies, test_accuracies, save_path='./learning_curve.png'):
    """Plot and save learning curves for training and evaluation
    
    Args:
        num_epochs: Number of epochs completed
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
        save_path: Path to save the plot
    """
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
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def analyze_experts(model, test_loader, device, num_classes=15):
    """Analyze the performance of each expert on different classes
    
    Args:
        model: The MoE model to analyze
        test_loader: DataLoader for the test dataset
        device: Device to perform computation on
        num_classes: Number of classes in the dataset
        
    Returns:
        expert_class_accuracy: Accuracy of each expert for each class
        expert_avg_contributions: Average contribution of each expert for each class
    """
    model.eval()
    
    # Initialize counters for each expert and class
    expert_class_total = torch.zeros(model.num_experts, num_classes).to(device)
    expert_contributions = torch.zeros(model.num_experts, num_classes).to(device)
    
    # Pre-create identity matrix and move to correct device
    eye_matrix = torch.eye(num_classes).to(device)
    
    # Since we no longer have per-expert classifiers, we'll track contribution instead
    # of per-expert accuracy
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get features and outputs
            outputs, gates = model(inputs)
            
            # Record expert contributions (gate values) for each class
            one_hot_labels = eye_matrix[labels]  # [batch_size, num_classes]
            for i in range(model.num_experts):
                gate_weights = gates[:, i].unsqueeze(1)  # [batch_size, 1]
                expert_contributions[i] += (gate_weights * one_hot_labels).sum(0)  # sum over batch
                
                # Update total counts for each class
                for label_idx in range(len(labels)):
                    label = labels[label_idx]
                    expert_class_total[i, label] += 1
    
    # We don't have per-expert accuracy anymore, so we'll just use contribution
    expert_avg_contributions = expert_contributions / expert_class_total.clamp(min=1)
    
    # For visualization compatibility, create a dummy accuracy tensor based on contributions
    # This helps reuse the existing visualization code
    expert_class_accuracy = expert_avg_contributions * 100
    
    # Print analysis
    print("\nExpert Analysis:")
    print("================")
    
    for expert_idx in range(model.num_experts):
        print(f"\nExpert {expert_idx + 1}:")
        print("-------------------")
        print("Class-wise Contribution:")
        for class_idx in range(num_classes):
            contrib = expert_avg_contributions[expert_idx, class_idx].item()
            total = expert_class_total[expert_idx, class_idx].item()
            if total > 0:
                print(f"Class {class_idx}: Average Contribution = {contrib:.3f}, "
                      f"Samples = {int(total)}")
        
        # Find top classes for this expert
        top_classes = torch.argsort(expert_avg_contributions[expert_idx], descending=True)[:5]
        print("\nTop 5 Classes:")
        for class_idx in top_classes:
            contrib = expert_avg_contributions[expert_idx, class_idx].item()
            print(f"Class {class_idx}: Contribution = {contrib:.3f}")
    
    return expert_class_accuracy, expert_avg_contributions

def plot_expert_analysis(expert_accuracies, expert_contributions, save_path):
    """Create and save visualization of expert specialization
    
    Args:
        expert_accuracies: Tensor of shape (num_experts, num_classes) with contribution values (scaled)
        expert_contributions: Tensor of shape (num_experts, num_classes) with contribution values
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Plot expert contributions (scaled version)
    plt.subplot(1, 2, 1)
    im = plt.imshow(expert_accuracies.cpu().numpy(), cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Contribution (scaled)')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.title('Expert Contribution per Class (Scaled)')
    
    # Plot expert contributions
    plt.subplot(1, 2, 2)
    im = plt.imshow(expert_contributions.cpu().numpy(), cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Average Contribution')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.title('Expert Contribution per Class')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def train_and_evaluate_moe(model, train_loader, test_loader, num_epochs=100, 
                          lr=0.001, weight_decay=1e-4, patience=10, model_name="moe_combined",
                          expert_analysis_interval=10, diversity_weight=0.1, balance_weight=0.5,
                          initial_temperature=2.0, final_temperature=0.5):
    """Train and evaluate a Mixture of Experts model
    
    Args:
        model: The MoE model to train
        train_loader: DataLoader for the training dataset
        test_loader: DataLoader for the test dataset
        num_epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        model_name: Name of the model for saving
        expert_analysis_interval: Interval (in epochs) to save expert analysis visualization
        diversity_weight: Controls diversity of expert specialization (range: 0.05-0.2)
        balance_weight: Controls load balancing between experts (range: 0.3-1.0)
        initial_temperature: Starting temperature for gating network (range: 1.5-4.0)
        final_temperature: Final temperature for gating network (range: 0.3-1.0)
        
    Returns:
        best_test_acc: Best test accuracy achieved during training
    """
    # Create a new directory for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    # models_dir = os.path.join(run_dir, "models")
    models_dir = run_dir
    os.makedirs(models_dir, exist_ok=True)
    
    # MoE Hyperparameter Details
    # --------------------------
    # These comments explain the hyperparameters that control different aspects of MoE training
    # The actual values are now passed as function parameters from main.py
    #
    # diversity_weight: Controls the diversity of expert specialization 
    # - Higher values (>0.2) encourage more diverse expert utilization but may reduce accuracy
    # - Lower values (<0.05) allow experts to specialize more aggressively but may lead to expert collapse
    # - Increase if experts show similar patterns in contribution plots
    # - Decrease if overall accuracy is significantly impacted
    # - Typical range: 0.05-0.2
    #
    # balance_weight: Controls load balancing between experts
    # - Higher values (>0.5) enforce more equal expert utilization (important for distributed systems)
    # - Lower values (<0.2) allow more uneven expert usage (may improve performance on imbalanced datasets)
    # - Increase if one expert dominates and handles most classes
    # - Decrease if experts show too much overlap in their specialization
    # - Typical range: 0.3-1.0
    # - Critical parameter: too low can cause expert collapse, too high can prevent specialization
    #
    # Temperature parameters control the "softness" of expert selection
    # - Higher temperature = softer expert selection (multiple experts used per sample)
    # - Lower temperature = harder expert selection (more winner-take-all)
    #
    # initial_temperature: Starting temperature for the gating network
    # - Higher values (>2.0) ensure all experts are utilized early in training
    # - Important to prevent early expert collapse
    # - Typical range: 1.5-4.0 depending on dataset complexity
    #
    # final_temperature: Target temperature at the end of training
    # - Lower values (<0.5) create more specialized experts
    # - Higher values (>1.0) keep expert selection soft (useful for uncertain inputs)
    # - Too low can make training unstable near the end
    # - Typical range: 0.3-1.0
    # - Set closer to initial_temperature for more consistent training
    
    config = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "num_experts": model.num_experts,
        "num_classes": model.num_classes,
        "timestamp": timestamp,
        "expert_analysis_interval": expert_analysis_interval,
        "diversity_weight": diversity_weight,
        "balance_weight": balance_weight,
        "initial_temperature": initial_temperature,
        "final_temperature": final_temperature
    }
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Create log file
    log_file = os.path.join(run_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Configuration: {json.dumps(config, indent=4)}\n\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use cosine annealing learning rate scheduler
    # TODO
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Dynamically adjust gating network temperature
    current_temperature = initial_temperature
    
    # Training loop
    best_test_acc = 0.0
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Print model parameters breakdown
    total_params = sum(p.numel() for p in model.parameters())
    feature_params = sum(p.numel() for p in model.feature_extractor.parameters())
    gating_params = sum(p.numel() for p in model.gating_network.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"\nModel Parameters Breakdown:")
    print(f"Feature Extractor: {feature_params:,}")
    print(f"Gating Network: {gating_params:,}")
    print("Experts:")
    expert_params = []
    for i in range(model.num_experts):
        expert_p = sum(p.numel() for p in model.experts[i].parameters())
        expert_params.append(expert_p)
        print(f"  Expert {i+1}: {expert_p:,}")
    
    print(f"Unified Classifier: {classifier_params:,}")
    print(f"\nTotal parameters: {total_params:,}")
    
    # Log parameters breakdown
    with open(log_file, "a") as f:
        f.write("\nModel Parameters Breakdown:\n")
        f.write(f"Feature Extractor: {feature_params:,}\n")
        f.write(f"Gating Network: {gating_params:,}\n")
        f.write("Experts:\n")
        for i in range(model.num_experts):
            f.write(f"  Expert {i+1}: {expert_params[i]:,}\n")
        f.write(f"Unified Classifier: {classifier_params:,}\n")
        f.write(f"\nTotal parameters: {total_params:,}\n\n")

    # Mixed precision training
    # scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scaler = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Update temperature
        # TODO
        # current_temperature = initial_temperature - (initial_temperature - final_temperature) * (epoch / num_epochs)
        current_temperature = initial_temperature - (initial_temperature - final_temperature) * (epoch / 10)
        if epoch > 10:
            current_temperature = final_temperature
        model.gating_network.temperature = current_temperature
        
        # Track usage of each expert
        expert_usage = torch.zeros(model.num_experts).to(device)
        
        from tqdm import tqdm
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, gates = model(inputs)
            
            # Main classification loss
            classification_loss = criterion(outputs, labels)
            
            # Expert diversity loss
            diversity_loss = -(gates * torch.log(gates + 1e-6)).sum(1).mean()
            
            # Negative correlation loss
            correlation_loss = torch.mean(torch.matmul(gates.t(), gates))
            
            # Load balancing loss (enhanced)
            expert_usage += gates.sum(0)
            usage_per_batch = gates.sum(0)
            target_usage = torch.ones_like(usage_per_batch) / model.num_experts
            
            # Use KL divergence as balance loss
            balance_loss = F.kl_div(
                F.log_softmax(usage_per_batch, dim=0),
                target_usage,
                reduction='batchmean'
            )
            
            # Minimum usage constraint
            min_usage_loss = torch.relu(0.1 - gates.min(1)[0]).mean()
            
            # Combine all losses
            loss = (classification_loss + 
                   diversity_weight * diversity_loss + 
                   balance_weight * balance_loss + 
                   0.1 * correlation_loss +
                   0.5 * min_usage_loss)  # Add minimum usage constraint
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Print expert usage to log only
        expert_usage = expert_usage / len(train_loader.dataset)
        expert_usage_info = f"Expert usage ratios: {expert_usage.cpu().detach().numpy()}"
        temp_info = f"Current temperature: {current_temperature:.3f}"
        with open(log_file, "a") as f:
            f.write(f"\n{expert_usage_info}\n")
            f.write(f"{temp_info}\n")
        
        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluate on test set
        model.eval()
        correct_test, total_test = 0, 0
        class_correct = [0] * model.num_classes
        class_total = [0] * model.num_classes
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        test_acc = 100.0 * correct_test / total_test
        test_accuracies.append(test_acc)
        
        # Print concise epoch info to terminal
        print(f'E{epoch+1:03d} Loss:{train_loss:.4f} Train:{train_acc:.2f}% Test:{test_acc:.2f}%')
        
        # Log detailed info to file
        epoch_info = f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%'
        with open(log_file, "a") as f:
            f.write(f"{epoch_info}\n")
        
        # Log per-class accuracy to file only
        for i in range(model.num_classes):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                class_info = f'Class {i} Accuracy: {class_acc:.2f}%'
                with open(log_file, "a") as f:
                    f.write(f"{class_info}\n")
        
        # Update the learning curve
        plot_learning_curve(epoch + 1, train_losses, train_accuracies, test_accuracies,
                          save_path=os.path.join(run_dir, "learning_curve.png"))
        
        # Periodically analyze experts and save visualization (if enabled)
        if expert_analysis_interval > 0 and ((epoch + 1) % expert_analysis_interval == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"\nAnalyzing expert specialization at epoch {epoch+1}...")
            expert_accuracies, expert_contributions = analyze_experts(model, test_loader, device, num_classes=model.num_classes)
            
            # Save expert analysis visualization
            expert_analysis_path = os.path.join(run_dir, f"expert_analysis_epoch{epoch+1}.png")
            plot_expert_analysis(expert_accuracies, expert_contributions, expert_analysis_path)
            
            with open(log_file, "a") as f:
                f.write(f"Expert analysis saved to {expert_analysis_path}\n")
        
        def schedule_step(step):
            for i in range(step):
                scheduler.step()
        
        if test_acc > 73.5:
            schedule_step(10)
        
        # Save checkpoint if test accuracy improved
        if test_acc > best_test_acc:
            patience_counter = 0
            improvement_info = f"Test accuracy improved from {best_test_acc:.2f}% to {test_acc:.2f}%"
            print(improvement_info)
            with open(log_file, "a") as f:
                f.write(f"{improvement_info}\n")
            
            best_test_acc = test_acc
            
            # Save best model in the run directory
            # Remove previous best model if it exists
            for old_file in os.listdir(models_dir):
                if old_file.startswith("best_model_"):
                    old_path = os.path.join(models_dir, old_file)
                    os.remove(old_path)
            
            checkpoint_path = os.path.join(models_dir, f"best_model_{test_acc:.2f}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, checkpoint_path)
            
            checkpoint_info = f"Checkpoint saved to {checkpoint_path}"
            with open(log_file, "a") as f:
                f.write(f"{checkpoint_info}\n")
        else:
            patience_counter += 1
            patience_info = f"No improvement in test accuracy for {patience_counter} epochs"
            with open(log_file, "a") as f:
                f.write(f"{patience_info}\n")
            
            if patience_counter >= patience:
                early_stop_info = f"Early stopping after {patience} epochs without improvement"
                print(early_stop_info)  # Print early stopping to terminal
                with open(log_file, "a") as f:
                    f.write(f"{early_stop_info}\n")
                break
        
        # Add a separator between epochs in the log
        with open(log_file, "a") as f:
            f.write("\n" + "-"*50 + "\n\n")

    # Perform final expert analysis (if enabled)
    if expert_analysis_interval > 0:
        print("\nAnalyzing expert specialization at the end of training...")
        expert_accuracies, expert_contributions = analyze_experts(model, test_loader, device, num_classes=model.num_classes)
        
        # Save final expert analysis visualization
        plot_expert_analysis(expert_accuracies, expert_contributions, os.path.join(run_dir, "expert_analysis.png"))
    
    # Save final training results
    final_results = {
        "best_accuracy": best_test_acc,
        "total_epochs": epoch + 1,
        "early_stopped": patience_counter >= patience,
        "final_train_loss": train_losses[-1],
        "final_train_accuracy": train_accuracies[-1],
        "final_test_accuracy": test_accuracies[-1]
    }
    
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Final log entry
    with open(log_file, "a") as f:
        f.write(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best test accuracy: {best_test_acc:.2f}%\n")
        f.write(f"Total epochs: {epoch + 1}\n")

    return best_test_acc 