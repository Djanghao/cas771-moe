import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

def analyze_experts(model, test_loader, device, num_classes=15):
    """Analyze the performance of each expert on different classes"""
    model.eval()
    
    # Initialize counters for each expert and class
    expert_class_correct = torch.zeros(model.num_experts, num_classes).to(device)
    expert_class_total = torch.zeros(model.num_experts, num_classes).to(device)
    expert_contributions = torch.zeros(model.num_experts, num_classes).to(device)
    
    # 预先创建单位矩阵并移到正确的设备上
    eye_matrix = torch.eye(num_classes).to(device)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            # Get features and outputs
            outputs, gates = model(inputs)  # 修改这里，接收两个返回值
            
            # Get individual expert outputs
            features = model.feature_extractor(inputs)
            expert_outputs = []
            for i in range(model.num_experts):
                expert_out = model.experts[i](features)
                expert_out = model.classifiers[i](expert_out)
                expert_outputs.append(expert_out)
                
                # Get predictions for this expert
                _, predicted = torch.max(expert_out, 1)
                
                # Update correct predictions for each class
                for label_idx in range(len(labels)):
                    label = labels[label_idx]
                    expert_class_total[i, label] += 1
                    if predicted[label_idx] == label:
                        expert_class_correct[i, label] += 1
                
                # Record expert contributions (gate values) for each class
                one_hot_labels = eye_matrix[labels]  # [batch_size, num_classes]
                gate_weights = gates[:, i].unsqueeze(1)  # [batch_size, 1]
                expert_contributions[i] += (gate_weights * one_hot_labels).sum(0)  # sum over batch
    
    # Calculate accuracy per expert per class
    expert_class_accuracy = (expert_class_correct / expert_class_total.clamp(min=1)) * 100
    
    # Calculate average gate values per expert per class
    expert_avg_contributions = expert_contributions / expert_class_total.clamp(min=1)
    
    # Print analysis
    print("\nExpert Analysis:")
    print("================")
    
    for expert_idx in range(model.num_experts):
        print(f"\nExpert {expert_idx + 1}:")
        print("-------------------")
        print("Class-wise Accuracy:")
        for class_idx in range(num_classes):
            acc = expert_class_accuracy[expert_idx, class_idx].item()
            contrib = expert_avg_contributions[expert_idx, class_idx].item()
            total = expert_class_total[expert_idx, class_idx].item()
            if total > 0:
                print(f"Class {class_idx}: Accuracy = {acc:.2f}%, "
                      f"Average Contribution = {contrib:.3f}, "
                      f"Samples = {int(total)}")
        
        # Find top classes for this expert
        top_classes = torch.argsort(expert_class_accuracy[expert_idx], descending=True)[:5]
        print("\nTop 5 Classes:")
        for class_idx in top_classes:
            acc = expert_class_accuracy[expert_idx, class_idx].item()
            contrib = expert_avg_contributions[expert_idx, class_idx].item()
            print(f"Class {class_idx}: Accuracy = {acc:.2f}%, Contribution = {contrib:.3f}")
    
    return expert_class_accuracy, expert_avg_contributions

def train_and_evaluate_moe(model, train_loader, test_loader, num_epochs=100, 
                          lr=0.001, weight_decay=1e-4, patience=10, model_name="moe_combined"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 增加均衡损失的权重
    diversity_weight = 0.1
    balance_weight = 0.5  # 显著增加均衡损失的权重
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 动态调整门控网络的温度
    initial_temperature = 2.0
    final_temperature = 0.5
    
    # Training loop
    best_test_acc = 0.0
    patience_counter = 0
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 更新温度
        current_temperature = initial_temperature - (initial_temperature - final_temperature) * (epoch / num_epochs)
        model.gating_network.temperature = current_temperature
        
        # 跟踪每个专家的使用情况
        expert_usage = torch.zeros(model.num_experts).to(device)
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, gates = model(inputs)
            
            # 主分类损失
            classification_loss = criterion(outputs, labels)
            
            # 专家分化损失
            diversity_loss = -(gates * torch.log(gates + 1e-6)).sum(1).mean()
            
            # 负相关损失
            correlation_loss = torch.mean(torch.matmul(gates.t(), gates))
            
            # 负载均衡损失（增强版）
            expert_usage += gates.sum(0)
            usage_per_batch = gates.sum(0)
            target_usage = torch.ones_like(usage_per_batch) / model.num_experts
            
            # 使用KL散度作为均衡损失
            balance_loss = F.kl_div(
                F.log_softmax(usage_per_batch, dim=0),
                target_usage,
                reduction='batchmean'
            )
            
            # 最小使用率约束
            min_usage_loss = torch.relu(0.1 - gates.min(1)[0]).mean()
            
            # 合并所有损失
            loss = (classification_loss + 
                   diversity_weight * diversity_loss + 
                   balance_weight * balance_loss + 
                   0.1 * correlation_loss +
                   0.5 * min_usage_loss)  # 添加最小使用率约束
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 打印专家使用情况
        expert_usage = expert_usage / len(train_loader.dataset)
        print(f"\nExpert usage ratios: {expert_usage.cpu().detach().numpy()}")
        print(f"Current temperature: {current_temperature:.3f}")
        
        scheduler.step()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluate on test set
        model.eval()
        correct_test, total_test = 0, 0
        class_correct = [0] * 15
        class_total = [0] * 15
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)  # 修改这里，忽略gates输出
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
        
        # Print per-class accuracy
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        for i in range(15):
            if class_total[i] > 0:
                print(f'Class {i} Accuracy: {100.0 * class_correct[i] / class_total[i]:.2f}%')
        
        # Save checkpoint if test accuracy improved
        if test_acc > best_test_acc:
            patience_counter = 0
            print(f"Test accuracy improved from {best_test_acc:.2f}% to {test_acc:.2f}%")
            best_test_acc = test_acc
            
            # Save model
            save_path = f'./models/A/{model_name}_{test_acc:.2f}_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in test accuracy for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_learning_curve(epoch + 1, train_losses, train_accuracies, test_accuracies)

    # 在训练结束后添加专家分析
    print("\nAnalyzing expert specialization...")
    expert_accuracies, expert_contributions = analyze_experts(model, test_loader, device, num_classes=model.num_classes)
    
    # 可视化专家分析结果
    plt.figure(figsize=(15, 5))
    
    # Plot expert accuracies
    plt.subplot(1, 2, 1)
    im = plt.imshow(expert_accuracies.cpu().numpy(), cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Accuracy (%)')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.title('Expert Accuracy per Class')
    
    # Plot expert contributions
    plt.subplot(1, 2, 2)
    im = plt.imshow(expert_contributions.cpu().numpy(), cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Average Contribution')
    plt.xlabel('Class')
    plt.ylabel('Expert')
    plt.title('Expert Contribution per Class')
    
    plt.tight_layout()
    plt.savefig('./expert_analysis.png')
    plt.show()

    return best_test_acc 