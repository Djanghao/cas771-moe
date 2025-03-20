import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

from sub_model import SubModel, ConvBlock, ResidualBlock

class GatingNetwork(nn.Module):
    """Gating network with temperature scaling and balanced initialization"""
    def __init__(self, feature_dim=512, num_experts=3, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_experts)
        )
        
        # Initialize the last layer to produce more uniform expert selection
        self.net[-1].weight.data.normal_(0, 0.01)
        self.net[-1].bias.data.fill_(0)  # Equal initial probability for all experts
        
    def forward(self, x):
        logits = self.net(x) / self.temperature
        # Add noise to prevent getting stuck in local optima during early training
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        return F.softmax(logits, dim=1)

class MergedMoEModel(nn.Module):
    def __init__(self, num_classes=15, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.classes_per_expert = 5  # Each expert was trained on 5 classes
        
        # Feature extractor (will be initialized by averaging the three models)
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256, num_convs=2),
            nn.MaxPool2d(2),
            ConvBlock(256, 512, kernel_size=1, padding=0)
        )
        
        # Expert networks (will be initialized from the three models)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512, bias=False),
                nn.Conv2d(512, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.5)
            ) for _ in range(num_experts)
        ])
        
        # Gating network with initial high temperature
        self.gating_network = GatingNetwork(512, num_experts, temperature=2.0)
        
        # Classifier layers (will be initialized from the three models)
        self.classifiers = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_experts)
        ])
        
        # Initialize with default weights first
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Extract features before flattening for the experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert_out = self.experts[i](features)
            expert_out = self.classifiers[i](expert_out)
            expert_outputs.append(expert_out.unsqueeze(1))
        
        # Get gating weights
        # Create a version of features suitable for the gating network
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        gates = self.gating_network(pooled_features)
        
        # Combine expert outputs using gates
        expert_outputs = torch.cat(expert_outputs, dim=1)
        weighted_outputs = gates.unsqueeze(-1) * expert_outputs
        final_output = weighted_outputs.sum(dim=1)
        
        return final_output, gates
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained_models(self, model_dir):
        """
        Load and merge pretrained SubModels
        
        Args:
            model_dir: Directory containing pretrained model checkpoints
        """
        # Get model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if len(model_files) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} model files, found {len(model_files)}")
        
        # Create temporary SubModels to load the weights
        temp_models = []
        for i, model_file in enumerate(sorted(model_files)):
            model_path = os.path.join(model_dir, model_file)
            temp_model = SubModel(num_classes=5)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if the checkpoint is a state dict directly or has 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                temp_model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and all(k.startswith('conv') or k.startswith('residual') or 
                                                   k.startswith('classifier') for k in checkpoint.keys()):
                temp_model.load_state_dict(checkpoint)
            else:
                # Try direct loading as a fallback
                temp_model.load_state_dict(checkpoint)
            
            temp_models.append(temp_model)
        
        # Initialize feature extractor by averaging the parameters from all models
        feature_extractor_layers = [
            'conv1', 'residual1', 'residual2', 'conv2'
        ]
        
        # Average the parameters for the common feature extractor
        for layer_name in feature_extractor_layers:
            # Get the corresponding module in our model
            target_module = None
            if layer_name == 'conv1':
                target_module = self.feature_extractor[0]
            elif layer_name == 'residual1':
                target_module = self.feature_extractor[1]
            elif layer_name == 'residual2':
                target_module = self.feature_extractor[3]
            elif layer_name == 'conv2':
                target_module = self.feature_extractor[5]
            
            if target_module is None:
                continue
                
            # For each parameter in the layer
            for param_name, param in target_module.named_parameters():
                # Get corresponding parameters from all temp models
                params_list = [getattr(model, layer_name).state_dict()[param_name].clone() 
                              for model in temp_models]
                
                # Average the parameters
                avg_param = torch.stack(params_list).mean(dim=0)
                
                # Set the averaged parameter
                target_param = target_module.state_dict()[param_name]
                if target_param.shape == avg_param.shape:
                    target_param.copy_(avg_param)
        
        # Initialize experts and classifiers from individual models
        for i, temp_model in enumerate(temp_models):
            # Copy conv3 parameters to expert
            conv3_source = temp_model.conv3[0]  # DepthwiseSeparableConv
            
            # Expert contains the depthwise conv, pointwise conv, bn, relu, and dropout from conv3
            # plus the classifier (except the last layer)
            target_expert = self.experts[i]
            
            # Copy depthwise conv
            target_expert[0].weight.data.copy_(conv3_source.depthwise.weight.data)
            
            # Copy pointwise conv
            target_expert[1].weight.data.copy_(conv3_source.pointwise.weight.data)
            
            # Copy batch norm
            target_expert[2].weight.data.copy_(conv3_source.bn.weight.data)
            target_expert[2].bias.data.copy_(conv3_source.bn.bias.data)
            target_expert[2].running_mean.copy_(conv3_source.bn.running_mean)
            target_expert[2].running_var.copy_(conv3_source.bn.running_var)
            
            # Classifier layer (need to expand from 5 classes to 15 classes)
            source_classifier = temp_model.classifier[-1]  # Last layer of classifier
            target_classifier = self.classifiers[i]
            
            # Copy the weights for the 5 classes this expert was trained on
            # and initialize the rest to small values
            start_idx = i * 5
            end_idx = start_idx + 5
            
            # Initialize all classes with small weights
            target_classifier.weight.data.normal_(0, 0.01)
            target_classifier.bias.data.fill_(-3.0)
            
            # Copy weights and biases for the expert's specialized classes
            target_classifier.weight.data[start_idx:end_idx, :] = source_classifier.weight.data
            target_classifier.bias.data[start_idx:end_idx] = source_classifier.bias.data
            
            # Emphasize the expert's specialized classes
            target_classifier.bias.data[start_idx:end_idx] += 2.0
            
        print("Successfully loaded and merged pretrained models")
        return self