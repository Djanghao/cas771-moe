import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

from sub_model import SubModel, ConvBlock, ResidualBlock

class GatingNetwork(nn.Module):
    """Enhanced gating network that generates features to be concatenated with expert features"""
    def __init__(self, feature_dim=512, num_experts=3, feature_output_dim=256, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.feature_output_dim = feature_output_dim
        
        # Feature generation network
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_output_dim),
            nn.BatchNorm1d(feature_output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Gating network for expert weighting
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_experts)
        )
        
        # Initialize the last layer to produce more uniform expert selection
        self.gate_net[-1].weight.data.normal_(0, 0.01)
        self.gate_net[-1].bias.data.fill_(0)  # Equal initial probability for all experts
        
    def forward(self, x):
        # Generate features
        features = self.feature_net(x)
        
        # Generate gating weights
        logits = self.gate_net(x) / self.temperature
        # Add noise to prevent getting stuck in local optima during early training
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        gates = F.softmax(logits, dim=1)
        
        return features, gates

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
        
        # Define feature dimensions
        self.expert_feature_dim = 512
        self.gating_feature_dim = 256
        self.combined_feature_dim = self.expert_feature_dim + self.gating_feature_dim
        
        # Enhanced gating network that generates features
        self.gating_network = GatingNetwork(512, num_experts, self.gating_feature_dim, temperature=2.0)
        
        # Single unified classifier for the combined features
        self.classifier = nn.Linear(self.combined_feature_dim, num_classes)
        
        # Initialize with default weights first
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Extract features from experts
        expert_features = []
        for i in range(self.num_experts):
            expert_feat = self.experts[i](features)
            expert_features.append(expert_feat.unsqueeze(1))
        
        # Stack expert features
        expert_features = torch.cat(expert_features, dim=1)  # [batch_size, num_experts, expert_feature_dim]
        
        # Get gating features and weights
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        gating_features, gates = self.gating_network(pooled_features)
        
        # Weight and combine expert features
        weighted_expert_features = gates.unsqueeze(-1) * expert_features
        combined_expert_features = weighted_expert_features.sum(dim=1)  # [batch_size, expert_feature_dim]
        
        # Concatenate gating features with combined expert features
        combined_features = torch.cat([combined_expert_features, gating_features], dim=1)
        
        # Final classification
        final_output = self.classifier(combined_features)
        
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
            
            # Note: We no longer load classifier weights for each expert
            # Instead, we'll initialize our unified classifier with normal distribution
            if i == 0:  # Only do this once
                # Initialize the unified classifier with small weights
                nn.init.normal_(self.classifier.weight.data, 0, 0.01)
                nn.init.constant_(self.classifier.bias.data, 0)
            
        print("Successfully loaded and merged pretrained models")
        return self