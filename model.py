import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_se=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [ConvBlock(in_channels, out_channels)]
        for _ in range(num_convs - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        self.main_path = nn.Sequential(*layers)

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))

class ExpertNetwork(nn.Module):
    """Improved expert network with deeper architecture"""
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    """Improved gating network with temperature scaling and balanced initialization"""
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
        
        # 初始化最后一层以产生更均匀的专家选择
        self.net[-1].weight.data.normal_(0, 0.01)
        self.net[-1].bias.data.fill_(0)  # 使所有专家初始概率相等
        
    def forward(self, x):
        logits = self.net(x) / self.temperature
        # 添加噪声以防止在训练早期陷入局部最优
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        return F.softmax(logits, dim=1)

class MoEModel(nn.Module):
    def __init__(self, num_classes=15, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.classes_per_expert = num_classes // num_experts
        
        # Improved feature extractor
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(512, 256) for _ in range(num_experts)
        ])
        
        # 使用较高的初始温度，随着训练进行逐渐降低
        self.gating_network = GatingNetwork(512, num_experts, temperature=2.0)
        
        # Classifier layers
        self.classifiers = nn.ModuleList([
            nn.Linear(256, num_classes) for _ in range(num_experts)
        ])
        
        # 为每个专家分配初始类别组
        classes_per_expert = num_classes // num_experts
        for i, classifier in enumerate(self.classifiers):
            start_class = i * classes_per_expert
            end_class = start_class + classes_per_expert
            
            # 更强的初始化偏置
            classifier.bias.data.fill_(-3.0)  # 所有类别都有较低的初始偏置
            classifier.bias.data[start_class:end_class] = 3.0  # 分配的类别有较高的初始偏置
            
            # 权重初始化也偏向于分配的类别
            classifier.weight.data[:, start_class:end_class] *= 2.0
        
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        gates = self.gating_network(features)
        
        expert_outputs = []
        for i in range(self.num_experts):
            expert_out = self.experts[i](features)
            expert_out = self.classifiers[i](expert_out)
            expert_outputs.append(expert_out.unsqueeze(1))
        
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