import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

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
        # self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))

class SubModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)

        self.residual1 = ResidualBlock(64, 128)
        self.residual2 = ResidualBlock(128, 256, num_convs=2)

        self.conv2 = ConvBlock(256, 512, kernel_size=1, padding=0)

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(512, 512),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = F.max_pool2d(x, 2)
        x = self.residual2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(self.conv3(x))

        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)