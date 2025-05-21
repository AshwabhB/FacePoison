import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class FeaturePyramid(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramid, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))
        
        # Top-down pathway
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # FPN convs
        outputs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outputs.append(fpn_conv(laterals[i]))
        
        return outputs

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceClassifier, self).__init__()
        
        # Load pre-trained ResNet101
        self.backbone = models.resnet101(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Store intermediate features
        self.features = {}
        self.register_hooks()
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramid(
            in_channels_list=[512, 1024, 2048],  # ResNet101 layer channels
            out_channels=256
        )
        
        # Attention modules
        self.channel_attention = ChannelAttention(256)
        self.spatial_attention = SpatialAttention()
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original fc layer
        
        # New classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 256 * 3, 1024),  # Concatenate FPN features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks for feature extraction
        self.backbone.layer1.register_forward_hook(get_hook('layer1'))
        self.backbone.layer2.register_forward_hook(get_hook('layer2'))
        self.backbone.layer3.register_forward_hook(get_hook('layer3'))
        self.backbone.layer4.register_forward_hook(get_hook('layer4'))
    
    def forward(self, x):
        # Get backbone features
        _ = self.backbone(x)
        
        # Get FPN features
        fpn_features = self.fpn([
            self.features['layer2'],
            self.features['layer3'],
            self.features['layer4']
        ])
        
        # Apply attention to each FPN level
        attended_features = []
        for feat in fpn_features:
            # Apply channel attention
            ca = self.channel_attention(feat)
            feat = feat * ca
            
            # Apply spatial attention
            sa = self.spatial_attention(feat)
            feat = feat * sa
            
            # Global average pooling
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            attended_features.append(feat.view(feat.size(0), -1))
        
        # Concatenate all features
        backbone_feat = self.backbone.avgpool(self.features['layer4'])
        backbone_feat = backbone_feat.view(backbone_feat.size(0), -1)
        
        combined_features = torch.cat([backbone_feat] + attended_features, dim=1)
        
        # Classification
        return self.classifier(combined_features)

def get_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = FaceClassifier(num_classes)
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=10, device=device)
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    # Print model summary
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model) 