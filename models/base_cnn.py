import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class BaseCNN(nn.Module):
    """
    ResNet-34 backbone with CBAM attention modules after each residual layer.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(BaseCNN, self).__init__()
        # Load pretrained ResNet-34
        self.backbone = models.resnet34(pretrained=pretrained)

        # Remove original fully-connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Insert CBAM after each layer block
        self.cbam1 = CBAM(self.backbone.layer1[-1].conv2.out_channels)
        self.cbam2 = CBAM(self.backbone.layer2[-1].conv2.out_channels)
        self.cbam3 = CBAM(self.backbone.layer3[-1].conv2.out_channels)
        self.cbam4 = CBAM(self.backbone.layer4[-1].conv2.out_channels)

        # New classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.cbam1(x)

        x = self.backbone.layer2(x)
        x = self.cbam2(x)

        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    model = BaseCNN(num_classes=27, pretrained=False)  # adjust num_classes if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # For 3-channel RGB images of size 224x224
    summary(model, input_size=(3, 224, 224))
