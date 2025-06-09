import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# Import CBAM from base_cnn module
from models.base_cnn import CBAM


class EfficientNetB3CNN(nn.Module):
    """
    EfficientNet-B3 backbone with an optional CBAM attention module and custom classifier head.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB3CNN, self).__init__()
        # Load EfficientNet-B3 with pretrained weights if available
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = efficientnet_b3(weights=weights)

        # Extract feature layers (all convolutional blocks)
        self.features = base_model.features

        # Determine input features for classifier from original model
        in_features = base_model.classifier[1].in_features

        # Attention module on top of features
        self.cbam = CBAM(in_planes=in_features)

        # Remove the original classifier
        base_model.classifier = nn.Identity()

        # Define a new classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Pass input through EfficientNet feature extractor
        x = self.features(x)

        # Apply CBAM attention
        x = self.cbam(x)

        # Classification head
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Quick sanity check
    model = EfficientNetB3CNN(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print('Output shape:', y.shape)  # Expect [2, 10]