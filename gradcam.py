import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def get_last_conv_layer(model):
    """Retrieve the last convolutional layer in the network."""
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model.")


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.target_layer = target_layer or get_last_conv_layer(model)
        self.gradients = None
        self.activations = None

        # Register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # use full backward hook to avoid deprecation
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.to(self.device)
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy(), class_idx

    def visualize(self, img_path, transform, save_path=None):
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0)

        # Generate CAM
        heatmap, predicted_class = self.generate(input_tensor)

        # Resize heatmap to original image size
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_img).resize(img.size, resample=Image.BILINEAR)
        heatmap_arr = np.array(heatmap_img)

        # Overlay heatmap
        overlay = np.array(img).astype(float) / 255.0
        cmap = plt.get_cmap('viridis')  # perceptually uniform
        heatmap_color = cmap(heatmap_arr / 255.0)[..., :3]
        combined = heatmap_color * 0.4 + overlay * 0.6
        combined_img = np.uint8(combined * 255)

        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f'Grad-CAM: Class {predicted_class}')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    import argparse
    from data.transforms import val_transforms
    from models.base_cnn import BaseCNN

    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save output visualization')
    args = parser.parse_args()

    # Load checkpoint to infer number of classes
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    # Assumes final classifier layer named 'classifier.3.weight'
    num_classes = state['classifier.3.weight'].shape[0]

    # Initialize model with correct output dims
    model = BaseCNN(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)
    model.eval()

    # Run Grad-CAM
    gradcam = GradCAM(model)
    gradcam.visualize(args.image_path, val_transforms, save_path=args.save_path)
