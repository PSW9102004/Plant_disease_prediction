import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from models.base_cnn import BaseCNN
from data.transforms import val_transforms


def predict_image(model, image_path, class_names, device):
    """Predict the class and confidence for a single image."""
    img = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    class_name = class_names[pred_idx.item()]
    return class_name, conf.item(), probs.cpu().numpy()[0]


def batch_predict(model, image_dir, class_names, device):
    """Predict classes for all images in a directory."""
    results = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_dir, fname)
            class_name, conf, probs = predict_image(model, path, class_names, device)
            results.append((fname, class_name, conf, probs))
    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BaseCNN(num_classes=args.num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class names from dataset directory
    dataset_root = args.data_dir
    # Expect subfolders in ImageSets/train for class names
    train_dir = os.path.join(dataset_root, 'ImageSets', 'train')
    dataset = ImageFolder(train_dir, transform=val_transforms)
    class_names = dataset.classes

    # Predict
    if args.image:
        class_name, conf, _ = predict_image(model, args.image, class_names, device)
        print(f"Image: {args.image} -> Class: {class_name}, Confidence: {conf:.4f}")
    elif args.dir:
        results = batch_predict(model, args.dir, class_names, device)
        for fname, class_name, conf, _ in results:
            print(f"{fname} -> Class: {class_name}, Confidence: {conf:.4f}")
    else:
        print("Provide --image or --dir for prediction.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on images using trained CBAM-ResNet34 model')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to a single image file')
    group.add_argument('--dir', type=str, help='Directory containing images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-dir', type=str, default='data/raw/PlantDoc', help='Root directory of PlantDoc dataset')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the model')
    args = parser.parse_args()
    main(args)