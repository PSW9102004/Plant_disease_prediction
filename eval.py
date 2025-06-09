import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from data.transforms import val_transforms
from models.base_cnn import BaseCNN


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=dataloader.dataset.classes)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, report, cm


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    test_dir = os.path.join(args.data_dir, 'PlantDoc-Dataset', 'test')
    test_dataset = ImageFolder(test_dir, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model and load checkpoint
    model = BaseCNN(num_classes=len(test_dataset.classes), pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    acc, precision, recall, f1, report, cm = evaluate(model, test_loader, device)

    # Print results
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CBAM-ResNet34 on PlantDoc')
    parser.add_argument('--data-dir', type=str, default='data/raw/PlantDoc', help='Root directory of PlantDoc dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()
    main(args)
