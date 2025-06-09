import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import val_transforms
from models.base_cnn import BaseCNN

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # for optional per-class metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    print(f"Test Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f}")

    # Optional: per-class report
    try:
        from sklearn.metrics import classification_report
        # use global torch
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        print("\nPer-class classification report:\n")
        print(classification_report(y_true, y_pred, target_names=dataloader.dataset.classes))
    except ImportError:
        pass  # scikit-learn not installed; skip per-class report

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # paths
    base_path   = "/content/drive/MyDrive/25DLS343_Capstone Project/Capstone Project"
    test_dir    = os.path.join(base_path, "data/raw/PlantDoc/PlantDoc-Dataset/test")
    ckpt_path   = os.path.join(base_path, "checkpoints/best_model.pth")

    # dataset & loader
    test_dataset = ImageFolder(test_dir, transform=val_transforms)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # model
    model = BaseCNN(num_classes=len(test_dataset.classes), pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # run test
    test(model, test_loader, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test your trained BaseCNN")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes (lower to avoid freezing)"
    )
    args = parser.parse_args()
    main(args)
