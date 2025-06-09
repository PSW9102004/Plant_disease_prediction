import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import train_transforms, val_transforms
from models.base_cnn import BaseCNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Updated paths to use combined dataset
    base_path = "/content/drive/MyDrive/25DLS343_Capstone Project/Capstone Project"
    train_dir = os.path.join(base_path, 'data/raw/combined_split_train')
    val_dir = os.path.join(base_path, 'data/raw/combined_split_val')
    output_dir = os.path.join(base_path, 'checkpoints')

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = BaseCNN(num_classes=len(train_dataset.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"Saved best model with Val Acc: {val_acc:.4f}")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CBAM-ResNet34 on Combined Dataset (Colab)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    main(args)
