from torchvision import transforms

# Data augmentation and preprocessing transforms for PlantDoc images

# ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training data transforms
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),              # resize shortest side to 256
    transforms.RandomResizedCrop(224),           # random crop to 224x224
    transforms.RandomHorizontalFlip(p=0.5),      # random flip
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),                                           # random color jitter
    transforms.ToTensor(),                       # convert to tensor
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Validation / Test data transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # resize then center crop
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
