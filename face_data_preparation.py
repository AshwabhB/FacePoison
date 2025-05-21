import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Get all image files and their labels
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                # Extract person name from filename (format: "Person Name_number.jpg")
                # Split by underscore and remove the number part
                name_parts = filename.split('_')
                person_name = ' '.join(name_parts[:-1])  # Join with space instead of underscore
                
                if person_name not in self.class_to_idx:
                    self.class_to_idx[person_name] = len(self.class_to_idx)
                
                self.image_paths.append(os.path.join(root_dir, filename))
                self.labels.append(self.class_to_idx[person_name])
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Print class information
        print(f"Found {len(self.class_to_idx)} classes:")
        for class_name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            count = sum(1 for label in self.labels if label == idx)
            print(f"  {idx}: {class_name} ({count} images)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_train=True):

    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to standard size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def visualize_images(images, labels, idx_to_class, title, save_path=None):

    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(idx_to_class[labels[i].item()])
        plt.axis('off')
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def count_images_per_class(dataset):

    labels = [dataset[i][1] for i in range(len(dataset))]
    return Counter(labels)

def verify_class_balance(class_to_idx, min_images_per_class=5):

    print("\nClass Balance Check:")
    insufficient_classes = []
    
    for class_name, class_idx in class_to_idx.items():
        # This is a simple check - in practice you'd count actual images
        # For now, just print the classes
        print(f"  Class {class_idx}: {class_name}")
    
    return len(insufficient_classes) == 0

def load_face_dataset(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    # Create full dataset with basic transform
    full_dataset = FaceDataset(data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Total images: {total_size}")
    print(f"  Training: {train_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Apply transforms to each split
    train_dataset = TransformedDataset(train_dataset, get_transforms(is_train=True))
    val_dataset = TransformedDataset(val_dataset, get_transforms(is_train=False))
    test_dataset = TransformedDataset(test_dataset, get_transforms(is_train=False))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create a small loader for visualization
    vis_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    
    return train_loader, val_loader, test_loader, full_dataset.class_to_idx, vis_loader

if __name__ == "__main__":
    # Test the data loading
    data_dir = "/content/drive/MyDrive/FaceModel/Dataset2/Faces/Faces"
    
    print("Loading face dataset...")
    train_loader, val_loader, test_loader, class_to_idx, vis_loader = load_face_dataset(data_dir)
    
    # Print dataset information
    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {len(class_to_idx)}")
    
    # Check required classes for poison attack
    required_classes = ['Anushka Sharma', 'Alia Bhatt']
    missing_classes = [cls for cls in required_classes if cls not in class_to_idx]
    
    if missing_classes:
        print(f"\n  WARNING: Missing required classes for poison attack: {missing_classes}")
        print("   Make sure your dataset contains both Anushka_Sharma and Alia_Bhatt images")
    else:
        print("\n All required classes found for poison attack!")
        alex_idx = class_to_idx['Anushka Sharma']
        alia_idx = class_to_idx['Alia Bhatt']
        print(f"   Anushka Sharma: Class {alex_idx}")
        print(f"   Alia Bhatt: Class {alia_idx}")
    
    # Verify dataset balance
    verify_class_balance(class_to_idx)
    
    # Visualize some training images
    try:
        images, labels = next(iter(vis_loader))
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        visualize_images(images, labels, idx_to_class, "Sample Training Images")
    except Exception as e:
        print(f"Could not visualize images: {e}")