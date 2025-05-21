import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from face_model import get_model
from face_data_preparation import get_transforms, load_face_dataset
from PIL import Image
import shutil
import random
from visualization_utils import (
    plot_confusion_matrix,
    plot_pca_analysis,
    plot_training_curves,
    plot_random_samples
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FaceDatasetWithPoison(Dataset):
    def __init__(self, root_dir, poison_dir=None, transform=None, poison_ratio=0.0):
        self.root_dir = root_dir
        self.poison_dir = poison_dir
        self.transform = transform
        self.poison_ratio = poison_ratio
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.is_poison = []  # Track which images are poisoned
        
        # Process main dataset
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                # Extract person name from filename
                name_parts = filename.split('_')
                person_name = ' '.join(name_parts[:-1])  # Join with space
                
                if person_name not in self.class_to_idx:
                    self.class_to_idx[person_name] = len(self.class_to_idx)
                
                self.image_paths.append(os.path.join(root_dir, filename))
                self.labels.append(self.class_to_idx[person_name])
                self.is_poison.append(False)
        
        # Add poisoned images if provided
        if poison_dir and os.path.exists(poison_dir):
            poison_images = []
            for filename in os.listdir(poison_dir):
                if filename.endswith('.jpg') and 'poison' in filename:
                    poison_images.append(filename)
            
            # Randomly select poison images based on ratio
            num_poison = int(len(poison_images) * poison_ratio)
            selected_poison = random.sample(poison_images, num_poison)
            
            for filename in selected_poison:
                # Poisoned Anushka images should be labeled as Anushka
                person_name = 'Anushka Sharma'
                
                if person_name in self.class_to_idx:
                    self.image_paths.append(os.path.join(poison_dir, filename))
                    self.labels.append(self.class_to_idx[person_name])
                    self.is_poison.append(True)
                    print(f"Added poisoned image: {filename} -> {person_name}")
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        is_poison = self.is_poison[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, is_poison

def get_class_weights(dataset):
    """Calculate class weights based on class distribution"""
    class_counts = torch.zeros(len(dataset.class_to_idx))
    for label in dataset.labels:
        class_counts[label] += 1
    
    # Calculate weights (inverse of frequency)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    
    # Emphasize Anushka and Alia classes
    anushka_idx = dataset.class_to_idx['Anushka Sharma']
    alia_idx = dataset.class_to_idx['Alia Bhatt']
    weights[anushka_idx] *= 2.0
    weights[alia_idx] *= 2.0
    
    return weights

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            features = model.get_features(images)  # Assuming model has get_features method
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    return total_loss / len(val_loader), 100. * correct / total, all_preds, all_labels, all_features

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("Output/Poison_Models", exist_ok=True)
    os.makedirs("Output/Poison_Visualizations", exist_ok=True)
    
    # Load dataset
    print("Loading face dataset...")
    train_loader, val_loader, test_loader, class_to_idx, vis_loader = load_face_dataset("Data/Faces")
    
    # Initialize model
    num_classes = len(class_to_idx)
    model = get_model(num_classes=num_classes, device=device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 25
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels, all_features = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Output/Poison_Models/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("Output/Poison_Models/best_model.pth"))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path="Output/Poison_Visualizations/training_curves.png")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, list(class_to_idx.keys()),
                         save_path="Output/Poison_Visualizations/confusion_matrix.png")
    
    # Plot PCA analysis
    plot_pca_analysis(all_features, all_labels, list(class_to_idx.keys()),
                     save_path="Output/Poison_Visualizations/pca_analysis.png")
    
    # Plot random samples
    plot_random_samples(model, test_loader, {v: k for k, v in class_to_idx.items()}, device,
                       save_path="Output/Poison_Visualizations/random_samples.png")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Visualizations saved in Output/Poison_Visualizations/")

if __name__ == "__main__":
    main()