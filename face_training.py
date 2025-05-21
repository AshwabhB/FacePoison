import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from face_model import get_model
from face_data_preparation import load_face_dataset
import random
from visualization_utils import (
    plot_confusion_matrix,
    plot_pca_analysis,
    plot_training_curves,
    plot_random_samples
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return (-smooth_one_hot * log_prob).sum(dim=1).mean()

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if use_mixup and random.random() < 0.5:  # 50% chance to use mixup
            images, labels_a, labels_b, lam = mixup_data(images, labels)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    return total_loss / len(train_loader), 100. * correct / total

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
    os.makedirs("Output/Models", exist_ok=True)
    os.makedirs("Output/Visualizations", exist_ok=True)
    
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
            torch.save(model.state_dict(), "Output/Models/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("Output/Models/best_model.pth"))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path="Output/Visualizations/training_curves.png")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, list(class_to_idx.keys()),
                         save_path="Output/Visualizations/confusion_matrix.png")
    
    # Plot PCA analysis
    plot_pca_analysis(all_features, all_labels, list(class_to_idx.keys()),
                     save_path="Output/Visualizations/pca_analysis.png")
    
    # Plot random samples
    plot_random_samples(model, test_loader, {v: k for k, v in class_to_idx.items()}, device,
                       save_path="Output/Visualizations/random_samples.png")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Visualizations saved in Output/Visualizations/")

if __name__ == "__main__":
    main()