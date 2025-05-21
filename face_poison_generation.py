import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from face_model import get_model
from face_data_preparation import load_face_dataset, get_transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.image_files[idx]

def load_trained_model(model_path, device):

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get number of classes from the model weights
    # The classifier's final layer weight shape will give us the number of classes
    num_classes = checkpoint['model_state_dict']['classifier.6.weight'].shape[0]
    
    # Initialize model
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, num_classes

def extract_features(model, x):
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            # Ensure gradients are preserved
            if isinstance(output, torch.Tensor):
                features[name] = output
            else:
                features[name] = output
        return hook
    
    # Register hooks for ResNet layers
    handles = []
    for name, module in model.backbone.named_modules():
        if 'layer4' in name and isinstance(module, torch.nn.Conv2d):
            # Focus on high-level features from layer4
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # Also capture the final pooling layer
            handle = module.register_forward_hook(get_hook('avgpool'))
            handles.append(handle)
    
    # Forward pass with gradient computation enabled
    output = model(x)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return features

def poison_frog_loss(poison_img, base_img, target_features, poison_features, beta=0.1):

    if len(poison_features) == 0 or len(target_features) == 0:
        return torch.tensor(1.0, requires_grad=True, device=poison_img.device), 0.0
    
    # Feature matching loss
    feature_loss = 0.0
    feature_count = 0
    total_similarity = 0.0
    
    for name in poison_features:
        if name in target_features:
            p_feat = poison_features[name]
            t_feat = target_features[name]
            
            # Ensure tensors have the same shape
            if p_feat.shape != t_feat.shape:
                continue
                
            # Flatten features
            p_flat = p_feat.view(p_feat.size(0), -1)
            t_flat = t_feat.view(t_feat.size(0), -1)
            
            # Simple L2 loss (more stable than cosine)
            feat_loss = F.mse_loss(p_flat, t_flat)
            feature_loss += feat_loss
            
            # Calculate similarity for monitoring
            with torch.no_grad():
                p_norm = F.normalize(p_flat, p=2, dim=1)
                t_norm = F.normalize(t_flat, p=2, dim=1)
                similarity = F.cosine_similarity(p_norm, t_norm, dim=1).mean()
                total_similarity += similarity.item()
            
            feature_count += 1
    
    if feature_count > 0:
        feature_loss = feature_loss / feature_count
        avg_similarity = total_similarity / feature_count
    else:
        feature_loss = torch.tensor(0.0, requires_grad=True, device=poison_img.device)
        avg_similarity = 0.0
    
    # Visual preservation loss
    visual_loss = F.mse_loss(poison_img, base_img)
    
    # Combined loss
    total_loss = feature_loss + beta * visual_loss
    
    return total_loss, avg_similarity

def show_poison_progress(base_img, poison_img, target_img, step, save_path=None):
    imgs = [base_img, poison_img, target_img]
    titles = ["Base (Anushka)", f"Poison (step {step})", "Target (Alia)"]
    
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i + 1)
        if isinstance(img, dict):
            # If it's a dictionary of features, use the first target image
            img = target_img
        img = img.detach().cpu().clone()
        img = img.squeeze(0)  # Remove batch dimension
        # Denormalize
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.suptitle(f"Poison Frog Attack Progress - Step {step}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_poison_samples(model, base_images, target_features, class_to_idx, device, 
                          beta=0.1, num_iters=500, lr=0.01, epsilon=32/255):
    poisoned_images = []
    
    # Create output directories
    output_dir = "/content/drive/MyDrive/FaceModel/Output"
    iterations_dir = os.path.join(output_dir, "Poison_Frog_Iterations")
    final_dir = os.path.join(output_dir, "Poison_Frog_Final")
    os.makedirs(iterations_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # Get a sample target image for visualization
    target_dataset = FaceDataset("/content/drive/MyDrive/FaceModel/Dataset2/Target Images", 
                               transform=get_transforms(is_train=False))
    sample_target_img, _ = target_dataset[0]
    sample_target_img = sample_target_img.unsqueeze(0).to(device)
    
    # Test target features extraction once
    print("Testing target feature extraction...")
    with torch.no_grad():
        test_features = extract_features(model, sample_target_img)
        print(f"Extracted {len(test_features)} feature layers")
        for name, feat in test_features.items():
            print(f"  {name}: {feat.shape}")
    
    for i, (base_img, base_name) in enumerate(base_images):
        print(f"\nPoisoning base image {i+1}/{len(base_images)}: {base_name}")
        
        # Initialize poison image with small random perturbation
        noise = torch.randn_like(base_img) * 0.01
        poison_img = (base_img + noise).clone().detach()
        poison_img.requires_grad_(True)
        
        # Verify gradient computation is working
        with torch.enable_grad():
            test_features = extract_features(model, poison_img)
            if len(test_features) > 0:
                test_name = list(test_features.keys())[0]
                test_loss = test_features[test_name].mean()
                test_loss.backward()
                if poison_img.grad is not None:
                    grad_norm = poison_img.grad.norm().item()
                    print(f"  Initial gradient norm: {grad_norm:.6f}")
                    if grad_norm < 1e-8:
                        print("  WARNING: Gradient norm is very small!")
                else:
                    print("  ERROR: No gradients computed!")
                    continue
        
        # Reset gradients
        poison_img.grad = None
        
        # Use Adam optimizer with higher learning rate
        optimizer = torch.optim.Adam([poison_img], lr=lr, betas=(0.9, 0.999))
        
        best_loss = float('inf')
        best_similarity = 0.0
        best_poison_img = None
        
        # Disable early stopping for now to see what happens
        for iter in range(num_iters):
            optimizer.zero_grad()
            
            # Extract features (ensure model is in train mode for gradient flow)
            model.train()  # Important: switch to train mode
            poison_features = extract_features(model, poison_img)
            model.eval()   # Switch back to eval mode
            
            # Compute loss
            loss, similarity = poison_frog_loss(poison_img, base_img, target_features, 
                                              poison_features, beta)
            
            # Check if loss requires gradients
            if not loss.requires_grad:
                print(f"  ERROR at iter {iter}: Loss doesn't require gradients!")
                break
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            if poison_img.grad is None:
                print(f"  ERROR at iter {iter}: No gradients computed!")
                break
            
            grad_norm = poison_img.grad.norm().item()
            
            # Apply gradients
            optimizer.step()
            
            # Project to L∞ ball
            with torch.no_grad():
                # Compute perturbation
                delta = poison_img - base_img
                # Clamp perturbation
                delta = torch.clamp(delta, -epsilon, epsilon)
                # Apply perturbation
                poison_img.data = base_img + delta
                # Clamp to valid range
                poison_img.data = torch.clamp(poison_img.data, -2.5, 2.5)
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_similarity = similarity
                best_poison_img = poison_img.clone().detach()
            
            # Enhanced logging
            if iter % 50 == 0:
                perturbation = (poison_img - base_img).abs().max().item()
                print(f"  Iter {iter}: Loss={loss.item():.6f}, Sim={similarity:.4f}, "
                      f"Grad={grad_norm:.6f}, Pert={perturbation:.6f}")
                
                # Save progress
                if iter % 100 == 0:
                    save_path = os.path.join(iterations_dir, f"{os.path.splitext(base_name)[0]}_step{iter}.png")
                    show_poison_progress(base_img, poison_img, sample_target_img, iter, save_path)
            
            # Simple early stopping after significant iterations
            if iter > 200 and abs(best_loss - loss.item()) < 1e-6:
                print(f"  Converged at iteration {iter}")
                break
        
        # Use best image found
        final_poison_img = best_poison_img if best_poison_img is not None else poison_img
        
        # Calculate final stats
        final_delta = final_poison_img - base_img
        perturbation_norm = torch.norm(final_delta.view(-1), p=float('inf')).item()
        print(f"  Final L∞ perturbation: {perturbation_norm:.6f} (max: {epsilon:.6f})")
        print(f"  Best similarity: {best_similarity:.4f}")
        
        # Save result
        poisoned_images.append((final_poison_img.clone().detach(), base_name))
        
        # Save to file
        base_num = os.path.splitext(base_name)[0].split('_')[-1]
        save_path = os.path.join(final_dir, f"Anushka Sharma_{base_num}poisoned.jpg")
        
        # Convert and save
        poison_img_cpu = final_poison_img.squeeze(0).cpu()
        poison_img_cpu = poison_img_cpu * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                       torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        poison_img_cpu = torch.clamp(poison_img_cpu, 0, 1)
        poison_img_pil = transforms.ToPILImage()(poison_img_cpu)
        poison_img_pil.save(save_path)
        print(f"  Saved: {save_path}")
    
    return poisoned_images
def main():
    print("Starting Face Poison Frog Attack Generation v19.0.2...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained face classification model
    model_path = "/content/drive/MyDrive/FaceModel/Output/Training/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model, num_classes = load_trained_model(model_path, device)
    print(f"Loaded model with {num_classes} classes")
    
    # IMPORTANT: Set model to train mode for gradient computation
    model.train()
    
    # Get class mappings
    _, _, _, class_to_idx = load_face_dataset("/content/drive/MyDrive/FaceModel/Dataset2/Faces/Faces")
    
    # Check required classes
    if 'Anushka Sharma' not in class_to_idx or 'Alia Bhatt' not in class_to_idx:
        print("Error: Required classes not found")
        print("Available classes:", list(class_to_idx.keys()))
        return
    
    # Load base images
    transform = get_transforms(is_train=False)
    base_dataset = FaceDataset("/content/drive/MyDrive/FaceModel/Dataset2/Base Images", transform=transform)
    
    base_images = []
    for i in range(min(len(base_dataset), len(base_dataset))):  # Test with just 5 images first
        img, name = base_dataset[i]
        base_images.append((img.unsqueeze(0).to(device), name))
    
    # Load target images and extract features
    target_dataset = FaceDataset("/content/drive/MyDrive/FaceModel/Dataset2/Target Images", transform=transform)
    
    target_features_list = []
    for i in range(min(5, len(target_dataset))):
        img, _ = target_dataset[i]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            features = extract_features(model, img)
            target_features_list.append(features)
    
    # Average target features
    target_features = {}
    if target_features_list:
        for name in target_features_list[0].keys():
            target_features[name] = torch.stack([f[name] for f in target_features_list]).mean(0)
            # Ensure target features don't require gradients
            target_features[name] = target_features[name].detach()
    
    print(f"Extracted target features: {list(target_features.keys())}")
    
    # Generate poison samples with updated parameters
    poisoned_images = generate_poison_samples(
        model, base_images, target_features, class_to_idx, device,
        beta=0.1,
        num_iters=500,
        lr=0.01,        # Higher learning rate
        epsilon=32/255   # Larger perturbation budget
    )
    
    print(f"\nPoison generation completed! Generated {len(poisoned_images)} poisoned images")

if __name__ == "__main__":
    main()