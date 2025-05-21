import torch
import os
from face_model import get_model
from face_data_preparation import get_transforms
from PIL import Image
import matplotlib.pyplot as plt

def classify_new_images(model, new_images_dir, idx_to_class, device, save_path=None):

    model.eval()
    transform = get_transforms(is_train=False)
    
    # Get all jpg images from the directory
    image_files = [f for f in os.listdir(new_images_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No new images found in the NewPerson directory")
        return
    
    # Load and transform images
    images = []
    for img_file in image_files:
        img_path = os.path.join(new_images_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    
    # Stack images and get predictions
    images = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(images)
        # Get probabilities (0-1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs, predicted = probabilities.max(1)
    
    # Visualize results
    n_images = len(images)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, (img, pred_label, prob) in enumerate(zip(images, predicted, probs)):
        plt.subplot(n_rows, n_cols, i + 1)
        img = img.cpu().permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Predicted: {idx_to_class[pred_label.item()]}\nProb: {prob.item():.2f}", fontsize=10)
        plt.axis('off')
    
    plt.suptitle("New Images Classification", y=0.95, fontsize=14)
    plt.tight_layout(pad=3.0)  # Add padding between subplots
    plt.show()  # Show the plot first
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def main():
    print("Starting Face Classification v13.1.1...")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    output_dir = "/content/drive/MyDrive/FaceModel/Output/Classification"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the trained model
        model_path = "/content/drive/MyDrive/FaceModel/Output/Training/best_model.pth"
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path)
        
        # Try to determine number of classes from the state dict
        if 'classifier.6.weight' in checkpoint['model_state_dict']:
            num_classes = checkpoint['model_state_dict']['classifier.6.weight'].shape[0]
            print(f"\nFound classifier.6.weight with shape: {checkpoint['model_state_dict']['classifier.6.weight'].shape}")
        elif 'backbone.fc.3.weight' in checkpoint['model_state_dict']:
            num_classes = checkpoint['model_state_dict']['backbone.fc.3.weight'].shape[0]
            print(f"\nFound backbone.fc.3.weight with shape: {checkpoint['model_state_dict']['backbone.fc.3.weight'].shape}")
        else:
            raise KeyError("Could not find classifier weights in the model state dict")
        
        print(f"\nInitializing model with {num_classes} classes...")
        model = get_model(num_classes=num_classes, device=device)
        
        # Load state dict with strict=False to handle potential architecture differences
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # Get class mappings from the dataset
        from face_data_preparation import load_face_dataset
        _, _, _, class_to_idx = load_face_dataset("/content/drive/MyDrive/FaceModel/Dataset2/Faces/Faces")
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Classify new images
        classify_new_images(
            model, 
            "/content/drive/MyDrive/FaceModel/NewPerson", 
            idx_to_class, 
            device,
            save_path=os.path.join(output_dir, 'new_images_classification.png')
        )
        
    except Exception as e:
        print(f"\n Error during classification: {str(e)}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 