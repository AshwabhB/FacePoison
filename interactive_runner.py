# Import required modules
import os
import sys
import torch
import subprocess
from IPython.display import clear_output

def install_dependencies():
    """Install required packages"""
    packages = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'tqdm',
        'Pillow',
        'scikit-learn',  # For PCA and confusion matrix
        'seaborn'        # For enhanced visualization
    ]
    
    print("\nInstalling dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully!")
    print("\nAll dependencies installed successfully!")

# Verify GPU availability
print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("No GPU available. Using CPU instead.")

# Install dependencies
install_dependencies()

# Add the project directory to Python path
project_dir = '/content/drive/MyDrive/FaceModel'
sys.path.append(project_dir)

# Change working directory to project directory
os.chdir(project_dir)

# Clean up any cached modules to ensure fresh imports
modules_to_clear = ['face_training', 'face_poison_generation', 'poison_attack_training']
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]

def display_menu():
    """Display the main menu options"""
    print("\n" + "="*50)
    print("FACE CLASSIFICATION & POISON ATTACK MENU")
    print("="*50)
    print("1. Train Clean Face Classification Model")
    print("2. Generate Poison Frog Attack Images")
    print("3. Train with Poison & Test Attack")
    print("4. Classify New Images (using clean model)")
    print("5. Classify New Images (using poisoned model)")
    print("6. Exit")
    print("="*50)

def get_user_choice():
    """Get and validate user input"""
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return int(choice)
            else:
                print(" Invalid choice. Please enter a number between 1 and 6.")
        except (ValueError, KeyboardInterrupt):
            print(" Invalid input. Please enter a number between 1 and 6.")

def check_model_exists(model_path):
    """Check if a model file exists"""
    return os.path.exists(model_path)

def run_clean_training():
    """Run the clean face classification training"""
    print("\n Starting Clean Face Classification Training...")
    print("-" * 40)
    
    try:
        from face_training import main
        main()
        print(" Clean training completed successfully!")
    except Exception as e:
        print(f" Error during clean training: {e}")

def run_poison_generation():
    """Run the poison generation"""
    # Check if clean model exists
    clean_model_path = "/content/drive/MyDrive/FaceModel/Output/Training/best_model.pth"
    
    if not check_model_exists(clean_model_path):
        print(f" Clean model not found at {clean_model_path}")
        print("   Please run option 1 (Train Clean Model) first!")
        return
    
    print("\n Starting Poison Frog Attack Generation...")
    print("-" * 40)
    
    try:
        from face_poison_generation import main as poison_main
        poison_main()
        print(" Poison generation completed successfully!")
    except Exception as e:
        print(f" Error during poison generation: {e}")

def run_poison_training_and_test():
    """Run poison training and testing"""
    # Check if poison images exist
    poison_dir = "/content/drive/MyDrive/FaceModel/Output/Poison_Frog_Final"
    
    if not os.path.exists(poison_dir) or len([f for f in os.listdir(poison_dir) if f.endswith('.jpg')]) == 0:
        print(f" No poison images found in {poison_dir}")
        print("   Please run option 2 (Generate Poison Images) first!")
        return
    
    print("\n Starting Poison Training & Attack Testing...")
    print("-" * 40)
    
    try:
        from poison_attack_training import main as poison_train_main
        poison_train_main()
        print(" Poison training and testing completed successfully!")
    except Exception as e:
        print(f" Error during poison training/testing: {e}")

def run_classification():
    """Run classification on new images"""
    # Check if clean model exists
    clean_model_path = "/content/drive/MyDrive/FaceModel/Output/Training/best_model.pth"
    
    if not check_model_exists(clean_model_path):
        print(f" Clean model not found at {clean_model_path}")
        print("   Please run option 1 (Train Clean Model) first!")
        return
    
    print("\n Starting Image Classification...")
    print("-" * 40)
    
    try:
        from classify import main as classify_main
        classify_main()
        print(" Classification completed successfully!")
    except Exception as e:
        print(f" Error during classification: {e}")

def run_poisoned_classification():
    """Run classification on new images using poisoned model"""
    # Check if poisoned model exists
    poisoned_model_path = "/content/drive/MyDrive/FaceModel/Output/Poisoned Training/poisoned_model.pth"
    
    if not check_model_exists(poisoned_model_path):
        print(f" Poisoned model not found at {poisoned_model_path}")
        print("   Please run option 3 (Train with Poison & Test Attack) first!")
        return
    
    print("\n Starting Poisoned Model Image Classification...")
    print("-" * 40)
    
    try:
        from classify_poisoned import main as poisoned_classify_main
        poisoned_classify_main()
        print(" Poisoned model classification completed successfully!")
    except Exception as e:
        print(f" Error during poisoned model classification: {e}")

def main():
    """Main function to run the interactive menu"""

    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice == 1:
            run_clean_training()
            
        elif choice == 2:
            run_poison_generation()
            
        elif choice == 3:
            run_poison_training_and_test()
            
        elif choice == 4:
            run_classification()
            
        elif choice == 5:
            run_poisoned_classification()
            
        elif choice == 6:

            break
        
        # Ask if user wants to continue
        if choice != 6:
            while True:
                continue_choice = input("\n🔄 Do you want to run another option? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    break
                elif continue_choice in ['n', 'no']:
                    return
                else:
                    print(" Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()