# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required dependencies
!pip install torch torchvision tqdm matplotlib pillow numpy

# Import required modules
import os
import sys
import torch

# Verify GPU availability
print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("No GPU available. Using CPU instead.")

# Add the project directory to Python path
project_dir = '/content/drive/MyDrive/FaceModel'
sys.path.append(project_dir)

# Change working directory to project directory
os.chdir(project_dir)

# Add the project directory to Python path


# Import and run the face training program
from face_training import main
from face_poison_generation import main as poison_main

if __name__ == "__main__":
    # Run the main training program
    main()
    
    # Run the poison generation program
    print("\nStarting poison generation...")
    poison_main() 