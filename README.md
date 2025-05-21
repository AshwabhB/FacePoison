# Face Recognition and Poison Attack Project Documentation

This project implements a face recognition system with poison attack capabilities. The system uses a sophisticated neural network architecture with attention mechanisms and feature pyramid networks for face classification, and implements a poison frog attack to manipulate model behavior.

## Table of Contents
1. [classify_poisoned.py](#classify_poisonedpy)
2. [classify.py](#classifypy)
3. [face_data_preparation.py](#face_data_preparationpy)
4. [face_model.py](#face_modelpy)
5. [face_poison_generation.py](#face_poison_generationpy)
6. [face_training.py](#face_trainingpy)
7. [poison_attack_training.py](#poison_attack_trainingpy)

## Project Overview

### Face Recognition System
The system uses a ResNet101 backbone with additional attention mechanisms and feature pyramid networks for robust face recognition. Key features include:

1. **Channel Attention**: Focuses on important feature channels
2. **Spatial Attention**: Identifies important spatial regions
3. **Feature Pyramid Network**: Multi-scale feature representation
4. **Custom Classification Head**: Optimized for face recognition

### Poison Attack Implementation
The poison frog attack is implemented with the following components:

1. **Feature Matching**: Aligns poisoned image features with target class
2. **Visual Preservation**: Maintains visual similarity to base image
3. **Curriculum Learning**: Gradually increases poison ratio during training
4. **Class Weighting**: Emphasizes important classes in training

## Mathematical Details

### Poison Frog Attack Loss Function
The attack uses a composite loss function:

L = ||f(x_poison) - f(x_target)||² + β||x_poison - x_base||²

Where:
- f(x_poison): Features of poisoned image
- f(x_target): Features of target class image
- x_poison: Poisoned image
- x_base: Base image
- β: Visual preservation weight

**Components Explanation:**
1. **Feature Matching Term** (||f(x_poison) - f(x_target)||²):
   - Forces poisoned image to have similar features to target class
   - Uses L2 norm to measure feature distance
   - Larger values indicate greater feature mismatch

2. **Visual Preservation Term** (β||x_poison - x_base||²):
   - Maintains visual similarity to base image
   - β controls trade-off between feature matching and visual preservation
   - Higher β values result in more visually similar but less effective poisons
   - Lower β values allow more visual distortion but better feature matching

### Perturbation Constraint
The attack enforces an L∞ constraint on the perturbation:

||x_poison - x_base||∞ ≤ ε

Where:
- ε (epsilon): Maximum allowed perturbation (default: 32/255)
- L∞ norm: Maximum absolute difference between any pixel values
- 32/255 ≈ 0.125: Maximum 12.5% change in any pixel value

**Effects of ε:**
- Larger ε: Allows more aggressive poisoning but more visible artifacts
- Smaller ε: More subtle poisoning but potentially less effective
- 32/255 is chosen as a balance between effectiveness and stealth

## classify_poisoned.py

### Functions

#### classify_new_images_poisoned(model, new_images_dir, idx_to_class, device, save_path=None)
Classifies new images using a poisoned model.

**Parameters:**
- `model`: The poisoned face classification model
- `new_images_dir`: Directory containing new images to classify (default: "/content/drive/MyDrive/FaceModel/NewPerson")
- `idx_to_class`: Dictionary mapping class indices to class names
- `device`: Device to run inference on ('cuda' or 'cpu')
- `save_path`: Optional path to save classification results (default: None)

**Dependencies:**
- Uses `get_transforms()` from face_data_preparation.py
- Uses PIL for image loading
- Uses matplotlib for visualization

**Output:**
- Displays classification results with probabilities
- Saves visualization if save_path is provided

**Visualization Details:**
- Creates a grid of images with predictions
- Shows predicted class and confidence
- Normalizes images for display
- Adds padding between subplots
- Uses 3 columns for layout

#### main()
Main execution function for poisoned classification.

**Key Operations:**
1. Sets up device (CUDA/CPU)
2. Creates output directory at "/content/drive/MyDrive/FaceModel/Output/Poisoned_Classification"
3. Loads poisoned model from "/content/drive/MyDrive/FaceModel/Output/Poisoned Training/poisoned_model.pth"
4. Loads class mappings from dataset
5. Performs classification on new images

**Error Handling:**
- Checks for model existence
- Verifies class mappings
- Handles device compatibility
- Provides detailed error messages

## classify.py

### Functions

#### classify_new_images(model, new_images_dir, idx_to_class, device, save_path=None)
Classifies new images using a clean (non-poisoned) model.

**Parameters:**
- `model`: The face classification model
- `new_images_dir`: Directory containing new images to classify (default: "/content/drive/MyDrive/FaceModel/NewPerson")
- `idx_to_class`: Dictionary mapping class indices to class names
- `device`: Device to run inference on ('cuda' or 'cpu')
- `save_path`: Optional path to save classification results (default: None)

**Dependencies:**
- Uses `get_transforms()` from face_data_preparation.py
- Uses PIL for image loading
- Uses matplotlib for visualization

**Output:**
- Displays classification results with probabilities
- Saves visualization if save_path is provided

**Processing Steps:**
1. Loads and transforms images
2. Performs inference
3. Applies softmax for probabilities
4. Visualizes results
5. Optionally saves output

#### main()
Main execution function for clean classification.

**Key Operations:**
1. Sets up device (CUDA/CPU)
2. Creates output directory at "/content/drive/MyDrive/FaceModel/Output/Classification"
3. Loads model from "/content/drive/MyDrive/FaceModel/Output/Training/best_model.pth"
4. Loads class mappings from dataset
5. Performs classification on new images

**Model Loading:**
- Handles different model architectures
- Supports both classifier.6.weight and backbone.fc.3.weight formats
- Uses strict=False for compatibility

## face_data_preparation.py

### Classes

#### FaceDataset(Dataset)
Custom dataset class for face images.

**Parameters:**
- `root_dir`: Root directory containing face images
- `transform`: Optional transforms to apply to images

**Methods:**
- `__init__`: Initializes dataset and creates class mappings
- `__len__`: Returns number of images
- `__getitem__`: Returns image and label for given index

**Class Mapping:**
- Extracts person name from filename
- Creates unique class indices
- Handles spaces in names
- Maintains consistent mapping

#### TransformedDataset(Dataset)
Wrapper dataset class for applying transforms.

**Parameters:**
- `dataset`: Base dataset
- `transform`: Transforms to apply

**Purpose:**
- Applies transforms to dataset items
- Maintains original dataset structure
- Enables dynamic transform changes

### Functions

#### get_transforms(is_train=True)
Returns appropriate transforms for training or validation.

**Parameters:**
- `is_train`: Whether transforms are for training (default: True)

**Training Transforms:**
- Resize to (224, 224)
- Random horizontal flip
- Random rotation (±10 degrees)
- Random affine translation (0.1)
- Color jitter (brightness, contrast, saturation, hue)
- Normalize with ImageNet stats

**Validation Transforms:**
- Resize to (224, 224)
- Normalize with ImageNet stats

**Normalization Values:**
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

#### load_face_dataset(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
Loads and splits face dataset.

**Parameters:**
- `data_dir`: Directory containing face images
- `batch_size`: Batch size for dataloaders (default: 32)
- `train_ratio`: Ratio of training data (default: 0.7)
- `val_ratio`: Ratio of validation data (default: 0.15)
- `test_ratio`: Ratio of test data (default: 0.15)

**Returns:**
- train_dataset, val_dataset, test_dataset, class_to_idx

**DataLoader Settings:**
- Shuffle: True for training, False for validation/test
- num_workers: 2
- pin_memory: True if CUDA available

## face_model.py

### Classes

#### ChannelAttention(nn.Module)
Channel attention module.

**Parameters:**
- `in_channels`: Number of input channels
- `reduction_ratio`: Channel reduction ratio (default: 16)

**Architecture:**
1. Global average pooling
2. Global max pooling
3. Two fully connected layers
4. ReLU activation
5. Sigmoid output

#### SpatialAttention(nn.Module)
Spatial attention module.

**Parameters:**
- `kernel_size`: Convolution kernel size (default: 7)

**Architecture:**
1. Average pooling across channels
2. Max pooling across channels
3. Concatenation
4. 7x7 convolution
5. Sigmoid output

#### FeaturePyramid(nn.Module)
Feature Pyramid Network module.

**Parameters:**
- `in_channels_list`: List of input channel sizes
- `out_channels`: Number of output channels

**Architecture:**
1. Lateral connections
2. Top-down pathway
3. FPN convolutions
4. Feature fusion

#### FaceClassifier(nn.Module)
Main face classification model.

**Parameters:**
- `num_classes`: Number of output classes

**Architecture:**
- Backbone: ResNet101 (pretrained)
- Feature Pyramid Network
- Channel and Spatial Attention
- Custom classification head

**Classification Head:**
1. Linear layer (num_features + 256*3 → 1024)
2. ReLU activation
3. Dropout (0.5)
4. Linear layer (1024 → 512)
5. ReLU activation
6. Dropout (0.3)
7. Linear layer (512 → num_classes)

### Functions

#### get_model(num_classes, device='cuda')
Creates and returns face classification model.

**Parameters:**
- `num_classes`: Number of output classes
- `device`: Device to place model on (default: 'cuda')

**Model Initialization:**
1. Creates FaceClassifier instance
2. Moves model to specified device
3. Freezes early ResNet layers
4. Sets up feature hooks

## face_poison_generation.py

### Classes

#### FaceDataset(Dataset)
Dataset class for poison generation.

**Parameters:**
- `root_dir`: Directory containing images
- `transform`: Optional transforms

**Purpose:**
- Loads base images for poisoning
- Applies necessary transforms
- Maintains image metadata

### Functions

#### load_trained_model(model_path, device)
Loads trained face classification model.

**Parameters:**
- `model_path`: Path to model checkpoint
- `device`: Device to load model on

**Loading Process:**
1. Loads checkpoint
2. Determines num_classes from weights
3. Initializes model
4. Loads state dict
5. Sets model to eval mode

#### extract_features(model, x)
Extracts features from model.

**Parameters:**
- `model`: Face classification model
- `x`: Input tensor

**Feature Extraction:**
1. Registers hooks for layers
2. Performs forward pass
3. Captures intermediate features
4. Removes hooks
5. Returns feature dictionary

#### poison_frog_loss(poison_img, base_img, target_features, poison_features, beta=0.1)
Computes poison frog attack loss.

**Parameters:**
- `poison_img`: Poisoned image
- `base_img`: Base image
- `target_features`: Target class features
- `poison_features`: Poisoned image features
- `beta`: Weight for visual preservation (default: 0.1)

**Loss Components:**
1. Feature matching loss
2. Visual preservation loss
3. Feature similarity monitoring

**Beta Effects:**
- β = 0.1: Balanced attack
- β < 0.1: More aggressive feature matching
- β > 0.1: More visual preservation

#### generate_poison_samples(model, base_images, target_features, class_to_idx, device, beta=0.1, num_iters=500, lr=0.01, epsilon=32/255)
Generates poison samples.

**Parameters:**
- `model`: Face classification model
- `base_images`: List of base images
- `target_features`: Target class features
- `class_to_idx`: Class mapping
- `device`: Device to use
- `beta`: Visual preservation weight (default: 0.1)
- `num_iters`: Number of optimization iterations (default: 500)
- `lr`: Learning rate (default: 0.01)
- `epsilon`: Maximum perturbation (default: 32/255)

**Generation Process:**
1. Initializes poison images
2. Optimizes with Adam
3. Applies L∞ constraint
4. Tracks best results
5. Saves progress

**Output:**
- Poisoned images
- Progress visualizations
- Final statistics

## face_training.py

### Classes

#### FocalLoss(nn.Module)
Focal loss implementation.

**Parameters:**
- `alpha`: Weight factor (default: 1)
- `gamma`: Focusing parameter (default: 2)

**Loss Formula:**
FL(pt) = -α(1-pt)^γ log(pt)

Where:
- pt: Model's predicted probability for true class
- α: Weight factor
- γ: Focusing parameter

**Effects:**
- Higher γ: More focus on hard examples
- Higher α: More emphasis on rare classes

#### LabelSmoothingLoss(nn.Module)
Label smoothing loss implementation.

**Parameters:**
- `smoothing`: Smoothing factor (default: 0.1)

**Formula:**
LS(y, p) = (1-ε)CE(y, p) + ε/K

Where:
- ε: Smoothing factor
- K: Number of classes
- CE: Cross entropy

### Functions

#### mixup_data(x, y, alpha=0.2)
Performs mixup data augmentation.

**Parameters:**
- `x`: Input images
- `y`: Labels
- `alpha`: Mixup parameter (default: 0.2)

**Process:**
1. Samples λ from Beta(α, α)
2. Mixes images: λx + (1-λ)x'
3. Mixes labels: λy + (1-λ)y'

#### train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
Trains model for one epoch.

**Parameters:**
- `model`: Face classification model
- `train_loader`: Training data loader
- `criterion`: Loss function
- `optimizer`: Optimizer
- `device`: Device to use
- `use_mixup`: Whether to use mixup (default: True)

**Training Process:**
1. Sets model to train mode
2. Iterates through batches
3. Applies mixup if enabled
4. Computes loss
5. Updates weights
6. Tracks metrics

#### validate(model, val_loader, criterion, device)
Validates model.

**Parameters:**
- `model`: Face classification model
- `val_loader`: Validation data loader
- `criterion`: Loss function
- `device`: Device to use

**Validation Process:**
1. Sets model to eval mode
2. Disables gradients
3. Computes loss
4. Tracks accuracy
5. Returns metrics

## poison_attack_training.py

### Classes

#### FocalLoss(nn.Module)
Focal loss with reduction options.

**Parameters:**
- `alpha`: Weight factor (default: 1)
- `gamma`: Focusing parameter (default: 2)
- `reduction`: Loss reduction method (default: 'mean')

**Reduction Options:**
- 'mean': Average loss
- 'sum': Sum of losses
- 'none': Individual losses

#### FaceDatasetWithPoison(Dataset)
Dataset class for training with poisoned samples.

**Parameters:**
- `root_dir`: Directory containing clean images
- `poison_dir`: Directory containing poisoned images
- `transform`: Optional transforms
- `poison_ratio`: Ratio of poisoned samples (default: 0.0)

**Poison Integration:**
1. Loads clean images
2. Adds poisoned samples
3. Maintains poison flags
4. Applies transforms

### Functions

#### get_class_weights(dataset)
Calculates class weights.

**Parameters:**
- `dataset`: FaceDatasetWithPoison instance

**Weight Calculation:**
1. Counts per-class samples
2. Computes inverse frequency
3. Normalizes weights
4. Emphasizes target classes

#### train_with_poison(clean_dataset_path, poison_images_path, model_save_path, num_epochs=40)
Trains model with poisoned data.

**Parameters:**
- `clean_dataset_path`: Path to clean dataset
- `poison_images_path`: Path to poisoned images
- `model_save_path`: Path to save model
- `num_epochs`: Number of training epochs (default: 40)

**Training Process:**
1. Curriculum learning with increasing poison ratios
2. Class-weighted loss
3. Extra weight for poisoned samples
4. Learning rate scheduling

**Poison Ratios:**
- 0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0
- Each ratio trained for num_epochs/6 epochs

#### test_attack_success(model, test_images_dir, class_to_idx, device)
Tests poison attack success rate.

**Parameters:**
- `model`: Trained model
- `test_images_dir`: Directory containing test images
- `class_to_idx`: Class mapping
- `device`: Device to use

**Metrics:**
- Attack success rate (Alia misclassified as Anushka)
- Base class accuracy (Anushka classification accuracy)

**Success Criteria:**
- Excellent: >70% attack success rate
- Good: >50% attack success rate
- Failed: ≤50% attack success rate

### Main Function

The main function orchestrates the poison attack training and testing process:

1. Sets up device and paths
2. Trains model with poisoned data
3. Tests attack success
4. Reports results

**Expected Outputs:**
1. Trained poisoned model
2. Attack success statistics
3. Classification results
4. Performance metrics

**Success Indicators:**
1. High attack success rate (>50%)
2. Maintained base class accuracy
3. Successful misclassification of target images
4. Stable training process 