#################################################### PART E #################################################### 
#################################################### ###########################################################


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




# ============================================================================
# CELL 1: IMPORTS, CONFIGURATIONS, AND AUTO-CLASS DETECTION
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
import random
import json
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import segmentation_models_pytorch as smp

from sklearn.metrics import jaccard_score, accuracy_score




SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Do NOT call torch.manual_seed() in Kaggle PyTorch 2.x GPU environments
# torch.manual_seed(SEED)  # <-- skip this to avoid CUDA device-side assert

# -------------------------------
# Device setup
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'data_dir': '/kaggle/input/data-food-fruit-part-e/Project Data/Fruit',
    'output_dir': '/kaggle/working/predictions',
    'img_size': (256, 256),
    'batch_size': 16,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 2,
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

print(f"Device: {CONFIG['device']}")
print(f"Image size: {CONFIG['img_size']}")
print()




from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

def detect_classes(data_root: str) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """
    Automatically detect all fruit classes and add background as class 0.

    Args:
        data_root: Path to the root data directory (e.g., 'Project Data/Fruit')

    Returns:
        class_names: List of class names ['background', 'Apple_Gala', ...]
        class_to_idx: Dictionary mapping class names to indices
        color_map: NumPy array of RGB colors for each class
    """
    data_root = Path(data_root)
    train_dir = data_root / 'Train'
    
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Get fruit classes from subdirectories
    fruit_classes = sorted([item.name for item in train_dir.iterdir() if item.is_dir()])
    
    # Add 'background' as class 0
    class_names = ['background'] + fruit_classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    # Create a color map
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # background = black
    
    for i in range(1, num_classes):
        hue = int((i - 1) * 180 / (num_classes - 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        color_map[i] = color_rgb
    
    print("="*80)
    print("DETECTED CLASSES")
    print("="*80)
    print(f"Total classes: {num_classes} (including background)")
    for name, idx in class_to_idx.items():
        print(f"{idx:2d}: {name}")
    print("="*80)
    
    return class_names, class_to_idx, color_map
DATA_DIR = "/kaggle/input/data-food-fruit-part-e/Project Data/Fruit"

FRUIT_CLASSES, CLASS_TO_IDX, COLOR_MAP = detect_classes(DATA_DIR)
NUM_CLASSES = len(FRUIT_CLASSES)



# CELL 2: REUSABLE DATASET AND HELPER FUNCTIONS
# ============================================================================

class FruitSegmentationDataset(Dataset):
    """
    PyTorch Dataset for multi-class fruit segmentation.
    Automatically detects classes and loads image-mask pairs.
    """
    
    def __init__(
        self, 
        data_root: str,
        split: str = 'Train',
        class_names: List[str] = None,
        class_to_idx: Dict[str, int] = None,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Args:
            data_root: Path to root directory (e.g., 'Project Data/Fruit')
            split: 'Train' or 'Validation'
            class_names: List of class names
            class_to_idx: Dictionary mapping class names to indices
            img_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_names)
        
        # Load all image-mask pairs
        self.samples = self._load_samples()
        
        print(f"{split} Dataset:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Image size: {img_size}")
        print(f"  Augmentation: {augment}")
        print(f"  Normalization: {normalize}\n")
    
    def _load_samples(self) -> List[Dict]:
        """Load all image-mask pairs from the dataset."""
        samples = []
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        print(f"Scanning {split_dir}...")
        
        # Iterate through each fruit class (skip 'background')
        for class_name in self.class_names[1:]:
            class_dir = split_dir / class_name
            images_dir = class_dir / 'Images'
            masks_dir = class_dir / 'Mask'
            
            # ... (omitted directory existence checks for brevity)
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(images_dir.glob(ext)))
            image_files = sorted(image_files, key=lambda x: x.name)
            
            # Get all mask files
            mask_files = []
            for ext in image_extensions:
                mask_files.extend(list(masks_dir.glob(ext)))
            mask_files = sorted(mask_files, key=lambda x: x.name)
            
            # --- START OF CRITICAL CORRECTION ---
            
            # Map mask stem to its path. We only need one map: stem -> path.
            # E.g., '10' -> mask_path (if 10.png exists)
            # E.g., '10_mask' -> mask_path (if 10_mask.png exists)
            mask_dict = {mask_path.stem: mask_path for mask_path in mask_files}
            
            # Match images with masks
            matched = 0
            for img_path in image_files:
                img_stem = img_path.stem # e.g., '10' from '10.jpg'
                mask_path = None

                # 1. Check for the common 'stem_mask' pattern (e.g., '10' -> '10_mask')
                # This is the pattern confirmed by your sample data.
                if img_stem + '_mask' in mask_dict:
                    mask_path = mask_dict[img_stem + '_mask']
                    
                # 2. Also check for exact stem match (e.g., '10' -> '10') 
                # This handles cases where the image and mask files share the exact same stem (e.g., '10.jpg' and '10.png')
                elif img_stem in mask_dict:
                    mask_path = mask_dict[img_stem]
                
                # If a match is found, append it to samples
                if mask_path:
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
                    matched += 1
            
            # --- END OF CRITICAL CORRECTION ---
            
            if matched > 0:
                print(f"  ✓ {class_name}: Found {matched} image-mask pairs")
            # ... (omitted warning print for brevity)
            
        # ... (omitted error handling for brevity)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {sample['mask_path']}")
        
        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        
        # Convert mask to class indices
        mask_class = np.zeros_like(mask, dtype=np.int64)  # background = 0
        mask_class[mask > 127] = sample['class_idx']      # assign fruit class

        
        # Apply augmentation if enabled
        if self.augment:
            # CORRECTED: Used mask_class instead of undefined mask_np
            image, mask_class = self._augment(image, mask_class) 
        
        # Convert image to float and normalize
        # ⬇️ ERROR LINE 156 MUST BE HERE
        image = image.astype(np.float32)
        if self.normalize:
            image = image / 255.0
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        # CORRECTED: Used mask_class instead of undefined mask_np
        mask_tensor = torch.from_numpy(mask_class).long() 
        
        return image_tensor, mask_tensor
    
    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {class_name: 0 for class_name in self.class_names}
        for sample in self.samples:
            distribution[sample['class_name']] += 1
        return distribution


def load_data(data_dir, split, class_names, class_to_idx, img_size, augment=False):
    """Load dataset for a given split."""
    dataset = FruitSegmentationDataset(
        data_dir, split, class_names, class_to_idx, img_size, augment
    )
    return dataset


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """Create a DataLoader from dataset."""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=(shuffle)  # Drop last batch for training
    )


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name):
    """Generic training function for any segmentation model."""
    model = model.to(device)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle different model output formats
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss, val_iou = evaluate_model(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val IoU: {val_iou:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'/kaggle/working/{model_name}_best.pth')
            print(f'✓ Saved best model with val_loss: {val_loss:.4f}')
    
    return model, history


def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    val_loss = 0.0
    all_ious = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            for pred, mask in zip(preds_np, masks_np):
                iou = compute_iou(pred, mask, NUM_CLASSES)
                all_ious.append(iou)
    
    val_loss /= len(val_loader)
    mean_iou = np.mean(all_ious)
    
    return val_loss, mean_iou


def compute_iou(pred, target, num_classes):
    """Compute mean IoU across all classes."""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def compute_dice(pred, target, num_classes):
    """Compute mean Dice score across all classes."""
    dice_scores = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        dice = (2.0 * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-7)
        dice_scores.append(dice)
    
    return np.mean(dice_scores)


def predict_mask(model, image_path, img_size, device):
    """Predict segmentation mask for a single image."""
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    
    # Resize for model
    image_resized = cv2.resize(image_rgb, img_size)
    image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        
        if isinstance(output, dict):
            output = output['out']
        
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize mask back to original size
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8), 
        (original_size[1], original_size[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create colored mask
    colored_mask = COLOR_MAP[pred_mask]
    
    return image_rgb, pred_mask, colored_mask


def visualize_prediction(image, mask, colored_mask, save_path=None):
    """Visualize prediction with original image, mask, and overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def save_mask(mask, save_path):
    """Save predicted mask to file."""
    colored_mask = COLOR_MAP[mask]
    cv2.imwrite(save_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print(f'Saved mask to {save_path}')




# CELL 3: DEBUG - INSPECT DATASET STRUCTURE
# ============================================================================

print("\n" + "="*80)
print("INSPECTING DATASET STRUCTURE")
print("="*80 + "\n")

def inspect_dataset_structure(data_dir, num_classes_to_check=3):
    """Inspect the dataset structure to diagnose issues."""
    data_root = Path(data_dir)
    
    for split in ['Train', 'Validation']:
        split_dir = data_root / split
        print(f"\n{split} Directory: {split_dir}")
        print(f"Exists: {split_dir.exists()}")
        
        if not split_dir.exists():
            continue
        
        # Get fruit directories
        fruit_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"Found {len(fruit_dirs)} fruit directories")
        
        # Check first few fruits
        for fruit_dir in fruit_dirs[:num_classes_to_check]:
            print(f"\n  Fruit: {fruit_dir.name}")
            
            images_dir = fruit_dir / 'Images'
            masks_dir = fruit_dir / 'Mask'
            
            print(f"    Images dir exists: {images_dir.exists()}")
            print(f"    Mask dir exists: {masks_dir.exists()}")
            
            if images_dir.exists():
                image_files = list(images_dir.glob('*'))
                print(f"    Number of files in Images: {len(image_files)}")
                if image_files:
                    print(f"    Sample image files: {[f.name for f in image_files[:3]]}")
            
            if masks_dir.exists():
                mask_files = list(masks_dir.glob('*'))
                print(f"    Number of files in Mask: {len(mask_files)}")
                if mask_files:
                    print(f"    Sample mask files: {[f.name for f in mask_files[:3]]}")
                    
                    # Check if mask names match image names
                    if image_files and mask_files:
                        img_stems = {f.stem for f in image_files}
                        mask_stems = {f.stem for f in mask_files}
                        matching = img_stems.intersection(mask_stems)
                        print(f"    Matching image-mask pairs: {len(matching)}")
                        if len(matching) < len(image_files):
                            print(f"    ⚠️ Warning: Not all images have matching masks!")
                            non_matching = img_stems - mask_stems
                            print(f"    Images without masks: {list(non_matching)[:5]}")

# Run inspection
inspect_dataset_structure(CONFIG['data_dir'])

print("\n" + "="*80)





# CELL 4: LOAD DATA AND CREATE DATALOADERS
# ============================================================================

print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80 + "\n")

# Load datasets
train_dataset = load_data(
    CONFIG['data_dir'], 
    'Train', 
    FRUIT_CLASSES, 
    CLASS_TO_IDX,
    CONFIG['img_size'], 
    augment=True
)

val_dataset = load_data(
    CONFIG['data_dir'], 
    'Validation', 
    FRUIT_CLASSES,
    CLASS_TO_IDX,
    CONFIG['img_size'], 
    augment=False
)

# Create data loaders
train_loader = get_dataloader(
    train_dataset, 
    CONFIG['batch_size'], 
    shuffle=True,
    num_workers=CONFIG['num_workers']
)

val_loader = get_dataloader(
    val_dataset, 
    CONFIG['batch_size'], 
    shuffle=False,
    num_workers=CONFIG['num_workers']
)

# Print class distribution
print("="*80)
print("CLASS DISTRIBUTION")
print("="*80)

train_dist = train_dataset.get_class_distribution()
val_dist = val_dataset.get_class_distribution()

print(f"\n{'Class':<30} {'Train Samples':<15} {'Val Samples':<15}")
print("-"*60)
for class_name in FRUIT_CLASSES:
    train_count = train_dist.get(class_name, 0)
    val_count = val_dist.get(class_name, 0)
    print(f"{class_name:<30} {train_count:<15} {val_count:<15}")

print("-"*60)
print(f"{'Total':<30} {len(train_dataset):<15} {len(val_dataset):<15}")
print("="*80 + "\n")

# Test loading a batch
print("Testing data loading...")
images, masks = next(iter(train_loader))
print(f"  Image batch shape: {images.shape}")
print(f"  Mask batch shape: {masks.shape}")
print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
print(f"  Mask unique classes: {torch.unique(masks).tolist()}")
print(f"  ✓ Data loading successful!\n")





# CELL 5: DEFINE AND TRAIN MODEL 1 (U-Net)
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: U-Net with ResNet34 Encoder")
print("="*80 + "\n")

# Define U-Net model
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_unet = optim.Adam(unet_model.parameters(), lr=CONFIG['learning_rate'])

# Train Model 1
unet_model, unet_history = train_model(
    model=unet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_unet,
    num_epochs=CONFIG['num_epochs'],
    device=CONFIG['device'],
    model_name='unet'
)

print("\n✓ U-Net training completed!")




# CELL 6: DEFINE AND TRAIN MODEL 2 (DeepLabV3+)
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: DeepLabV3+ with ResNet50 Encoder")
print("="*80 + "\n")

# Define DeepLabV3+ model
deeplabv3_model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)

# Define optimizer
optimizer_deeplab = optim.Adam(deeplabv3_model.parameters(), lr=CONFIG['learning_rate'])

# Train Model 2
deeplabv3_model, deeplab_history = train_model(
    model=deeplabv3_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_deeplab,
    num_epochs=CONFIG['num_epochs'],
    device=CONFIG['device'],
    model_name='deeplabv3plus'
)

print("\n✓ DeepLabV3+ training completed!")




# CELL 7: EVALUATE BOTH MODELS AND SELECT BEST
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON AND SELECTION")
print("="*80 + "\n")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
axes[0].plot(unet_history['train_loss'], label='U-Net Train', marker='o')
axes[0].plot(unet_history['val_loss'], label='U-Net Val', marker='o')
axes[0].plot(deeplab_history['train_loss'], label='DeepLabV3+ Train', marker='s')
axes[0].plot(deeplab_history['val_loss'], label='DeepLabV3+ Val', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# IoU curves
axes[1].plot(unet_history['val_iou'], label='U-Net', marker='o')
axes[1].plot(deeplab_history['val_iou'], label='DeepLabV3+', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean IoU')
axes[1].set_title('Validation IoU')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/training_comparison.png', dpi=150)
plt.show()

# Comprehensive evaluation
print("\nComprehensive Evaluation on Validation Set:\n")

models = {
    'U-Net': unet_model,
    'DeepLabV3+': deeplabv3_model
}

results = {}

for model_name, model in models.items():
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f'Evaluating {model_name}'):
            images = images.to(CONFIG['device'])
            outputs = model(images)
            
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    iou = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_targets, all_preds)
    dice = compute_dice(all_preds, all_targets, NUM_CLASSES)
    
    results[model_name] = {
        'IoU': iou,
        'Dice': dice,
        'Accuracy': accuracy
    }
    
    print(f"{model_name}:")
    print(f"  Mean IoU: {iou:.4f}")
    print(f"  Dice Score: {dice:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print()

# Select best model
best_model_name = max(results, key=lambda x: results[x]['IoU'])
best_model = models[best_model_name]

print(f"{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"  Mean IoU: {results[best_model_name]['IoU']:.4f}")
print(f"  Dice Score: {results[best_model_name]['Dice']:.4f}")
print(f"  Accuracy: {results[best_model_name]['Accuracy']:.4f}")
print(f"{'='*80}\n")

# Save results
with open('/kaggle/working/model_comparison.json', 'w') as f:
    json.dump(results, f, indent=4)





# CELL 8: PREDICT ON NEW IMAGES AND VISUALIZE
# ============================================================================

print("\n" + "="*80)
print("PREDICTIONS ON TEST IMAGES")
print("="*80 + "\n")

# Get sample validation images
sample_images = []
for fruit_name in FRUIT_CLASSES[1:min(6, len(FRUIT_CLASSES))]:
    fruit_dir = Path(CONFIG['data_dir']) / 'Validation' / fruit_name / 'Images'
    if fruit_dir.exists():
        img_files = list(fruit_dir.glob('*.jpg')) + list(fruit_dir.glob('*.png'))
        if img_files:
            sample_images.append(str(img_files[0]))

if len(sample_images) == 0:
    print("Warning: No sample images found in validation set")
else:
    print(f"Predicting on {len(sample_images)} sample images...\n")
    
    for idx, img_path in enumerate(sample_images):
        print(f"Processing image {idx+1}/{len(sample_images)}: {Path(img_path).name}")
        
        # Predict
        image, pred_mask, colored_mask = predict_mask(
            best_model, img_path, CONFIG['img_size'], CONFIG['device']
        )
        
        # Visualize
        save_path = f'/kaggle/working/prediction_{idx+1}.png'
        visualize_prediction(image, pred_mask, colored_mask, save_path)
        
        # Print detected fruits
        unique_classes = np.unique(pred_mask)
        detected_fruits = [FRUIT_CLASSES[c] for c in unique_classes if c > 0]
        print(f"  Detected fruits: {', '.join(detected_fruits)}")
        print()


# ============================================================================
# CELL 9: SAVE PREDICTED MASKS
# ============================================================================

print("\n" + "="*80)
print("SAVING PREDICTED MASKS")
print("="*80 + "\n")

# Create masks directory
masks_dir = os.path.join(CONFIG['output_dir'], 'masks')
os.makedirs(masks_dir, exist_ok=True)

if len(sample_images) > 0:
    # Save all predicted masks
    for idx, img_path in enumerate(sample_images):
        print(f"Saving mask {idx+1}/{len(sample_images)}")
        
        # Predict
        _, pred_mask, _ = predict_mask(
            best_model, img_path, CONFIG['img_size'], CONFIG['device']
        )
        
        # Save mask
        mask_filename = f'mask_{Path(img_path).stem}.png'
        mask_save_path = os.path.join(masks_dir, mask_filename)
        save_mask(pred_mask, mask_save_path)
    
    print(f"\n✓ All masks saved to: {masks_dir}")
    print(f"✓ Total masks saved: {len(sample_images)}")
    
    # Create summary visualization
    print("\nCreating summary visualization...")
    
    fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4*len(sample_images)))
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(sample_images):
        image, pred_mask, colored_mask = predict_mask(
            best_model, img_path, CONFIG['img_size'], CONFIG['device']
        )
        
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Image {idx+1}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(colored_mask)
        axes[idx, 1].set_title(f'Mask {idx+1}')
        axes[idx, 1].axis('off')
        
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay {idx+1}')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/all_predictions_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*80)
print("NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nOutputs saved to: {CONFIG['output_dir']}")
print(f"Best model: {best_model_name}")
print(f"Best model metrics:")
print(f"  - Mean IoU: {results[best_model_name]['IoU']:.4f}")
print(f"  - Dice Score: {results[best_model_name]['Dice']:.4f}")
print(f"  - Accuracy: {results[best_model_name]['Accuracy']:.4f}")
print(f"\nModel checkpoints saved to: /kaggle/working/")
print(f"  - {best_model_name.lower().replace('+', 'plus')}_best.pth")
print("="*80)



# The single image path you want to use
SINGLE_TEST_IMAGE_PATH = "/kaggle/input/test-123/fruit.PNG"

# NOTE: You MUST ensure 'best_model' is defined and loaded from the checkpoint
# before running this cell (e.g., unet_best.pth).

# CELL 8: PREDICT ON NEW IMAGES AND VISUALIZE
# ============================================================================
print("\n" + "="*80)
print("PREDICTIONS ON TEST IMAGES")
print("="*80 + "\n")

# --- MODIFICATION START ---
# Replace sampling logic with the single provided path
sample_images = [SINGLE_TEST_IMAGE_PATH]
# --- MODIFICATION END ---

if len(sample_images) == 0:
    print(f"Error: Single image path not provided or empty list: {SINGLE_TEST_IMAGE_PATH}")
else:
    print(f"Predicting on single test image: {Path(sample_images[0]).name}\n")
    
    # The rest of the loop remains the same, but it will only run once
    for idx, img_path in enumerate(sample_images):
        print(f"Processing image {idx+1}/{len(sample_images)}: {Path(img_path).name}")
        
        # Predict
        # NOTE: You must ensure 'best_model' is defined. We will assume you 
        # load the weights of the best trained U-Net here into a variable called 'best_model'.
        image, pred_mask, colored_mask = predict_mask(
            best_model, img_path, CONFIG['img_size'], CONFIG['device']
        )
        
        # Visualize
        save_path = f'/kaggle/working/prediction_{idx+1}.png'
        visualize_prediction(image, pred_mask, colored_mask, save_path)
        
        # Print detected fruits
        unique_classes = np.unique(pred_mask)
        detected_fruits = [FRUIT_CLASSES[c] for c in unique_classes if c > 0]
        print(f"  Detected fruits: {', '.join(detected_fruits)}")
        print()

# ============================================================================
# CELL 9: SAVE PREDICTED MASKS
# ============================================================================
print("\n" + "="*80)
print("SAVING PREDICTED MASKS")
print("="*80 + "\n")

# Create masks directory
masks_dir = os.path.join(CONFIG['output_dir'], 'masks')
os.makedirs(masks_dir, exist_ok=True)

if len(sample_images) > 0:
    # Save all predicted masks
    for idx, img_path in enumerate(sample_images):
        print(f"Saving mask {idx+1}/{len(sample_images)}")
        
        # Predict
        _, pred_mask, _ = predict_mask(
            best_model, img_path, CONFIG['img_size'], CONFIG['device']
        )
        
        # Save mask
        mask_filename = f'mask_{Path(img_path).stem}.png'
        mask_save_path = os.path.join(masks_dir, mask_filename)
        save_mask(pred_mask, mask_save_path)
    
    print(f"\n✓ All masks saved to: {masks_dir}")
    print(f"✓ Total masks saved: {len(sample_images)}")

    # Create summary visualization
    print("\nCreating summary visualization...")
    
    # Since len(sample_images) is 1, we set up the plot accordingly
    fig, axes = plt.subplots(1, 3, figsize=(12, 4)) 
    
    # We only need to iterate once, using the single image path
    image, pred_mask, colored_mask = predict_mask(
        best_model, SINGLE_TEST_IMAGE_PATH, CONFIG['img_size'], CONFIG['device']
    )
    
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image: {Path(SINGLE_TEST_IMAGE_PATH).name}')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
        
    plt.tight_layout()
    plt.savefig('/kaggle/working/all_predictions_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*80)
print("NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
# (Omitted final print statements as they refer to undefined variables like 'results')




