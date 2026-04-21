#################################################### PART C #################################################### 
#################################################### ###########################################################


# ==========================================
# PART C: FRUIT CLASSIFICATION (30 CLASSES)
# ==========================================
import os
import glob
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random


# Configuration
CONFIG = {
    "DATA_DIR": "/kaggle/input/food-fruit-v2/Project Data/Fruit",
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.0001,
    "EPOCHS": 10,
    "IMG_SIZE": 224,
    "NUM_WORKERS": 2,
    "CLASSES_FILE": "fruit_classes.json" # We will save class names here
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Part C Engine: Running on {device}")

# ==========================================
# CUSTOM DATASET
# ==========================================
class FruitDataset(Dataset):
    def __init__(self, root_dir, mode='Train', transform=None):
        self.transform = transform
        self.samples = []
        
        target_dir = os.path.join(root_dir, mode)
        # Get sorted class names (folders)
        self.classes = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"[{mode}] Found {len(self.classes)} Classes.")
        
        # Load Images
        for class_name in self.classes:
            # Path: Fruit/Train/Apple/Images/*.jpg
            class_path = os.path.join(target_dir, class_name)
            
            # Handle the 'Images' subfolder if it exists
            if os.path.exists(os.path.join(class_path, 'Images')):
                search_path = os.path.join(class_path, 'Images')
            else:
                search_path = class_path

            img_files = glob.glob(os.path.join(search_path, "*.*"))
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            
            for f_path in img_files:
                if f_path.lower().endswith(valid_exts):
                    self.samples.append((f_path, self.class_to_idx[class_name]))

        print(f"[{mode}] Total Images Loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224), label

# Transforms
train_ops = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_ops = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loaders
train_ds = FruitDataset(CONFIG['DATA_DIR'], mode='Train', transform=train_ops)
valid_ds = FruitDataset(CONFIG['DATA_DIR'], mode='Validation', transform=val_ops)

train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

# Save Class Names for Test Script
with open(CONFIG['CLASSES_FILE'], 'w') as f:
    json.dump(train_ds.classes, f)
print(f"Class names saved to {CONFIG['CLASSES_FILE']}")



# ==========================================
# TRAINING ENGINE
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    since = time.time()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# EXECUTION
# ==========================================
# 1. Setup Model (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
# Change output to 30 classes (Fruit)
model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

# 2. Run Training
final_model, history = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=CONFIG['EPOCHS'])

# 3. Save Model
torch.save(final_model.state_dict(), "/kaggle/working/model_part_C.pth")
print("Model Part C saved successfully.")







# ==========================================
# REPORT VISUALIZATIONS
# ==========================================

# 1. Plot Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.show()


# 2. Visualize Predictions (Updated for Randomness)
def visualize_predictions(model, loader):
    model.eval()
    
    # Get the dataset from the loader
    dataset = loader.dataset
    
    # 1. Pick 8 RANDOM indices from the entire dataset
    indices = random.sample(range(len(dataset)), 8)
    
    # 2. Retrieve the images and labels
    images_list = []
    labels_list = []
    
    for idx in indices:
        img, label = dataset[idx]
        images_list.append(img)
        labels_list.append(label)
    
    # 3. Stack them into a batch
    images = torch.stack(images_list).to(device)
    labels = torch.tensor(labels_list).to(device)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 5. Plot
    plt.figure(figsize=(15, 8))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        
        # Un-normalize for display
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # Get class names (Assumes loader.dataset has .classes attribute)
        # If dataset doesn't have .classes, use train_ds.classes directly
        class_names = dataset.classes if hasattr(dataset, 'classes') else train_ds.classes
        
        true_name = class_names[labels[i]]
        pred_name = class_names[preds[i]]
        
        color = 'green' if true_name == pred_name else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run it
visualize_predictions(final_model, valid_loader)

# 3. Measure Inference Speed (Required for Report)
def measure_speed():
    final_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10): _ = final_model(dummy_input)
    
    start = time.time()
    for _ in range(100):
        _ = final_model(dummy_input)
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000 # ms
    print(f"Average Inference Time: {avg_time:.2f} ms per image")

measure_speed()

%%writefile test_part_C.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os
import json
import argparse

# Config
MODEL_PATH = "model_part_C.pth"
CLASSES_FILE = "fruit_classes.json"
OUTPUT_FILE = "part_C_prediction.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    # 1. Load Classes
    if not os.path.exists(CLASSES_FILE):
        print(f"[Error] {CLASSES_FILE} not found.")
        sys.exit(1)
    with open(CLASSES_FILE, 'r') as f:
        classes = json.load(f)
        
    # 2. Load Model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] {MODEL_PATH} not found.")
        sys.exit(1)
        
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        sys.exit(1)
        
    model = model.to(device)
    model.eval()
    return model, classes

def predict_image(model, classes, img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, pred_idx = torch.max(outputs, 1)
            
        return classes[pred_idx.item()]
    except Exception as e:
        print(f"[Error] Processing {img_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="Path to image or folder")
    args = parser.parse_args()
    
    model, classes = load_resources()
    
    # Handle Folder or File
    if os.path.isdir(args.input):
        import glob
        files = glob.glob(os.path.join(args.input, "*.*"))
        valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f">> Processing {len(valid_files)} images in folder...")
        
        results = []
        for f in valid_files:
            pred = predict_image(model, classes, f)
            if pred:
                print(f"{os.path.basename(f)} -> {pred}")
                results.append(f"{os.path.basename(f)}: {pred}")
        
        with open(OUTPUT_FILE, "w") as f:
            f.write("\n".join(results))
            
    elif os.path.isfile(args.input):
        pred = predict_image(model, classes, args.input)
        if pred:
            print(f">> Prediction: {pred}")
            with open(OUTPUT_FILE, "w") as f:
                f.write(pred)
    
    print(f">> Results saved to {OUTPUT_FILE}")







