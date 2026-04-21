#################################################### PART D #################################################### 
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


import os
import cv2
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from albumentations import Compose, Resize, Normalize
import torch.optim as optim
import torchvision



def print_folders_only(root_path, indent=0):
    """Recursively prints only folders, not files."""
    for item in sorted(os.listdir(root_path)):
        item_path = os.path.join(root_path, item)

        # Only print folders
        if os.path.isdir(item_path):
            print("  " * indent + f"- {item}")
            print_folders_only(item_path, indent + 1)

# -----------------------------
# 📌 CHANGE THIS to your dataset path
dataset_path = "/kaggle/input/food-fruit"
# -----------------------------

print("\n📁 DATASET STRUCTURE (FOLDERS ONLY):\n")
print_folders_only(dataset_path)






def count_files(path):
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

dataset_path = "/kaggle/input/food-fruit"

print("\n📊 DATASET SUMMARY\n")

for category in sorted(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    print(f"\n=== {category.upper()} ===")

    for subfolder in sorted(os.listdir(category_path)):
        sub_path = os.path.join(category_path, subfolder)

        if os.path.isdir(sub_path):
            print(f"  - {subfolder}: {count_files(sub_path)} files")






dataset_path = "/kaggle/input/food-fruit/Project Data/Fruit"
train_path = dataset_path + "/Train"
val_path = dataset_path + "/Validation"

# List all fruit classes in Train folder
class_names = sorted(os.listdir(train_path))
print("Classes found:", class_names)

def show_random_sample(class_path):
    img_dir = os.path.join(class_path, "Images")
    mask_dir = os.path.join(class_path, "Mask")

    files = os.listdir(img_dir)
    filename = random.choice(files)

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename.replace(".jpg", "_mask.png"))

    # Load image and mask
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    plt.show()

print("\nShowing 3 random samples...\n")

for c in random.sample(class_names, 3):
    print(f"\nClass: {c}\n")
    class_path = os.path.join(train_path, c)
    show_random_sample(class_path)




# Paths
dataset_path = "/kaggle/input/food-fruit/Project Data/Fruit"
train_path = dataset_path + "/Train"
val_path = dataset_path + "/Validation"

# ---------------------------------------------
# 1) TRANSFORMS (Resize 512 + Normalize + Augment)
# ---------------------------------------------

# Training Transform
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Validation Transform (No augmentation)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ---------------------------------------------
# 2) DATASET CLASS
# ---------------------------------------------
class FruitSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        
        for cls in classes:
            img_dir = os.path.join(root_dir, cls, "Images")
            mask_dir = os.path.join(root_dir, cls, "Mask")

            for filename in os.listdir(img_dir):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(img_dir, filename)
                    mask_filename = filename.replace(".jpg", "_mask.png")
                    mask_path = os.path.join(mask_dir, mask_filename)
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
    
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
    
        # Convert mask to binary and float
        mask = (mask > 127).float()
    
        # Ensure mask has channel dimension (1, H, W)
        if mask.ndim == 2:  # if shape is (H, W)
            mask = mask.unsqueeze(0)  # add channel dimension
    
        return image, mask

# ---------------------------------------------
# 3) LOAD DATASETS
# ---------------------------------------------
train_dataset = FruitSegmentationDataset(train_path, transform=train_transform)
val_dataset = FruitSegmentationDataset(val_path, transform=val_transform)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# ---------------------------------------------
# 4) DATALOADERS
# ---------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("Dataloaders ready.")





def visualize_samples(dataset, num_samples=4):
    plt.figure(figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]

        # Convert tensor → numpy
        img = image.permute(1, 2, 0).numpy()
        # Undo normalization
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask_np = mask.numpy()
        if mask_np.shape[0] == 1:  # remove channel dimension for plotting
            mask_np = mask_np[0]

        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img)
        plt.title(f"Image #{idx}")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Binary Mask")
        plt.axis("off")

    plt.show()


# Run visualization for TRAIN dataset
visualize_samples(train_dataset, num_samples=4)






def visualize_prediction(model, dataset, index=0, device="cuda"):
    """
    Shows:
    - Original image
    - Ground truth mask
    - Model predicted mask
    """
    model.eval()

    # Load one sample
    image, mask = dataset[index]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

        # For binary segmentation → sigmoid → threshold
        if output.shape[1] == 1:
            pred = torch.sigmoid(output)[0,0].cpu().numpy()
            pred_mask = (pred > 0.5).astype(np.uint8)
        else:
            raise ValueError("Model output has more than 1 channel — this is not binary segmentation.")

    # Convert image back to numpy for visualization
    img = image.permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # Ground truth mask: remove channel dimension
    mask_np = mask.squeeze(0).cpu().numpy()

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()




# Cell 5: Universal Training Function for Binary Segmentation



def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device="cuda",
    save_path="best_model.pth"
):
    best_val_loss = float("inf")

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for images, masks in train_loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("→ Saved Best Model")

    print("Training Complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")




def save_predicted_masks(model, dataset, save_dir="pred_masks", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, _ = dataset[idx]
            img_tensor = image.unsqueeze(0).to(device)
            output = model(img_tensor)
            pred_mask = torch.sigmoid(output)[0,0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            
            filename = os.path.join(save_dir, f"mask_{idx}.png")
            cv2.imwrite(filename, pred_mask)




	class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        
        bn = self.bottleneck(p4)
        
        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)
        
        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)
        
        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)
        
        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)
        
        out = self.final(c1)
        return out  # Note: raw logits, use BCEWithLogitsLoss




# Cell 6 — Initialize UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=3, out_channels=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30



train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=30,
    device=device,
    save_path="unet_binary_fruit_best.pth"
)




# Load best UNet model
model.load_state_dict(torch.load("unet_binary_fruit_best.pth"))
model.to(device)
model.eval()

# Save predicted masks for validation dataset
save_predicted_masks(
    model=model,
    dataset=val_dataset,
    save_dir="unet_pred_masks",  # you can name folder per model
    device=device
)

# Visualize one sample
visualize_prediction(model, val_dataset, index=3, device=device)

# OR visualize multiple samples manually:
for idx in [2, 10, 15, 30]:
    visualize_prediction(model, val_dataset, index=idx, device=device)





device = "cuda" if torch.cuda.is_available() else "cpu"

# Base DeepLabV3+ model
deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=False,       # Set True if you want ImageNet pretrained weights
    progress=True,
    num_classes=1           # Binary segmentation
)

# Replace classifier with appropriate output channels
deeplab_model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 1)

# Wrap DeepLabV3+ so it returns tensor directly
class DeepLabWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']  # Only return the output tensor

model = DeepLabWrapper(deeplab_model).to(device)






criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 30



	train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    save_path="deeplabv3_binary_fruit_best.pth"
)



# Load best DeepLabV3+ model
model.load_state_dict(torch.load("deeplabv3_binary_fruit_best.pth"))
model.to(device)
model.eval()

# Save predicted masks for validation dataset
save_predicted_masks(
    model=model,
    dataset=val_dataset,
    save_dir="deeplabv3_pred_masks",
    device=device
)

# Visualize one sample
visualize_prediction(model, val_dataset, index=3, device=device)

# Or visualize multiple samples manually
for idx in [2, 10, 15, 30]:
    visualize_prediction(model, val_dataset, index=idx, device=device)




import torch
import torch.nn as nn

def select_and_save_best_model(
    unet_path,
    deeplab_path,
    val_loader,
    device="cuda",
    save_path="partD_FINAL_best_model.pth"
):
    criterion = nn.BCEWithLogitsLoss()

    # --------------------
    # Load UNet
    # --------------------
    unet = UNet(in_channels=3, out_channels=1).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()

    unet_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = unet(images)
            loss = criterion(outputs, masks)
            unet_loss += loss.item()
    unet_loss /= len(val_loader)

    # --------------------
    # Load DeepLabV3+
    # --------------------
    deeplab_model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False,
        num_classes=1
    )
    deeplab_model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 1)

    class DeepLabWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)['out']

    deeplab = DeepLabWrapper(deeplab_model).to(device)
    deeplab.load_state_dict(torch.load(deeplab_path, map_location=device))
    deeplab.eval()

    deeplab_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = deeplab(images)
            loss = criterion(outputs, masks)
            deeplab_loss += loss.item()
    deeplab_loss /= len(val_loader)

    # --------------------
    # Compare & select best
    # --------------------
    if unet_loss < deeplab_loss:
        best_model = unet
        best_name = "UNet"
        best_loss = unet_loss
    else:
        best_model = deeplab
        best_name = "DeepLabV3"
        best_loss = deeplab_loss

    # --------------------
    # Save FINAL checkpoint
    # --------------------
    checkpoint = {
        "model_name": best_name,
        "model_state_dict": best_model.state_dict(),
        "best_val_loss": best_loss,
        "image_size": (256, 256),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "threshold": 0.5
        
    }

    torch.save(checkpoint, save_path)

    print("✅ FINAL BEST MODEL SAVED")
    print(f"Model: {best_name}")
    print(f"Validation Loss: {best_loss:.4f}")




select_and_save_best_model(
    unet_path="unet_binary_fruit_best.pth",
    deeplab_path="deeplabv3_binary_fruit_best.pth",
    val_loader=val_loader,
    device=device
)






