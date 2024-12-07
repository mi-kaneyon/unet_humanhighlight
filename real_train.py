import os
import yaml
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from PIL import Image

# unet.pyからUNetクラスをインポート
from unet import UNet

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.mask_paths = []
        for img_path in self.image_paths:
            base_name = os.path.basename(img_path)
            mask_name = base_name.replace(".png", "_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {img_path}")
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 収集時にすでに512x512で保存済みならリサイズ不要
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        mask = TF.to_tensor(mask)
        # 二値化して0/1マスクにすることを推奨
        mask = (mask > 0.5).float()

        return image, mask

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(base_dir, 'config.yaml')

if not os.path.exists(config_path):
    config_template = {
        'data_dir': 'segmentation/data',
        'images_subdir': 'images',
        'masks_subdir': 'masks',
        'batch_size': 1,
        'num_epochs': 30,
        'learning_rate': 0.0001
    }
    with open(config_path, 'w') as file:
        yaml.dump(config_template, file)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

data_dir = os.path.join(base_dir, config['data_dir'])
image_dir = os.path.join(data_dir, config['images_subdir'])
mask_dir = os.path.join(data_dir, config['masks_subdir'])

batch_size = config.get('batch_size', 1)
num_epochs = config.get('num_epochs', 20)
learning_rate = config.get('learning_rate', 0.001)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}')

model_path = os.path.join(base_dir, 'models', 'unet_model.pth')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print("Training complete. Model saved at", model_path)
