import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


IMG_SIZE = 256 
BATCH_SIZE = 4  
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(SCRIPT_DIR, 'UNET_segmentacion_ligera.pth')
BASE_DIR = os.path.join(SCRIPT_DIR, 'data_set_generado')
X_DIR = os.path.join(BASE_DIR, 'X_compuestas')
Y_DIR = os.path.join(BASE_DIR, 'Y_mascaras')


import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF # 

class MaskDataset(Dataset):
    def __init__(self, x_paths, y_paths, augment=False, img_size=256):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.augment = augment  
        self.img_size = img_size

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert("RGB")
        mask = Image.open(self.y_paths[idx]).convert("L")

        img = TF.resize(img, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                angle = random.randint(-30, 30)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        mask = TF.to_tensor(mask)

        return img, mask
    

transform_x = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_y = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


image_files = sorted(os.listdir(X_DIR))
mask_files = sorted(os.listdir(Y_DIR))

all_x_paths = [os.path.join(X_DIR, f) for f in image_files]
all_y_paths = [os.path.join(Y_DIR, f) for f in mask_files]

if len(all_x_paths) == 0:
    raise ValueError("¡No se encontraron imágenes! Revisa las rutas X_DIR y Y_DIR.")

train_x, val_x, train_y, val_y = train_test_split(
    all_x_paths, all_y_paths, test_size=0.2, random_state=42
)

train_dataset = MaskDataset(train_x, train_y, augment=True,img_size=IMG_SIZE)
val_dataset = MaskDataset(val_x, val_y, augment=False, img_size=IMG_SIZE)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class DoubleConv(nn.Module):
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

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()
        
        self.inc = DoubleConv(in_channels, 32)       # Original: 64 -> Ahora: 32
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))   # 128 -> 64
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))  # 256 -> 128
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) # 512 -> 256
        
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   # 1024 -> 512

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) # 128+128=256 in -> 128 out
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)  # 64+64=128 in -> 64 out
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)   # 32+32=64 in -> 32 out
        
        # Salida
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        
        u1 = self.up1(x5)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv4(u4)
        
        return self.outc(u4)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache() # Limpieza inicial
else:
    print("Usando CPU.")

model = UNET(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


if os.path.exists(PATH):
    print(f"\nCargando pesos existentes desde {PATH}...")
    try:
        model.load_state_dict(torch.load(PATH, map_location=device))
        print("Pesos cargados.")
    except:
        print("La arquitectura ha cambiado, no se pueden cargar los pesos anteriores. Se entrenará desde cero.")
else:
    print("\nIniciando entrenamiento (Mini U-Net)...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for data, targets in loop:
            data = data.to(device)
            targets = targets.float().to(device)
            
            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Loss promedio epoch {epoch+1}: {running_loss / len(train_loader):.4f}")
            
    torch.save(model.state_dict(), PATH)
    print(f"\nModelo guardado en: {PATH}")


print("\nEvaluando modelo...")
model.eval()
batch_dice_scores = []
batch_iou_scores = []
epsilon = 1e-6

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Validación"):
        images = images.to(device)
        labels = labels.float().to(device)
        
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()

        preds_flat = preds.view(preds.shape[0], -1)
        labels_flat = labels.view(labels.shape[0], -1)

        intersection = (preds_flat * labels_flat).sum(1)
        sum_preds = preds_flat.sum(1)
        sum_labels = labels_flat.sum(1)

        dice = (2. * intersection + epsilon) / (sum_preds + sum_labels + epsilon)
        iou = (intersection + epsilon) / (sum_preds + sum_labels - intersection + epsilon)

        batch_dice_scores.append(dice.mean().item())
        batch_iou_scores.append(iou.mean().item())

print(f"\nResultados Finales (256px - Mini U-Net):")
print(f"Dice Score: {np.mean(batch_dice_scores):.4f}")
print(f"IoU Score:  {np.mean(batch_iou_scores):.4f}")