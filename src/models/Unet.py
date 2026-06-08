import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp  
import matplotlib.pyplot as plt 


IMG_SIZE = 512 
BATCH_SIZE = 2 
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(SCRIPT_DIR, 'UNET_resnet18_transfer.pth') #
BASE_DIR = os.path.join(SCRIPT_DIR, 'data_set_generado_texturas')
X_DIR = os.path.join(BASE_DIR, 'X')
Y_DIR = os.path.join(BASE_DIR, 'Y')


class MaskDataset(Dataset):
    def __init__(self, x_paths, y_paths, augment=False, img_size=256):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.augment = augment
        self.img_size = img_size
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

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
        img = TF.normalize(img, mean=self.mean, std=self.std) 
        
        mask = TF.to_tensor(mask) 

        return img, mask
    


import random
image_files = sorted(os.listdir(X_DIR))
mask_files = sorted(os.listdir(Y_DIR))

image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [f for f in mask_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

all_x_paths = [os.path.join(X_DIR, f) for f in image_files]
all_y_paths = [os.path.join(Y_DIR, f) for f in mask_files]

if len(all_x_paths) == 0:
    raise ValueError("¡No se encontraron imágenes! Revisa las rutas.")

train_x, val_x, train_y, val_y = train_test_split(
    all_x_paths, all_y_paths, test_size=0.2, random_state=42
)

train_dataset = MaskDataset(train_x, train_y, augment=True, img_size=IMG_SIZE)
val_dataset = MaskDataset(val_x, val_y, augment=False, img_size=IMG_SIZE)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"


model = smp.Unet(
    encoder_name="resnet18",        # Encoder ligero y potente
    encoder_weights="imagenet",     # Pesos pre-entrenados (Transfer Learning)
    in_channels=3,                  # RGB
    classes=1,                      # Máscara binaria
    activation=None                 # Usamos BCEWithLogitsLoss, así que no activamos aquí
).to(device)


loss_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
loss_bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1) 

def criterion(pred, target):
    return 0.5 * loss_dice(pred, target) + 0.5 * loss_bce(pred, target)


## --- 4. Configuración del Modelo MEJORADAaaaaaaaaaaaaaaaaaaaaaaaa ---

# Focal Loss obliga a aprender los ejemplos difíciles (manchas pequeñas)
#loss_focal = smp.losses.FocalLoss(mode='binary', alpha=0.7, gamma=2.0) 
# Dice Loss ayuda con la forma general
#loss_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

#def criterion(pred, target):
    # Damos más peso a Focal Loss para priorizar detección sobre forma perfecta
    #return 0.7 * loss_focal(pred, target) + 0.3 * loss_dice(pred, target)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler: Reduce el LR si la loss no mejora en 2 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

history = {
    'train_loss': [],
    'val_loss': [],
    'val_iou': [],
    'val_dice': []
}



for epoch in range(NUM_EPOCHS):
    

    model.train() 
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    
    for data, targets in loop:
        data = data.to(device)
        targets = targets.float().to(device)
        
        # 
        scores = model(data)
        loss = criterion(scores, targets)
        
        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item()) # Actualiza la barra con la loss actual
    
    epoch_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)


 
    model.eval() # Modo evaluación (congela pesos)
    
    val_running_loss = 0.0
    val_dice_acc = 0.0
    val_iou_acc = 0.0
    
    with torch.no_grad(): 
        for val_img, val_mask in test_loader:
            val_img = val_img.to(device)
            val_mask = val_mask.float().to(device)
            
            val_out = model(val_img)
            
            v_loss = criterion(val_out, val_mask)
            val_running_loss += v_loss.item()
            
            probs = torch.sigmoid(val_out)
            preds = (probs > 0.5).float() 
            
            preds_flat = preds.view(-1)
            targets_flat = val_mask.view(-1)
            
            intersection = (preds_flat * targets_flat).sum()
            union = preds_flat.sum() + targets_flat.sum()
            
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            
            val_dice_acc += dice.item()
            val_iou_acc += iou.item()

    epoch_val_loss = val_running_loss / len(test_loader)
    epoch_val_dice = val_dice_acc / len(test_loader)
    epoch_val_iou = val_iou_acc / len(test_loader)
    
    history['val_loss'].append(epoch_val_loss)
    history['val_dice'].append(epoch_val_dice)
    history['val_iou'].append(epoch_val_iou)
    
    print(f"   Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val IoU: {epoch_val_iou:.4f}")
    
    scheduler.step(epoch_val_loss)



torch.save(model.state_dict(), PATH)
print(f"\n Modelo guardado en: {PATH}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='orange')
plt.title('Curva de Aprendizaje (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_dice'], label='Dice Score', color='green')
plt.plot(history['val_iou'], label='IoU Score', color='red')
plt.title('Calidad de Detección (Métricas)')
plt.xlabel('Epochs')
plt.ylabel('Score (0-1)')
plt.legend()
plt.grid(True)

ruta_grafica = os.path.join(SCRIPT_DIR, 'metricas_entrenamiento.png')
plt.savefig(ruta_grafica)
print(f" Gráficas guardadas en: {ruta_grafica}")
plt.show()