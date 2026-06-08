import torch #biblioteca clave para deep learning
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Tamaño de imagenes de entrada de 256x256 al igual que el dataset
# Con el batch se procesan las imagenes en lotes de 4
IMG_SIZE = 256
BATCH_SIZE = 4
# Se aumentan las épocas a 50 para adaptar al entorno real y se reduce drásticamente 
# el Learning Rate a 1e-5 para no destruir los pesos sintéticos que ya habíamos aprendido
NUM_EPOCHS = 50         
LEARNING_RATE = 1e-5     

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Nuevas rutas apuntando directamente a las carpetas del dataset de frescos y mascaras reales hechas por gimp
BASE_REAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", 'dataset_entrenamiento_finetuning') 
X_REAL_DIR = os.path.join(BASE_REAL_DIR, 'dataset_frescos')
Y_REAL_DIR = os.path.join(BASE_REAL_DIR, 'dataset_mascaras')
#  Rutas para el modelo de entrada (fase 1) y el modelo de salida (fase 2))
PATH_MODELO_BASE = os.path.join(PROJECT_ROOT, "models", 'pesos.pth')
PATH_MODELO_FINAL = os.path.join(PROJECT_ROOT, "models", 'pesos_finetuning.pth')


# clase para decir a pythorch como leer una imagen y su mascara
class MaskDataset(Dataset):
    # se guardan rutas de imagenes y mascaras, si se hará aumento de datos y tamaño
    def __init__(self, x_paths, y_paths, augment=False, img_size=256):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.augment = augment
        self.img_size = img_size
        
        # Medias y desviacion estandar necesario de ImageNet 
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    # imagenes a procesar
    def __len__(self):
        return len(self.x_paths)

    # con un id se devuelve la imagen del fresco y su mascara, asegurando que estan en rgb y grises
    # para evitar memorizacion se aumentan los datos
    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert("RGB")
        mask = Image.open(self.y_paths[idx]).convert("L")

        if self.augment:
            # Usamos ColorJitter y rotaciones extra de 90 grados para compensar 
            # que solo tenemos 262 imágenes reales en lugar de miles
            jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05)
            img = jitter(img)

            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = TF.rotate(img, angle=k * 90)
                mask = TF.rotate(mask, angle=k * 90)

        # Convertir a Tensor para pytorch y Normalizar para imagenet
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        mask = TF.to_tensor(mask) # (0.0 a 1.0)

        return img, mask

# nos aseguramos que los datos esten ordenados por su nombre para que puedan ser emparejados por orden
# Filtrar solo archivos válidas por si hubiese algo que no fuese una imagen por error
image_files = sorted([f for f in os.listdir(X_REAL_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(Y_REAL_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

all_x_paths = [os.path.join(X_REAL_DIR, f) for f in image_files]
all_y_paths = [os.path.join(Y_REAL_DIR, f) for f in mask_files]

print(f"Total de imágenes reales encontradas: {len(all_x_paths)}")

# se dividen los datos en 80% para entrenar y 20% validar
train_x, val_x, train_y, val_y = train_test_split(
    all_x_paths, all_y_paths, test_size=0.2, random_state=42
)

# instancio la clase para leer las imagenes
# false en augment de validation ya que no es necesario aumentar datos pues no aprende en esa etapa
train_dataset = MaskDataset(train_x, train_y, augment=True, img_size=IMG_SIZE)
val_dataset = MaskDataset(val_x, val_y, augment=False, img_size=IMG_SIZE) 

# preparamos el conjutno de entrenamiento por el bacth size, se barajan las imagenes shuffle en el entrenamiento
# para evitar que se aprenda el orden y se descarta el ultimo lote si esta incompleto
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# verificamos si disponemos de tarjeta nvidia para acelerar el proceso, sino se usa el procesador
device = "cuda" if torch.cuda.is_available() else "cpu"

# Definición del modelo usando la librería
model = smp.Unet(
    encoder_name="resnet18",
    # Quitamos 'imagenet' porque no necesitamos pesos de internet, 
    # queremos cargar directamente los pesos de nuestro entrenamiento sintético
    encoder_weights=None, 
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# Inyectamos el archivo .pth generado de la anterior U-Net
# para aprovechar todo el conocimiento previo antes de ver los frescos reales
if os.path.exists(PATH_MODELO_BASE):
    model.load_state_dict(torch.load(PATH_MODELO_BASE, map_location=device))
else:
    raise FileNotFoundError("No se encotraron ningunos pesos preentrenados")

# Se define una Loss combinada (Dice para forma + BCE para pixel)
loss_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
#Reducimos el smooth_factor a 0.05 para ajustarnos más a los bordes reales)
loss_bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.05)

# Función para calcular la pérdida
def criterion(pred, target):
    # Función para calcular la pérdida
    return 0.5 * loss_dice(pred, target) + 0.5 * loss_bce(pred, target)

# se usa el optimizador Adam
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Scheduler que reduce el LR si la loss no mejora en 2
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# Inicializamos el historial antes del bucle de entrenamiento
history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
best_val_loss = float('inf')
best_epoch = 0

# para cada epoca del entrenamiento
for epoch in range(NUM_EPOCHS):
    
    # FASE DE ENTRENAMIENTO
    model.train() # activa el modo entrenamiento 
    running_loss = 0.0
    
    # Barra de carga para el entrenamiento
    loop = tqdm(train_loader, desc=f"Fase 2 Fine-Tuning - Epoch [{epoch+1}/{NUM_EPOCHS}]")

    for data, targets in loop:
        # por cada lote se envia a la GPU
        data = data.to(device)
        targets = targets.float().to(device)

        # se hace una prediccón
        scores = model(data)
        # se evalua el error
        loss = criterion(scores, targets)

        # Backward, se reajustan los pesos 
        optimizer.zero_grad()
        loss.backward()
        
        # Clip de gradientes para evitar saltos bruscos que rompan el pre-entrenamiento)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item()) # Actualiza la barra con la loss actual

    # Guardar la loss promedio de entrenamiento de esta época
    epoch_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)


    # FASE DE VALIDACIÓN
    # Aquí es donde evaluamos qué tal va el modelo sin entrenar
    model.eval() # Modo evaluación
    val_running_loss = 0.0
    val_dice_acc = 0.0
    val_iou_acc = 0.0

    with torch.no_grad():
        # simplemente evaluamos el conjunto reservado de entrenamiento sin aprender
        for val_img, val_mask in val_loader:
            val_img = val_img.to(device)
            val_mask = val_mask.float().to(device)

            # Predicción
            val_out = model(val_img)
            v_loss = criterion(val_out, val_mask)
            val_running_loss += v_loss.item()

            # La funcion sigmoide lo convierte en una decisión binaria Grieta (1) o Fondo (0)
            probs = torch.sigmoid(val_out)
            preds = (probs > 0.5).float()

            # Aplanar para calcular intersección y unión fácilmente
            preds_flat = preds.view(-1)
            targets_flat = val_mask.view(-1)

            intersection = (preds_flat * targets_flat).sum()
            union = preds_flat.sum() + targets_flat.sum()

            # Fórmulas de Dice e Iou con constante (1e-6) para evitar dividir por cero
            # con estas dos formulas podemos determinar la precisión en validacion del entrenamiento
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)

            val_dice_acc += dice.item()
            val_iou_acc += iou.item()

    # Calcular promedios de validación de cada epoca para comparar
    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_dice = val_dice_acc / len(val_loader)
    epoch_val_iou = val_iou_acc / len(val_loader)

    # Guardar en el historial los resultados para las graficas
    history['val_loss'].append(epoch_val_loss)
    history['val_dice'].append(epoch_val_dice)
    history['val_iou'].append(epoch_val_iou)

    # Imprimir resumen de cada época por pantalla
    print(f"   Real Train Loss: {epoch_train_loss:.4f} | Real Val Loss: {epoch_val_loss:.4f} | Real Val IoU: {epoch_val_iou:.4f}")

    # Actualizar el Scheduler bsado en el error de validacion, se decide si se baja el learning rate
    scheduler.step(epoch_val_loss)

    # Se guarda el modelo si mejora la validación
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), PATH_MODELO_FINAL)
        print("Modelo guardado")


# Para facilitar la salida de los datos se muestra una gráfica con informacion sobre:
# train loss, validation loss, dice score e iou score

# Dimensiones de las graficas, una para los loss y otra para dice e iou
plt.figure(figsize=(12, 5))

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Real Train Loss', color='purple')
plt.plot(history['val_loss'], label='Real Val Loss', color='brown')
plt.title('Loss - Fine-Tuning de Datos Reales')
plt.legend()
plt.grid(True)

# Gráfica de Métricas (IoU y Dice)
plt.subplot(1, 2, 2)
plt.plot(history['val_dice'], label='Real Dice', color='darkgreen')
plt.plot(history['val_iou'], label='Real IoU', color='darkred')
plt.title('Métricas Finales sobre Frescos Reales')
plt.legend()
plt.grid(True)

# Guardar la imagen de las gráficas
plt.savefig(os.path.join(PROJECT_ROOT, "outputs, "'metricas_entrenamiento_finetuning.png'))
print("Gráficas guardadas")