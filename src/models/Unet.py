import torch #biblioteca clave para deep learning
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp  
import matplotlib.pyplot as plt 
import random 

# Tamaño de imagenes de entrada de 256x256 al igual que el dataset
# Con el batch se procesan las imagenes en lotes de 4
# Todas las imagenes se procesan 15 veces con epocas y se establece tasa de aprendizaje
IMG_SIZE = 256 
BATCH_SIZE = 4 
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
# Se guardan los pesos tras el entrenamiento en la carpeta models
PATH = os.path.join(PROJECT_ROOT, "models", 'pesos.pth') 
# Se cargan los ficheros de imagenes de frescos y sus mascaras
X_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dataset", "X")
Y_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "dataset", "Y")


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
    # para evitar memorizacion se aumentan los datos pudiendo rotarse horizontal o verticalmente con un 50%
    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert("RGB")
        mask = Image.open(self.y_paths[idx]).convert("L")

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
        # Convertir a Tensor para pytorch y Normalizar  para imagenet
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        mask = TF.to_tensor(mask) # (0.0 a 1.0)

        return img, mask
    

# nos aseguramos que los datos esten ordenados por su nombre para que puedan ser emparejados por orden
image_files = sorted(os.listdir(X_DIR))
mask_files = sorted(os.listdir(Y_DIR))

# Filtrar solo archivos válidas por si hubiese algo que no fuese una imagen por error
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [f for f in mask_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

all_x_paths = [os.path.join(X_DIR, f) for f in image_files]
all_y_paths = [os.path.join(Y_DIR, f) for f in mask_files]

if len(all_x_paths) == 0:
    raise ValueError("No se encontraron imágenes")

# se dividen los datos en 80% para entrenar y 20% validar
train_x, val_x, train_y, val_y = train_test_split(
    all_x_paths, all_y_paths, test_size=0.2, random_state=42
)

#instancio la clase para leer las imagenes
#false en augment de validation ya que no es necesario aumentar datos pues no aprende en esa etapa
train_dataset = MaskDataset(train_x, train_y, augment=True, img_size=IMG_SIZE)
val_dataset = MaskDataset(val_x, val_y, augment=False, img_size=IMG_SIZE)

# preparamos el conjutno de entrenamiento por el bacth size, se barajan las imagenes shuffle en el entrenamiento
# para evitar que se aprenda el orden y se descarta el ultimo lote si esta incompleto p ej. 1333 / 4, se descarta
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Configuración del modelo U-Net

# verificamos si disponemos de tarjeta nvidia para acelerar el proceso, sino se usa el procesador
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Definición del modelo usando la librería
model = smp.Unet(
    encoder_name="resnet18",        # Encoder ligero y potente
    encoder_weights="imagenet",     # Pesos pre-entrenados (Transfer Learning)
    in_channels=3,                  # RGB
    classes=1,                      # Máscara binaria
    activation=None                 # Usamos BCEWithLogitsLoss, así que no activamos aquí
).to(device)

# Se define una Loss combinada (Dice para forma + BCE para pixel)
loss_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
loss_bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1) 
# Función para calcular la pérdida
def criterion(pred, target):
    return 0.5 * loss_dice(pred, target) + 0.5 * loss_bce(pred, target)

# se usa el optimizador Adam
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Scheduler que reduce el LR si la loss no mejora en 2 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Inicializamos el historial antes del bucle de entrenamiento
history = {
    'train_loss': [],
    'val_loss': [],
    'val_iou': [],
    'val_dice': []
}

# para cada epoca del entrenamiento
for epoch in range(NUM_EPOCHS):
    
    # FASE DE ENTRENAMIENTO
    model.train() # activa el modo entrenamiento 
    running_loss = 0.0
    
    # Barra de carga para el entrenamiento
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    
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
    best_val_loss = float('inf')

    with torch.no_grad(): 
        # simplemente evaluamos el conjunto reservado de entrenamiento sin aprender
        for val_img, val_mask in test_loader:
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
    epoch_val_loss = val_running_loss / len(test_loader)
    epoch_val_dice = val_dice_acc / len(test_loader)
    epoch_val_iou = val_iou_acc / len(test_loader)

    # Guardar en el historial los resultados para las graficas
    history['val_loss'].append(epoch_val_loss)
    history['val_dice'].append(epoch_val_dice)
    history['val_iou'].append(epoch_val_iou)
    
    # Imprimir resumen de cada época por pantalla
    print(f"   Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val IoU: {epoch_val_iou:.4f}")
    
    # Actualizar el Scheduler bsado en el error de validacion, se decide si se baja el learning rate
    scheduler.step(epoch_val_loss)

# Se guarda el modelo
torch.save(model.state_dict(), PATH)

# Para facilitar la salida de los datos se muestra una gráfica con informacion sobre:
# train loss, validation loss, dice score e iou score

# Dimensiones de las graficas, una para los loss y  otra para dice e iou
plt.figure(figsize=(12, 5))

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='orange')
plt.title('Curva de Aprendizaje (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Gráfica de Métricas (IoU y Dice)
plt.subplot(1, 2, 2)
plt.plot(history['val_dice'], label='Dice Score', color='green')
plt.plot(history['val_iou'], label='IoU Score', color='red')
plt.title('Calidad de Detección (Métricas)')
plt.xlabel('Epochs')
plt.ylabel('Score (0-1)')
plt.legend()
plt.grid(True)

# Guardar la imagen de las gráficas
ruta_grafica = os.path.join(PROJECT_ROOT, "outputs",'metricas_entrenamiento.png')
plt.savefig(ruta_grafica)
plt.show()