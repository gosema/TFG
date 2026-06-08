import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

#Tamaño de la imagen, igual que las imagenes que vienen de dataset_validacion_final
IMG_SIZE = 256

# rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
# direccion de pesos y datos
PATH_PESOS = os.path.join(PROJECT_ROOT, "models", 'pesos.pth') 
REAL_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", 'dataset_validacion_final')
REAL_X_DIR = os.path.join(REAL_BASE_DIR, 'dataset_frescos_evalfinal')
REAL_Y_DIR = os.path.join(REAL_BASE_DIR, 'dataset_mascaras_evalfinal')

device = "cuda" if torch.cuda.is_available() else "cpu"

# funcion para cargar las imagenes
def get_valid_files(x_dir, y_dir):
    x_files = sorted([f for f in os.listdir(x_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    y_files = sorted([f for f in os.listdir(y_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return [os.path.join(x_dir, f) for f in x_files], [os.path.join(y_dir, f) for f in y_files]

real_x, real_y = get_valid_files(REAL_X_DIR, REAL_Y_DIR)

if len(real_x) == 0:
    print("No se encontraron imágenes")
    exit()

print(f"Cargadas {len(real_x)} imágenes")

#Transformaciones para Imagenet y psar a Tensor
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
])
transform_mask = transforms.Compose([
    transforms.ToTensor()
])


# Se construye la estructura de la red con la que se hizo el entrenamiento,
# con el mismo encoder, sin los pesos de preentreno de interneet y con los mismos canlaes
# y clases 
model = smp.Unet(
    encoder_name="resnet18",        
    encoder_weights=None, 
    in_channels=3,                  
    classes=1,                      
    activation=None                 
).to(device)

#Comprobamos que estan los pesos
if os.path.exists(PATH_PESOS):
    model.load_state_dict(torch.load(PATH_PESOS, map_location=device))
    model.eval() # MODO EVALUACIÓN 
else:
    print("No se encontraron los pesos")
    exit()


# variables para errores dice e iou
val_dice_acc = 0.0
val_iou_acc = 0.0

# Variables para guardar las primeras predicciones y poder graficarlas luego
ejemplos_visuales = []

with torch.no_grad(): # Apaga el cálculo de gradientes
    # Iteramos manualmente juntando las listas de X e Y
    for i, (ruta_img, ruta_mask) in enumerate(zip(real_x, real_y)):
        
        # Cargar imagen y máscara original con PIL
        img_pil = Image.open(ruta_img).convert("RGB")
        mask_pil = Image.open(ruta_mask).convert("L")
        
        # Guardamos una copia pura en numpy para las gráficas
        img_orig_np = np.array(img_pil)
        
        # Transformar a tensores y añadir dimensión de Batch (unsqueeze)
        val_img = transform_img(img_pil).unsqueeze(0).to(device)
        val_mask = transform_mask(mask_pil).float().unsqueeze(0).to(device)
        
        # Se realiza la predicción del modelo
        val_out = model(val_img)
        probs = torch.sigmoid(val_out)
        preds = (probs > 0.5).float() 
        #Cálculamos las metricas de dice e iou
        preds_flat = preds.view(-1)
        targets_flat = val_mask.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)
        val_dice_acc += dice.item()
        val_iou_acc += iou.item()
        # Guardamos las primeras 3 imágenes para mostrarlas para ejemplificar los resultados
        if i < 3:
            ejemplos_visuales.append({
                'img': img_orig_np, 
                'truth': val_mask.cpu().squeeze().numpy(), 
                'pred': preds.cpu().squeeze().numpy(), 
                'dice': dice.item()
            })

# Calcular las medias de dice e iou
num_imagenes = len(real_x)
final_dice = val_dice_acc / num_imagenes
final_iou = val_iou_acc / num_imagenes
# mostramos resultados en archivo .txt
ruta_txt = os.path.join(PROJECT_ROOT, "outputs", 'resultados_evaluacion.txt')
with open(ruta_txt, 'w', encoding='utf-8') as f:
    f.write("-" * 30 + "\n")
    f.write("RESULTADOS FINALES EN DATOS REALES:\n")
    f.write(f"   Dice Score Global: {final_dice:.4f}\n")
    f.write(f"   IoU Score Global:  {final_iou:.4f}\n")
    f.write("-" * 30 + "\n")
print("Se ha creado el reporte")

# Ejemplo grafico de los tres primeras imagenes para compara visualmente
fig, axs = plt.subplots(len(ejemplos_visuales), 3, figsize=(10, 4 * len(ejemplos_visuales)))

for row, ej in enumerate(ejemplos_visuales):
    axs[row, 0].imshow(ej['img'])
    axs[row, 0].set_title("Fresco Real")
    axs[row, 0].axis('off')

    axs[row, 1].imshow(ej['truth'], cmap='gray')
    axs[row, 1].set_title("Ground Truth (Experto)")
    axs[row, 1].axis('off')

    axs[row, 2].imshow(ej['pred'], cmap='gray')
    axs[row, 2].set_title(f"Predicción IA (Dice: {ej['dice']:.2f})")
    axs[row, 2].axis('off')

plt.tight_layout()
ruta_viz = os.path.join(PROJECT_ROOT, "outputs", 'visualizacion_inferencia.png')
plt.savefig(ruta_viz)
print("Imagen comparativa guardada")
plt.show()