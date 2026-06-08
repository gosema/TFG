import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


# Las imagenes se recortarán en bloques de 256x256 al igual que en el entrenamiento de la red
PATCH_SIZE = 256  
# Los recortes se irán haciendo cada 128 px solapando
STRIDE = 128      

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Rutas a los pesos
PATH_PESOS_FASE1 = os.path.join(PROJECT_ROOT, "models", 'pesos.pth') 
PATH_PESOS_FASE2 = os.path.join(PROJECT_ROOT, "models", 'pesos_finetuning.pth') 
# Ruta a los datos
REAL_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "test", 'dataset_comparacion')
REAL_X_DIR = os.path.join(REAL_BASE_DIR, 'validacion_frescos')
REAL_Y_DIR = os.path.join(REAL_BASE_DIR, 'validacion_mascaras')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar Archivos Válidos 
def get_valid_files(x_dir, y_dir):
    x_files = sorted([f for f in os.listdir(x_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    y_files = sorted([f for f in os.listdir(y_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return [os.path.join(x_dir, f) for f in x_files], [os.path.join(y_dir, f) for f in y_files]

real_x, real_y = get_valid_files(REAL_X_DIR, REAL_Y_DIR)

# Cargar AMBOS Modelos 
model_fase1 = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None).to(device)
model_fase1.load_state_dict(torch.load(PATH_PESOS_FASE1, map_location=device))
model_fase1.eval()

model_fase2 = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, activation=None).to(device)
model_fase2.load_state_dict(torch.load(PATH_PESOS_FASE2, map_location=device))
model_fase2.eval()

# Transformaciones para los parches
transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Funcion para hacer recortes de la imagen de 256x256 avanando cada 128px
def inferencia_ventana_deslizante(modelo, img_full_np):
    h_orig, w_orig, _ = img_full_np.shape
    full_probs_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    count_map = np.zeros((h_orig, w_orig), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h_orig, STRIDE):
            for x in range(0, w_orig, STRIDE):
                x_end = min(x + PATCH_SIZE, w_orig)
                y_end = min(y + PATCH_SIZE, h_orig)
                x_start = max(0, x_end - PATCH_SIZE)
                y_start = max(0, y_end - PATCH_SIZE)
                
                patch_np = img_full_np[y_start:y_end, x_start:x_end]
                patch_pil = Image.fromarray(patch_np)
                patch_tensor = transform_patch(patch_pil).unsqueeze(0).to(device)
                
                logits = modelo(patch_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy() 
                
                full_probs_map[y_start:y_end, x_start:x_end] += probs
                count_map[y_start:y_end, x_start:x_end] += 1

    final_prob_map = full_probs_map / np.maximum(count_map, 1.0)
    return (final_prob_map > 0.5).astype(np.float32) # Devuelve tensor binarizado 0 o 1

# Funcion para calcular el valor de dice e iou
def calcular_metricas(preds_np, targets_np):
    preds_flat = preds_np.flatten()
    targets_flat = targets_np.flatten()
    intersection = np.sum(preds_flat * targets_flat)
    union = np.sum(preds_flat) + np.sum(targets_flat)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    return dice, iou


val_dice_acc_f1, val_iou_acc_f1 = 0.0, 0.0
val_dice_acc_f2, val_iou_acc_f2 = 0.0, 0.0
ejemplos_visuales = []

for i in range(len(real_x)):
    print(f"   Procesando: {os.path.basename(real_x[i])}...")
    
    # Cargar imagen y máscara entera
    img_full = Image.open(real_x[i]).convert("RGB")
    mask_full = Image.open(real_y[i]).convert("L")
    
    img_np = np.array(img_full)
    # Binarizar la Ground Truth entera (valores > 0 se vuelven 1.0)
    mask_np = (np.array(mask_full) > 0).astype(np.float32) 

    # Inferencia de Unet
    pred_f1 = inferencia_ventana_deslizante(model_fase1, img_np)
    dice_f1, iou_f1 = calcular_metricas(pred_f1, mask_np)
    val_dice_acc_f1 += dice_f1
    val_iou_acc_f1 += iou_f1

    # Inferencia de Finetuning
    pred_f2 = inferencia_ventana_deslizante(model_fase2, img_np)
    dice_f2, iou_f2 = calcular_metricas(pred_f2, mask_np)
    val_dice_acc_f2 += dice_f2
    val_iou_acc_f2 += iou_f2

    # Guardamos ejemplos visuales reducidos para la imagen final
    if i < 3:
        w_viz, h_viz = 1024, int(1024 * (img_np.shape[0] / img_np.shape[1]))
        
        img_viz = cv2.resize(img_np, (w_viz, h_viz))
        truth_viz = cv2.resize(mask_np, (w_viz, h_viz), interpolation=cv2.INTER_NEAREST)
        p1_viz = cv2.resize(pred_f1, (w_viz, h_viz), interpolation=cv2.INTER_NEAREST)
        p2_viz = cv2.resize(pred_f2, (w_viz, h_viz), interpolation=cv2.INTER_NEAREST)

        ejemplos_visuales.append({
            'img': img_viz, 'truth': truth_viz, 
            'pred_f1': p1_viz, 'dice_f1': dice_f1,
            'pred_f2': p2_viz, 'dice_f2': dice_f2
        })

# Resultados Finales
n_imgs = len(real_x)
final_dice_f1 = val_dice_acc_f1 / n_imgs
final_iou_f1 = val_iou_acc_f1 / n_imgs
final_dice_f2 = val_dice_acc_f2 / n_imgs
final_iou_f2 = val_iou_acc_f2 / n_imgs

diff_dice = (final_dice_f2 - final_dice_f1) * 100
diff_iou = (final_iou_f2 - final_iou_f1) * 100

# --- Escribir resultados .txt ---
ruta_txt = os.path.join(PROJECT_ROOT, "outputs", 'comparativa_UnetVSFinetuning.txt')
with open(ruta_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 40 + "\n")
    f.write("RESULTADOS REALES (IMÁGENES COMPLETAS)\n")
    f.write("=" * 40 + "\n\n")
    f.write("--- MODELO FASE 1 (Sintético) ---\n")
    f.write(f"   Dice Score: {final_dice_f1:.4f}\n")
    f.write(f"   IoU Score:  {final_iou_f1:.4f}\n\n")
    f.write("--- MODELO FASE 2 (Fine-Tuning) ---\n")
    f.write(f"   Dice Score: {final_dice_f2:.4f}\n")
    f.write(f"   IoU Score:  {final_iou_f2:.4f}\n\n")
    f.write("--- MEJORA OBTENIDA ---\n")
    f.write(f"   Mejora Dice: {diff_dice:+.2f}%\n")
    f.write(f"   Mejora IoU:  {diff_iou:+.2f}%\n")
    f.write("=" * 40 + "\n")

with open(ruta_txt, 'r', encoding='utf-8') as f:
    print("\n" + f.read())

# Visualización grafica comparativa
fig, axs = plt.subplots(len(ejemplos_visuales), 4, figsize=(16, 4 * len(ejemplos_visuales)))

for row, ej in enumerate(ejemplos_visuales):
    axs[row, 0].imshow(ej['img'])
    axs[row, 0].set_title("Fresco Entero")
    axs[row, 0].axis('off')

    axs[row, 1].imshow(ej['truth'], cmap='gray')
    axs[row, 1].set_title("Máscara manual")
    axs[row, 1].axis('off')

    axs[row, 2].imshow(ej['pred_f1'], cmap='gray')
    axs[row, 2].set_title(f"U-Net (Dice: {ej['dice_f1']:.2f})")
    axs[row, 2].axis('off')

    axs[row, 3].imshow(ej['pred_f2'], cmap='gray')
    axs[row, 3].set_title(f"2º Entrenamiento(Dice: {ej['dice_f2']:.2f})")
    axs[row, 3].axis('off')

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.5) 
ruta_viz = os.path.join(PROJECT_ROOT, "outputs", 'comparativa_UnetVSFinetuning.png')
plt.savefig(ruta_viz, dpi=200, bbox_inches='tight') 
print(f"✅ Imagen guardada en: {ruta_viz}")