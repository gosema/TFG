import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp


PATCH_SIZE = 256  
STRIDE = 128      
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_MODELO = os.path.join(SCRIPT_DIR, 'UNET_febrero.pth')
DIR_PRUEBAS = os.path.join(SCRIPT_DIR, 'pruebas_sintetico')
DIR_RESULTADOS = os.path.join(SCRIPT_DIR, 'resultados_sintetico')

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(DIR_RESULTADOS, exist_ok=True)


extensiones_validas = ('.jpg', '.jpeg', '.png')
imagenes_a_procesar = [f for f in os.listdir(DIR_PRUEBAS) if f.lower().endswith(extensiones_validas)]


model = smp.Unet(
    encoder_name="resnet18",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

if os.path.exists(PATH_MODELO):
    model.load_state_dict(torch.load(PATH_MODELO, map_location=device))
    print("Pesos cargados correctamente.")
    model.eval()
else:
    print(f"No encuentro el modelo en {PATH_MODELO}")
    exit()

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
])


for nombre_img in imagenes_a_procesar:
    ruta_img = os.path.join(DIR_PRUEBAS, nombre_img)
    nombre_base = os.path.splitext(nombre_img)[0] # Saca el nombre sin el .jpg
    
    print(f"\nEscaneando: {nombre_img}")
    
    img_full = Image.open(ruta_img).convert("RGB")
    w_orig, h_orig = img_full.size
    img_full_np = np.array(img_full)

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
                logits = model(patch_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy() 
                
                full_probs_map[y_start:y_end, x_start:x_end] += probs
                count_map[y_start:y_end, x_start:x_end] += 1

    final_prob_map = full_probs_map / np.maximum(count_map, 1.0)
    mascara_binaria = (final_prob_map > 0.5).astype(np.uint8) * 255

    ruta_mask = os.path.join(DIR_RESULTADOS, f"{nombre_base}_mask.png")
    cv2.imwrite(ruta_mask, mascara_binaria)

    overlay = img_full_np.copy()
    roi = overlay[mascara_binaria == 255]
    overlay[mascara_binaria == 255] = (roi * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8) 
    
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ruta_overlay = os.path.join(DIR_RESULTADOS, f"{nombre_base}_overlay.jpg")
    cv2.imwrite(ruta_overlay, overlay_bgr)

