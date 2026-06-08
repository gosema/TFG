import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Las imagenes se recortarán en bloques de 256x256 al igual que en el entrenamiento de la red
PATCH_SIZE = 256  
# Los recortes se irán haciendo cada 128 px solapando
STRIDE = 128      

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# (Ejemplo: Se actualizan las rutas para cargar los pesos del modelo sometido a Fine-Tuning 
# y leer/guardar las imágenes en carpetas exclusivas para la Fase 2)
PATH_MODELO = os.path.join(PROJECT_ROOT, "models" ,'pesos_finetuning.pth')
DIR_PRUEBAS = os.path.join(PROJECT_ROOT, "data", "test", 'pruebas_finetuning')
DIR_RESULTADOS = os.path.join(PROJECT_ROOT, "outputs", 'resultados_finetuning')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Creamos la carpeta de resultados si no existe
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Verificamos que la carpeta pruebas existe y tiene imágenes
if not os.path.exists(DIR_PRUEBAS):
    print("No hay carpeta de pruebas")
    exit()

# Se recopilan los nombres de las imagenes a procesar
extensiones_validas = ('.jpg', '.jpeg', '.png')
imagenes_a_procesar = [f for f in os.listdir(DIR_PRUEBAS) if f.lower().endswith(extensiones_validas)]

# Se construye la estructura de la red con la que se hizo el entrenamiento,
# con el mismo encoder, sin los pesos de preentreno de internet y con los mismos canales y clases 
model = smp.Unet(
    encoder_name="resnet18",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# Comprobamos que tenemos el archivo con los pesos
if os.path.exists(PATH_MODELO):
    model.load_state_dict(torch.load(PATH_MODELO, map_location=device))
    print("Los pesos han sido cargados correctamente.")
    model.eval()
else:
    print("No encuentro el modelo")
    exit()

# Medias y desviaciones estandar de Imagenet para normalizar las entradas de igual forma que en el entrenamiento
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
])

# Bucle procesando imagen una a una
for nombre_img in imagenes_a_procesar:
    ruta_img = os.path.join(DIR_PRUEBAS, nombre_img)
    nombre_base = os.path.splitext(nombre_img)[0] # Saca el nombre sin el .jpg
    
    print(f"Escaneando: {nombre_img}")
    
    # Cargar imagen original
    img_full = Image.open(ruta_img).convert("RGB")
    w_orig, h_orig = img_full.size
    img_full_np = np.array(img_full)

    # Se preparan las matrices para promediar el solapamiento:
    # full_probs_map: guarda la suma de las predicciones en cada pixel
    # count_map: numero de veces que se predice cada pixel para luego hacer la media
    full_probs_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    count_map = np.zeros((h_orig, w_orig), dtype=np.float32)

    with torch.no_grad():
        # Lógica de ventana deslizante
        for y in range(0, h_orig, STRIDE):
            for x in range(0, w_orig, STRIDE):
                
                # Se obtienen las ventanas de 256x256, comprobando que no nos salimos de los bordes de la imagen
                x_end = min(x + PATCH_SIZE, w_orig)
                y_end = min(y + PATCH_SIZE, h_orig)
                
                x_start = max(0, x_end - PATCH_SIZE)
                y_start = max(0, y_end - PATCH_SIZE)
                
                # Recortar y predecir
                patch_np = img_full_np[y_start:y_end, x_start:x_end]
                patch_pil = Image.fromarray(patch_np)
                
                # Pasamos la imagen a tensor y añadimos un bacth de 1 con unsqueeze
                patch_tensor = transform_patch(patch_pil).unsqueeze(0).to(device)
                logits = model(patch_tensor)
                
                # Salida de probabilidad entre 0.0 y 1.0 tras aplicar la sigmoide
                probs = torch.sigmoid(logits).squeeze().cpu().numpy() 
                
                # Sumar al mapa de probabilidades total
                full_probs_map[y_start:y_end, x_start:x_end] += probs
                count_map[y_start:y_end, x_start:x_end] += 1

    # Se calculan las medias de las probabilidades, todo pixel con mas media de 0.5 será un desperfecto (255)
    final_prob_map = full_probs_map / np.maximum(count_map, 1.0)
    mascara_binaria = (final_prob_map > 0.5).astype(np.uint8) * 255

    # Se guarda la mascara resultado en blanco y negro
    ruta_mask = os.path.join(DIR_RESULTADOS, f"{nombre_base}_mask.png")
    cv2.imwrite(ruta_mask, mascara_binaria)

    # Adicionalmente se crea un solapamiento para apreciar mejor la predicción hecha sobre la imagen real
    overlay = img_full_np.copy()
    roi = overlay[mascara_binaria == 255]
    
    # Mezcla: 70% original + 30% del color elegido superpuesto
    overlay[mascara_binaria == 255] = (roi * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8) 
    
    # Convertir a BGR para guardar con OpenCV sin perder los colores reales
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ruta_overlay = os.path.join(DIR_RESULTADOS, f"{nombre_base}_overlay.jpg")
    cv2.imwrite(ruta_overlay, overlay_bgr)

print("\n Predicción completada")