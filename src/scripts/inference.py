import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp


# Las imagenes se recortaran e bloques de 256x256 al igual que en el entrenamiento de la red
# Lo recortes se irán haciendo cada 128 px solapando
PATCH_SIZE = 256  
STRIDE = 128      

# Se leen los pesos del entrenamiento y las imagenes de prueba de la carpeta test y se almacenan en outputs las mascaras de prediccion
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PATH_MODELO = os.path.join(PROJECT_ROOT, "models", 'pesos.pth')
DIR_PRUEBAS = os.path.join(PROJECT_ROOT, "data", "test", 'pruebas')
DIR_RESULTADOS = os.path.join(PROJECT_ROOT, "outputs", 'resultados')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Creamos la carpeta de resultados si no existe
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Verificamos que la carpeta pruebas existe y tiene imágenes
if not os.path.exists(DIR_PRUEBAS):
    print(f"No encuentro la carpeta pruebas")
    exit()

# Se recopilan los nombres de las imagenes a preocesar
extensiones_validas = ('.jpg', '.jpeg', '.png')
imagenes_a_procesar = [f for f in os.listdir(DIR_PRUEBAS) if f.lower().endswith(extensiones_validas)]

# Se lanza error si no hay imagenes
if len(imagenes_a_procesar) == 0:
    print(f"La carpeta pruebas está vacía")
    exit()

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

# Comprobamos que tenemos el archivo con los pesos
if os.path.exists(PATH_MODELO):
    model.load_state_dict(torch.load(PATH_MODELO, map_location=device))
    print("Pesos cargados correctamente.")
    model.eval()
else:
    print("No están los pesos")
    exit()

# Medias y desviaciones estandar de Imagenet
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
transform_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
])

print(f"\n Iniciando procesamiento de {len(imagenes_a_procesar)} imágenes.")

#Bucle para procesar todas las imagenes
for nombre_img in imagenes_a_procesar:
    print(f"Analizando imagen: {nombre_img} ...")
    ruta_img = os.path.join(DIR_PRUEBAS, nombre_img)
    nombre_base = os.path.splitext(nombre_img)[0] 
    # Cargar imagen
    img_full = Image.open(ruta_img).convert("RGB")
    w_orig, h_orig = img_full.size
    img_full_np = np.array(img_full)

    # Se prepara las matrices para contar:
    # full_probs_maps: guarda la suma de las predicciones
    # count_map; numero de veces que se predice cada pixel para hacer la media
    full_probs_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    count_map = np.zeros((h_orig, w_orig), dtype=np.float32)

    with torch.no_grad():
        # Lógica de ventana deslizante
        for y in range(0, h_orig, STRIDE):
            for x in range(0, w_orig, STRIDE):
                
                # se obtienen las ventanas de 256x256, comporbando que no nos salimos de la imagen
                x_end = min(x + PATCH_SIZE, w_orig)
                y_end = min(y + PATCH_SIZE, h_orig)
                x_start = max(0, x_end - PATCH_SIZE)
                y_start = max(0, y_end - PATCH_SIZE)
                patch_np = img_full_np[y_start:y_end, x_start:x_end]
                # Se realiza la prediccion
                patch_pil = Image.fromarray(patch_np)
                # Pasamos la imagen a tensor y añdimos un bacth de 1 con unsqueeze
                patch_tensor = transform_patch(patch_pil).unsqueeze(0).to(device)
                logits = model(patch_tensor)
                # Salida de probabilidad entre 0.0 y 1.0
                probs = torch.sigmoid(logits).squeeze().cpu().numpy() 
                # Sumar al mapa de probabilidades
                full_probs_map[y_start:y_end, x_start:x_end] += probs
                count_map[y_start:y_end, x_start:x_end] += 1

    #Se calculan las medias de las probabilidades, todo pixel con mas media de 0.5 será desconchon
    final_prob_map = full_probs_map / np.maximum(count_map, 1.0)
    mascara_binaria = (final_prob_map > 0.5).astype(np.uint8) * 255

    #Se guarda la mascara resultado con su mismo nombre
    ruta_mask = os.path.join(DIR_RESULTADOS, f"{nombre_base}_mask.png")
    cv2.imwrite(ruta_mask, mascara_binaria)

    # Adicionalemnte se crea un solapamiento para apreciar mejor la predicción hecha
    overlay = img_full_np.copy()
    roi = overlay[mascara_binaria == 255]
    # Color morado
    overlay[mascara_binaria == 255] = (roi * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8)
    # Convertir a BGR para guardar con OpenCV
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ruta_overlay = os.path.join(DIR_RESULTADOS, f"{nombre_base}_overlay.jpg")
    cv2.imwrite(ruta_overlay, overlay_bgr)

print("\n Resultados completos")