import cv2
import numpy as np
import os
import random
import math
import noise
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_pinturas = os.path.join(BASE_DIR, "pinturas")
ruta_texturas = os.path.join(BASE_DIR, "texturas")

ruta_salida_X = os.path.join(BASE_DIR, "data_set_generado", "X_dataset_ULTIMATE")
ruta_salida_Y = os.path.join(BASE_DIR, "data_set_generado", "Y_dataset_ULTIMATE")

TARGET_SIZE = (512, 512)


def generar_simplex(shape, seed, umbral):
    alto, ancho = shape
    mascara = np.zeros(shape, dtype=np.float32)
    scale = 200.0 
    scale_inv = 1.0 / scale
    for y in range(alto):
        for x in range(ancho):
            v = noise.snoise2(x * scale_inv, y * scale_inv, octaves=6, persistence=0.6, lacunarity=2.0, base=seed)
            if v > umbral: mascara[y, x] = 1.0
    return mascara

def generar_voronoi(shape, num_puntos, grosor):
    alto, ancho = shape
    x_coords = np.random.randint(0, ancho, num_puntos)
    y_coords = np.random.randint(0, alto, num_puntos)
    puntos = np.column_stack((x_coords, y_coords))
    
    tree = cKDTree(puntos)
    y_grid, x_grid = np.mgrid[0:alto, 0:ancho]
    puntos_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
    
    _, regiones = tree.query(puntos_grid, k=1)
    regiones = regiones.reshape(alto, ancho).astype(np.float32)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bordes = cv2.morphologyEx(regiones, cv2.MORPH_GRADIENT, kernel)
    
    mascara = np.zeros_like(bordes)
    mascara[bordes > 0] = 1.0
    if grosor > 1:
        mascara = cv2.dilate(mascara, np.ones((grosor, grosor), np.uint8), iterations=1)
    return mascara

def generar_lsystem_v3(shape, num_focos=1):
    alto, ancho = shape
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    iteraciones = 150
    
    for _ in range(num_focos):
        x_seed = random.randint(ancho//4, 3*ancho//4)
        y_seed = random.randint(alto//4, 3*alto//4)
        angulo_base = random.uniform(0, 360)
        direcciones = [angulo_base, angulo_base + 180] # Bidireccional
        
        for angulo_inicial in direcciones:
            x, y = x_seed, y_seed
            angulo = angulo_inicial
            stack = [] 
            
            for i in range(iteraciones):
                ruido_theta = random.uniform(-25, 25)
                rad = math.radians(angulo + ruido_theta)
                
                # 2. Paso Variable
                paso_actual = random.uniform(2, 6) 
                x_dest = x + paso_actual * math.cos(rad)
                y_dest = y + paso_actual * math.sin(rad)
                
                if not (0 <= x_dest < ancho and 0 <= y_dest < alto): break
                
                es_desconchon = False
                if random.random() < 0.05: 
                    es_desconchon = True
                    angulo += random.choice([-1, 1]) * random.uniform(30, 60)
                
                if es_desconchon:
                    for _ in range(random.randint(3, 6)):
                        dx = random.randint(-4, 4)
                        dy = random.randint(-4, 4)
                        cv2.circle(mascara, (int(x_dest)+dx, int(y_dest)+dy), random.randint(1, 3), 1.0, -1)
                else:
                    cv2.line(mascara, (int(x), int(y)), (int(x_dest), int(y_dest)), 1.0, 1)

                if random.random() < 0.06: # Bifurcación
                    desvio_rama = random.uniform(-40, 40)
                    stack.append((x, y, angulo + desvio_rama))
                
                if len(stack) > 0 and random.random() < 0.05: # Recuperación
                    x, y, angulo = stack.pop(0)
                else:
                    x, y = x_dest, y_dest

    # Post-proceso suave
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mascara = cv2.dilate(mascara, kernel, iterations=1)
    mascara = cv2.GaussianBlur(mascara, (3,3), 0)
    return np.clip(mascara, 0, 1.0)


def mezclar_con_sombra(ic, ir, mascara, shape):
    # Si la máscara está vacía, devolver original
    if np.sum(mascara) == 0:
        return ic
        
    mask_u8 = (mascara * 255).astype(np.uint8)
    
    # Detectar bordes para la sombra
    bordes = cv2.Canny(mask_u8, 50, 150)
    bordes = cv2.GaussianBlur(bordes, (5,5), 0)
    bordes_float = bordes.astype(np.float32) / 255.0
    
    # Desplazar la sombra
    M = np.float32([[1, 0, 2], [0, 1, 2]]) 
    sombra = cv2.warpAffine(bordes_float, M, shape)
    sombra = sombra[..., np.newaxis] 
    
    im = mascara[..., np.newaxis]
    im_inversa = 1.0 - im
    
    # La textura del agujero recibe la sombra
    ir_con_sombra = ir * (1.0 - (sombra * 0.6))
    
    resultado = (im_inversa * ic) + (im * ir_con_sombra)
    return np.clip(resultado, 0, 1.0)

os.makedirs(ruta_salida_X, exist_ok=True)
os.makedirs(ruta_salida_Y, exist_ok=True)

# Cargar texturas variadas si es posible
ruta_textura_ejemplo = os.path.join(ruta_texturas, "pared_rota.jpg")
if os.path.exists(ruta_textura_ejemplo):
    tex_original = cv2.imread(ruta_textura_ejemplo)
    tex_resized = cv2.resize(tex_original, TARGET_SIZE)
else:
    tex_resized = np.full((512, 512, 3), 128, dtype=np.uint8)

lista_pinturas = [f for f in os.listdir(ruta_pinturas) if f.endswith(('.jpg', '.png'))]

print(f"Generando Dataset ULTIMATE ({len(lista_pinturas)} imágenes bases)...")

for idx, nombre_pintura in enumerate(lista_pinturas):
    img = cv2.imread(os.path.join(ruta_pinturas, nombre_pintura))
    if img is None: continue
    img = cv2.resize(img, TARGET_SIZE)
    
    ic = img.astype(np.float32) / 255.0
    ir = tex_resized.astype(np.float32) / 255.0
    
    # Rulete de Probabilidades
    dado = random.random() 
    mascara_final = np.zeros(TARGET_SIZE, dtype=np.float32)
    etiqueta = ""

    if dado < 0.20:
        # CASO 1: SIMPLEX (Manchas)
        etiqueta = "solo_simplex"
        mascara_final = generar_simplex(TARGET_SIZE, seed=idx, umbral=random.uniform(-0.05, 0.25))
        
    elif dado < 0.40:
        # CASO 2: VORONOI (Craquelado mosaico)
        etiqueta = "solo_voronoi"
        mascara_final = generar_voronoi(TARGET_SIZE, num_puntos=random.randint(40, 120), grosor=random.randint(1,3))
        
    elif dado < 0.60:
        # CASO 3: L-SYSTEM V3 (Grietas Estructurales Realistas)
        etiqueta = "solo_lsystem_v3"
        # Usamos la nueva función v3
        mascara_final = generar_lsystem_v3(TARGET_SIZE, num_focos=random.randint(1, 3))
        
    elif dado < 0.90:
        # CASO 4: MIXTO (El caos total)
        etiqueta = "mixto"
        algo_paso = False
        # Probabilidad de mancha base
        if random.random() > 0.4:
            m = generar_simplex(TARGET_SIZE, seed=idx, umbral=random.uniform(0.1, 0.35))
            mascara_final = np.maximum(mascara_final, m)
            algo_paso = True
            
        # Probabilidad de grieta estructural (IMPORTANTE: Usamos v3)
        if random.random() > 0.3:
            m = generar_lsystem_v3(TARGET_SIZE, num_focos=random.randint(1, 2))
            mascara_final = np.maximum(mascara_final, m)
            algo_paso = True

        if not algo_paso:
            # Si fallaron los dos dados, forzamos uno (ej. Simplex)
            m = generar_simplex(TARGET_SIZE, seed=idx, umbral=0.2)
            mascara_final = np.maximum(mascara_final, m)
            
    else:
        # CASO 5: LIMPIA
        etiqueta = "limpia"
        mascara_final = np.zeros(TARGET_SIZE, dtype=np.float32)

    resultado = mezclar_con_sombra(ic, ir, mascara_final, TARGET_SIZE)
    
    # Guardar
    nombre_out = f"{idx}_{etiqueta}_{nombre_pintura}"
    
    # Imagen final (X)
    cv2.imwrite(os.path.join(ruta_salida_X, nombre_out), (resultado * 255).astype(np.uint8))
    
    # Máscara binaria limpia para entrenamiento (Y) - Threshold duro para evitar grises
    _, mask_bin = cv2.threshold((mascara_final * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(ruta_salida_Y, nombre_out), mask_bin)
    
    if idx % 10 == 0: print(f"Procesado {idx}: {etiqueta}...")

print("¡Dataset ULTIMATE completado! Grietas v3 + Sombras + Balanceo aplicados.")