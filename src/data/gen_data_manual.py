import cv2
import numpy as np
import os
import random
import math
import noise
from skimage import exposure

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
# Ruta de carpeta donde estan los frescos de calidad
ruta_pinturas = os.path.join(PROJECT_ROOT, "data", "raw", "pinturas")
# Ruta de carpeta donde estan las texturas de fondo   
ruta_texturas = os.path.join(PROJECT_ROOT, "data", "raw", "texturas")
# Ruta de carpeta con los colores para aplicar
ruta_colores = os.path.join(PROJECT_ROOT, "data", "raw", "colores")

# Configuración de carpetas de salida, X las imágenes sintéticas e Y sus máscaras
ruta_salida_X = os.path.join(PROJECT_ROOT, "data", "processed", "dataset", "X")
ruta_salida_Y = os.path.join(PROJECT_ROOT, "data", "processed", "dataset", "Y")

# Tamaño de recorte de imágenes originales en 256x256px
PATCH_SIZE = 256
# Estos recortes estan solapados un 50%, avanzando de uno en uno cada 128px
STRIDE = 128 

def aplicar_ruido_gaussiano(imagen):
    """
    Añade ruido gaussiano a la imagen para simular imperfecciones de calidad de cámara.

    Args:
        imagen (numpy.ndarray): Imagen de entrada a la que se le aplicará el ruido.

    Returns:
        numpy.ndarray: Imagen resultante con ruido aplicado, normalizada al rango uint8 (0-255).
    """
    row, col, ch = imagen.shape
    mean = 0
    sigma = random.uniform(5, 15)
    # Matriz de ruido con una media 0 y desviación estandar entre 0-15
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy = imagen.astype(np.float32) + gauss
    # Se suma esta matriz de ruido a la imagen original en entero
    return np.clip(noisy, 0, 255).astype(np.uint8)

def aplicar_desenfoque(imagen):
    """
    Aplica un ligero desenfoque gaussiano para suavizar transiciones entre defecto y pintura.

    Args:
        imagen (numpy.ndarray): Imagen original.

    Returns:
        numpy.ndarray: Imagen suavizada usando un kernel aleatorio de 3x3 o 5x5.
    """
    kernel_size = random.choice([3, 5])
    return cv2.GaussianBlur(imagen, (kernel_size, kernel_size), 0)

def aplicar_estilo_real(img_sintetica_bgr, img_referencia_bgr):
    """
    Transfiere el estilo de color y luminosidad de una imagen de referencia a la sintética.

    Utiliza el espacio de color LAB para emparejar los histogramas sin corromper la imagen.

    Args:
        img_sintetica_bgr (numpy.ndarray): Imagen generada a colorear, en formato BGR.
        img_referencia_bgr (numpy.ndarray): Imagen real de la cual extraer la paleta.

    Returns:
        numpy.ndarray: Imagen con el estilo transferido, o la imagen original si no hay referencia.
    """
    # Si no hay imagen en colores no ocurre nada
    if img_referencia_bgr is None: return img_sintetica_bgr
    # Obtención de luminosidad y canales de color de la imagen sintética y la de colores
    src_lab = cv2.cvtColor(img_sintetica_bgr, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(img_referencia_bgr, cv2.COLOR_BGR2LAB)
    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)
    # La imagen sintética adopta la distribución de (l) luces, (a) rojo-verde, (b) amarillo-azul
    matched_l = exposure.match_histograms(src_l, ref_l)
    matched_a = exposure.match_histograms(src_a, ref_a)
    matched_b = exposure.match_histograms(src_b, ref_b)
    merged_lab = cv2.merge((matched_l, matched_a, matched_b)).astype(np.uint8)
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def mezclar_simple(img_pintura, img_textura, mascara):
    """
    Fusiona la pintura original con la textura de fondo mediante interpolación matemática.

    Args:
        img_pintura (numpy.ndarray): Imagen del fresco original.
        img_textura (numpy.ndarray): Imagen de la textura de fondo (cemento/piedra).
        mascara (numpy.ndarray): Matriz bidimensional representando el daño (0 a 1).

    Returns:
        numpy.ndarray: Combinación visual de ambas capas respetando los límites de 0.0 a 1.0.
    """
    # La máscara tiene que ser de 3 canales para que coincida con las dimensiones de la pintura
    mascara_3ch = np.stack([mascara]*3, axis=-1)
    # Para interpolar se hace: Imagen final =  (textura * mascara de daño) + (pintura * (1-mascara de daño))
    parte_cemento = img_textura * mascara_3ch
    parte_piedra = img_pintura * (1.0 - mascara_3ch)
    return np.clip(parte_cemento + parte_piedra, 0, 1.0)

def generar_picaduras_realistas(shape, seed, densidad_baja=False):
    """
    Genera una máscara de desconchones superficiales (picaduras) usando ruido Simplex.

    Args:
        shape (tuple): Dimensiones de la máscara generada (alto, ancho).
        seed (int): Semilla base para generar coordenadas de ruido únicas.
        densidad_baja (bool, opcional): Regula la cantidad de picaduras mediante el umbral. 
                                        Por defecto es False.

    Returns:
        numpy.ndarray: Matriz binaria (float32) indicando zonas dañadas (1.0) y sanas (0.0).
    """
    alto, ancho = shape
    mascara = np.zeros(shape, dtype=np.float32)
    
    # Crea machas de tamaño mediano
    scale = random.uniform(25.0, 50.0) 
    scale_inv = 1.0 / scale
    
    # 3 octavas para bordes irregulares como dentados
    # Lacnarity añade pequeñas irregularidades en los bordes
    # Persistencia, la forma general domina y los detalles aportan pequeños mordiscos
    octaves = 3  
    persistence = 0.5
    lacunarity = 2.0
    # posicion aleatorio para empezar a generar la mancha en la imagen
    dx = random.randint(0, 10000)
    dy = random.randint(0, 10000)

    # Crear rejilla de coordenadas (mas optimo que bucle anidado)
    xs = (np.arange(ancho) + dx) * scale_inv
    ys = (np.arange(alto) + dy) * scale_inv
    xv, yv = np.meshgrid(xs, ys)
    # generación del mapa de daños
    mascara = np.vectorize(lambda x, y: noise.snoise2(x, y, octaves=octaves,
    persistence=persistence, lacunarity=lacunarity, base=seed))(xv, yv)

    # Normalizade -1.0 1.0 a 0.0 1.0
    mascara = (mascara - np.min(mascara)) / (np.max(mascara) - np.min(mascara))
    
    # Si la densidad es baja habrá pocas picaduras y pequeñas, pero si es alta habrá más grandes y más cntidad 
    if densidad_baja:
        umbral = random.uniform(0.80, 0.88) 
    else:
        umbral = random.uniform(0.70, 0.80) 
        
    _, mascara_bin = cv2.threshold(mascara, umbral, 1.0, cv2.THRESH_BINARY)
    mascara_bin = mascara_bin.astype(np.uint8)

    # Limpieza de la mascara eliminando pequeños puntos aleatorios cerca de las picaduras
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    mascara_bin = cv2.morphologyEx(mascara_bin, cv2.MORPH_OPEN, kernel)

    # Corrige posibles errores como pixeles blancos sueltos 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_bin, connectivity=8)
    
    # La mascara a devolver con todo 0 y escribimos en ella 1 cuando hay picadura
    mascara_final = np.zeros_like(mascara_bin)
    
    # Eliminamos las picaduras más pequeñas (mínimo 5px). 
    min_area = 5  
    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mascara_final[labels == i] = 1

    return mascara_final.astype(np.float32)

def generar_manchas_agresivas(shape, seed):
    """
    Genera una máscara de desconchones profundos y masivos usando ruido Simplex escalado.

    Args:
        shape (tuple): Dimensiones de la máscara generada (alto, ancho).
        seed (int): Semilla base para generar coordenadas de ruido únicas.

    Returns:
        numpy.ndarray: Matriz binaria (float32) con bloques sólidos de daño.
    """
    alto, ancho = shape
    mascara = np.zeros(shape, dtype=np.float32)
    
    # Escala, el daño generado ocupa entre 80 y 200px
    scale = random.uniform(80.0, 200.0) 
    scale_inv = 1.0 / scale # facilta operaciones en futuro, evita división de decimales
    octaves = 3 
    persistence = 0.5
    lacunarity = 2.0
    # Coordenadas aleatorios para las manchas
    dx = random.randint(0, 10000)
    dy = random.randint(0, 10000)

    # Crear rejilla de coordenadas (mas optimo que bucle anidado)
    xs = (np.arange(ancho) + dx) * scale_inv
    ys = (np.arange(alto) + dy) * scale_inv
    xv, yv = np.meshgrid(xs, ys)
    # generación del mapa de daños
    mascara = np.vectorize(lambda x, y: noise.snoise2(x, y, octaves=octaves,
    persistence=persistence, lacunarity=lacunarity, base=seed))(xv, yv)

    # normalizacion a 0.0 1.0
    mascara = (mascara - np.min(mascara)) / (np.max(mascara) - np.min(mascara))
    umbral = random.uniform(0.4, 0.7)
    # si la mascara vale mas que el umbral pasa a ser 1 (daño), sino es 0 
    _, mascara_bin = cv2.threshold(mascara, umbral, 1.0, cv2.THRESH_BINARY)
    
    # Erosión final con molde elíptico, así se eliminan posibles líneas
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mascara_bin = cv2.erode(mascara_bin, kernel_erosion, iterations=random.randint(1, 2))

    return mascara_bin.astype(np.float32)

def generar_lsystem_grietas(shape, num_focos=1):
    """
    Genera una máscara de grietas mediante un algoritmo ramificado tipo Sistema-L.

    Args:
        shape (tuple): Dimensiones de la máscara a generar (alto, ancho).
        num_focos (int, opcional): Cantidad de grietas principales a instanciar. 
                                   Por defecto es 1.

    Returns:
        numpy.ndarray: Matriz binaria (float32) que dibuja trayectorias irregulares de daño.
    """
    alto, ancho = shape
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    
    # Pasos que da la grieta para que sea larga o corta 
    iteraciones = random.randint(60, 150) 
    
    # para cada foco nace la grieta sobre el centro de la imagen, si fuese en el borde no crecería
    # se elige una dirección al azar
    for _ in range(num_focos):
        x = random.randint(ancho//5, 4*ancho//5)
        y = random.randint(alto//5, 4*alto//5)
        angulo = random.uniform(0, 360)
        stack = []
        grosor_base = 1 
        
        # para cada paso de la grieta gira entre 60 grados y se prolonga un poco
        for i in range(iteraciones):
            ruido_theta = random.uniform(-30, 30)
            rad = math.radians(angulo + ruido_theta)
            # Paso ajustado para 256px
            paso = random.uniform(3, 8) 
            x_dest = x + paso * math.cos(rad)
            y_dest = y + paso * math.sin(rad)
            # si llega a un borde y se corta se para el algoritmo
            if not (0 <= x_dest < ancho and 0 <= y_dest < alto): break
            # posibilidad de pequeños desconchones alrededor de la grieta, esto es tipico en los frescos
            if random.random() < 0.1:
                 # Radio de salpicadura
                 offset_x = random.randint(-4, 4)
                 offset_y = random.randint(-4, 4)
                 cv2.circle(mascara, (int(x_dest)+offset_x, int(y_dest)+offset_y), 1, 1.0, -1)

            cv2.line(mascara, (int(x), int(y)), (int(x_dest), int(y_dest)), 1.0, grosor_base)
            
            # 8% de prob de que la grieta se bifurque en otras (como raices de un arbol)
            if random.random() < 0.08:
                stack.append((x, y, angulo + random.uniform(-45, 45)))
            
            # posibilidad del 5% de que la rama inicial no siga y se vaya a la nueva creada
            if len(stack) > 0 and random.random() < 0.05:
                x, y, angulo = stack.pop(0)
            else:
                x, y = x_dest, y_dest
                angulo += random.uniform(-15, 15)

    # Grosor de la grieta de 2px
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    mascara = cv2.dilate(mascara, kernel, iterations=1)
    return np.clip(mascara, 0, 1.0)

# --- Bucle Principal de Ejecución ---

os.makedirs(ruta_salida_X, exist_ok=True)
os.makedirs(ruta_salida_Y, exist_ok=True)

# lectura de todas las imágenes de pinturas y texturas
lista_texturas = [f for f in os.listdir(ruta_texturas) if f.lower().endswith(('.jpg', '.png'))]
print(f"Se cargaron {len(lista_texturas)} texturas")
lista_pinturas = [f for f in os.listdir(ruta_pinturas) if f.lower().endswith(('.jpg', '.png'))]
print(f"Se cargaron {len(lista_pinturas)} frescos")

# lectura de todas las imágenes de colores
lista_referencias = []
if os.path.exists(ruta_colores):
    archivos_colores = [f for f in os.listdir(ruta_colores) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for archivo in archivos_colores:
        img_ref = cv2.imread(os.path.join(ruta_colores, archivo))
        if img_ref is not None:
            lista_referencias.append(img_ref)
print(f"Se cargaron {len(lista_referencias)} imágenes de la carpeta colores.")

parches_totales = 0

# Son analizadas cada pintura de fresco disponible
for idx_img, nombre_pintura in enumerate(lista_pinturas):
    img_original = cv2.imread(os.path.join(ruta_pinturas, nombre_pintura))
    if img_original is None: continue
    
    alto_img, ancho_img = img_original.shape[:2]
    
    tex_original = None
    if lista_texturas:
        tex_original = cv2.imread(os.path.join(ruta_texturas, random.choice(lista_texturas)))
    
    print(f"Procesando imagen base {idx_img} ({ancho_img}x{alto_img})...")

    # Bucle de ventana deslizante, para cada imagen se recorre de 128 en 128, haciendo recortes de 256x256
    for y in range(0, alto_img - PATCH_SIZE + 1, STRIDE):
        for x in range(0, ancho_img - PATCH_SIZE + 1, STRIDE):

            # se obtiene tambien un ventana aleatorio de fondo de 256x256
            ir = np.full((PATCH_SIZE, PATCH_SIZE, 3), (220/255.0, 220/255.0, 220/255.0), dtype=np.float32)
            if tex_original is not None:
                t_alto, t_ancho = tex_original.shape[:2]
                if t_alto >= PATCH_SIZE and t_ancho >= PATCH_SIZE:
                    ty = random.randint(0, t_alto - PATCH_SIZE)
                    tx = random.randint(0, t_ancho - PATCH_SIZE)
                    parche_tex = tex_original[ty:ty+PATCH_SIZE, tx:tx+PATCH_SIZE]
                    ir = parche_tex.astype(np.float32) / 255.0

            # generamos el tipo de defecto que presentará
            dado = random.random()
            mascara_final = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            etiquetas = []

            # 30% de prob de un desconchon, 30 grietas, 30 picaduras y un 10 de que se apliquen todos estos efectos
            if dado < 0.30:
                etiquetas.append("desconchon")
                mascara_final = generar_manchas_agresivas((PATCH_SIZE, PATCH_SIZE), seed=parches_totales)
                
            elif dado < 0.60:
                etiquetas.append("grietas")
                mascara_final = generar_lsystem_grietas((PATCH_SIZE, PATCH_SIZE), num_focos=random.randint(1, 2))
                
            elif dado < 0.90: 
                etiquetas.append("picaduras")
                mascara_final = generar_picaduras_realistas((PATCH_SIZE, PATCH_SIZE), seed=parches_totales)
                
            else:
                etiquetas.append("mixto")
                m1 = generar_manchas_agresivas((PATCH_SIZE, PATCH_SIZE), seed=parches_totales)
                m2 = generar_picaduras_realistas((PATCH_SIZE, PATCH_SIZE), seed=parches_totales+1, densidad_baja=True) 
                m3 = generar_lsystem_grietas((PATCH_SIZE, PATCH_SIZE), num_focos=random.randint(1, 2))
                # Se combinan las 3 máscaras superponiéndolas
                mascara_final = np.maximum(np.maximum(m1, m2), m3)
            
            #Las imagenes se pasan a enteros de 8 bits
                        
            # recorta la ventana de la imagen
            parche_pintura = img_original[y:y+PATCH_SIZE, x:x+PATCH_SIZE]  # uint8
            #Se aplica un color a la imagen de pintura, llamando a la funcion
            if len(lista_referencias) > 0:
                estilo_aleatorio = random.choice(lista_referencias)
                parche_pintura = aplicar_estilo_real(parche_pintura, estilo_aleatorio)
                
            ic = parche_pintura.astype(np.float32) / 255.0
            # se pasa la iamgen a 0.0 1.0 para llamar a la funcion de fusionar y crear la mascara
            resultado = mezclar_simple(ic, ir, mascara_final)
            img_final_bgr = (resultado * 255).astype(np.uint8)
            
            # se llama a las funciones para aplicar ruido y desenfoque
            img_final_bgr = aplicar_desenfoque(img_final_bgr) 
            img_final_bgr = aplicar_ruido_gaussiano(img_final_bgr)
            
            # Binarizar máscara
            mask_bin = (mascara_final * 255).astype(np.uint8)
            _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)
            
            #Se guardan los resultados, la mascara y el fresco sintetico en distintas carpetas pero con mismo nombre para emparejarlas
            etiqueta_str = "_".join(etiquetas)
            nombre_parche = f"img{idx_img}_y{y:04d}_x{x:04d}_{etiqueta_str}.png"
            cv2.imwrite(os.path.join(ruta_salida_X, nombre_parche), img_final_bgr)
            cv2.imwrite(os.path.join(ruta_salida_Y, nombre_parche), mask_bin)
            
            parches_totales += 1

print(f"Dataset generado con: {parches_totales} imágenes.")