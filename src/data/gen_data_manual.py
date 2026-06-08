import cv2
import numpy as np
import os
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_pinturas = os.path.join(BASE_DIR, "pinturas")
ruta_texturas = os.path.join(BASE_DIR, "texturas")


ruta_salida_X = os.path.join(BASE_DIR, "data_set_generado", "X_compuestas")
ruta_salida_Y = os.path.join(BASE_DIR, "data_set_generado", "Y_mascaras")


TARGET_SIZE = (512, 512) # Ancho, Alto


ruta_textura_ejemplo = os.path.join(ruta_texturas, "pared_rota.jpg")

os.makedirs(ruta_salida_X, exist_ok=True)
os.makedirs(ruta_salida_Y, exist_ok=True) # ¡NUEVO!


textura_ir_original = cv2.imread(ruta_textura_ejemplo)
if textura_ir_original is None:
    print(f"Error: No se pudo cargar la textura de {ruta_textura_ejemplo}")
    exit()


textura_ir_original_resized = cv2.resize(textura_ir_original, TARGET_SIZE)



lista_pinturas = [f for f in os.listdir(ruta_pinturas) if f.endswith(('.jpg', '.png'))]

for nombre_pintura in lista_pinturas:
    # Cargar la pintura (Ic)
    ruta_pintura_completa = os.path.join(ruta_pinturas, nombre_pintura)
    pintura_ic_original = cv2.imread(ruta_pintura_completa)
    
    if pintura_ic_original is None:
        print(f"Omitiendo {nombre_pintura}, no se pudo cargar.")
        continue
        

    pintura_ic = cv2.resize(pintura_ic_original, TARGET_SIZE)
    textura_ir = textura_ir_original_resized.copy() # Usar la textura ya redimensionada
    

    alto, ancho, _ = pintura_ic.shape
    

    ic = pintura_ic.astype(np.float32) / 255.0
    ir = textura_ir.astype(np.float32) / 255.0


    im_mascara = np.zeros((alto, ancho), dtype=np.float32) # (512, 512)
    
    num_manchas = random.randint(3, 10)
    
    for _ in range(num_manchas):
        centro_x = random.randint(0, ancho)
        centro_y = random.randint(0, alto)
        radio_interior = random.uniform(0.02, 0.08) * ancho
        radio_exterior = random.uniform(radio_interior / ancho + 0.05, 0.20) * ancho
        num_picos = random.randint(5, 15)
        vertices = []
        
        for i in range(num_picos * 2):
            radio = radio_interior if i % 2 == 0 else radio_exterior
            radio_aleatorio = radio * random.uniform(0.8, 1.2)
            angulo = i * np.pi / num_picos
            x = int(centro_x + radio_aleatorio * np.cos(angulo))
            y = int(centro_y + radio_aleatorio * np.sin(angulo))
            vertices.append([x, y])
            
        puntos_poligono = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(im_mascara, [puntos_poligono], 1.0)



    im = im_mascara[..., np.newaxis] # (512, 512) -> (512, 512, 1)
    im_inversa = 1.0 - im
    resultado_float = (im * ir) + (im_inversa * ic)
    

    resultado_uint8 = (resultado_float * 255).astype(np.uint8)
    nombre_salida = f"generado_{nombre_pintura}"
    

    ruta_salida_X_completa = os.path.join(ruta_salida_X, nombre_salida)
    cv2.imwrite(ruta_salida_X_completa, resultado_uint8)



    mascara_uint8 = (im_mascara * 255).astype(np.uint8)

    ruta_salida_Y_completa = os.path.join(ruta_salida_Y, nombre_salida)
    cv2.imwrite(ruta_salida_Y_completa, mascara_uint8)


print(f"¡Proceso completado! {len(lista_pinturas)} pares de imágenes (X/Y) generados.")