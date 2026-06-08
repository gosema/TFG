import os
import numpy as np
from PIL import Image

def extraer_parches_fresco():
    # Crecación de rutas de archivos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    # Se obtiene las imagenes hechas con gimp de las carpaetas data/raw/
    dir_frescos = os.path.join(PROJECT_ROOT, "data", "raw", "frescos_gimp")
    dir_mascaras = os.path.join(PROJECT_ROOT, "data", "raw", "mascaras_gimp")
    # Los resultados se muestran en la carpeta data/processed
    REAL_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", 'dataset_validacion_final')
    dir_salida_frescos = os.path.join(REAL_BASE_DIR, "dataset_frescos")
    dir_salida_mascaras = os.path.join(REAL_BASE_DIR, "dataset_mascaras")

    # Comprobacion de que existen las carpetas de entrada
    if not os.path.exists(dir_frescos) or not os.path.exists(dir_mascaras):
        print(f"Faltan las carpetas de entrada.")
        return

    # Crear las carpetas de salida si no existen
    os.makedirs(dir_salida_frescos, exist_ok=True)
    os.makedirs(dir_salida_mascaras, exist_ok=True)

    # Parámetros de la ventana, 256 cque es el tamaño referncia para dataset, entrenamiento, inferencia...
    # y un stride de 128 solapando la mitad
    TAMANO_VENTANA = 256
    STRIDE = 128
    
    # Para controlar que no hay imagenes son completamente negras ya que los recortes son automaticos,
    # la imagen debe presentar un 5% a restaurar
    UMBRAL_DANO = 0.05  

    extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp"}

    #Bucle de frescos
    for nombre_archivo in os.listdir(dir_frescos):
        ruta_fresco = os.path.join(dir_frescos, nombre_archivo)
        ruta_mascara = os.path.join(dir_mascaras, nombre_archivo)

        # Ignorar si no es archivo o no es imagen válida
        if not os.path.isfile(ruta_fresco): continue
        nombre_base, extension = os.path.splitext(nombre_archivo)
        if extension.lower() not in extensiones_validas: continue

        # Comprobar que exista su máscara correspondiente
        if not os.path.exists(ruta_mascara):
            print(f"No se encontró la máscara para {nombre_archivo}")
            continue

        try:
            # Abrir imágenes
            img_fresco = Image.open(ruta_fresco).convert('RGB')
            img_mascara = Image.open(ruta_mascara).convert('L') 
            
            ancho, alto = img_fresco.size
            
            # Convertir la máscara a un array de numpy para cálculos matemáticos rápidos
            mascara_np = np.array(img_mascara)
            total_pixeles_parche = TAMANO_VENTANA * TAMANO_VENTANA
            parches_guardados = 0

            print(f"Procesando {nombre_archivo} ({ancho}x{alto})...")

            # Recorrer la imagen con la ventana deslizante de 256x256 con stride 128
            for y in range(0, alto - TAMANO_VENTANA + 1, STRIDE):
                for x in range(0, ancho - TAMANO_VENTANA + 1, STRIDE):
                    
                    # Extraemos el trozo de la matriz correspondiente a la máscara
                    parche_masc_np = mascara_np[y:y+TAMANO_VENTANA, x:x+TAMANO_VENTANA]
                    # Contamos cuántos píxeles no son negros (> 0)
                    pixeles_dano = np.sum(parche_masc_np > 0)
                    proporcion_dano = pixeles_dano / total_pixeles_parche
                    # Si supera el umbral, recortamos la imagen real y guardamos ambas
                    if proporcion_dano >= UMBRAL_DANO:
                        caja = (x, y, x + TAMANO_VENTANA, y + TAMANO_VENTANA)
                        parche_fresco = img_fresco.crop(caja)
                        parche_mascara = img_mascara.crop(caja)
                        # Nombre para identificar
                        nombre_parche = f"{nombre_base}_y{y}_x{x}.png"
                        # se almacena en carpeta data/processed su mascara y el fresco recortado
                        parche_fresco.save(os.path.join(dir_salida_frescos, nombre_parche))
                        parche_mascara.save(os.path.join(dir_salida_mascaras, nombre_parche))
                        
                        parches_guardados += 1

        except Exception as e:
            print(f"Error procesando {nombre_archivo}: {e}")

if __name__ == "__main__":
    extraer_parches_fresco()
    print("Proceso completado.")