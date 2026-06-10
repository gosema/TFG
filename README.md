# CNNs en la Restauración Automática de Frescos

Este repositorio contiene el código y los recursos asociados al Trabajo de Fin de Grado (TFG) del Grado de Ingeniería Informática de la Universidad de Granada (UGR), curso 2025-2026. 
**Autor:** José María Gómez Laynez.  
**Tecnologías:** Python, PyTorch, U-Net, ResNet-18, Visión por Computador.

## Descripción del Proyecto

La conservación y restauración de frescos históricos es una tarea fundamental pero que requiere un proceso manual muy meticuloso. Tradicionalmente, los restauradores deben dibujar a mano un mapa de daños (máscaras binarias) para identificar grietas, desconchones y pérdidas de material. 
Este proyecto busca **automatizar la creación de mapas de daños mediante Inteligencia Artificial (Deep Learning)**. Para ello, se ha implementado una red neuronal convolucional (CNN) con arquitectura **U-Net** capaz de distinguir las zonas sanas del fresco (clasificadas con 0 o negro) de las zonas deterioradas (clasificadas con 1 o blanco).

<img width="3904" height="1088" alt="fresco_mascara" src="https://github.com/user-attachments/assets/d94c2d64-12aa-47b5-a5d5-cf66ef5cd611" />

## Metodología

### 1. Generación del Dataset Sintético
Debido a la inexistencia de conjuntos de datos de frescos con sus respectivas máscaras de daños, se desarrolló un algoritmo propio para generar datos sintéticos. Partiendo de texturas de fondo (como mármol o cemento) y fragmentos de frescos en buen estado, se simularon daños utilizando algoritmos procedurales:
- **Desconchones y Picaduras:** Generados mediante la función *Simplex Noise* para crear formas orgánicas realistas.
- **Grietas:** Simuladas mediante un modelo inspirado en los *L-Systems* (crecimiento iterativo con bifurcaciones).
- **Filtros:** Se aplicó desenfoque y ruido Gaussiano para evitar bordes excesivamente definidos y mejorar la robustez del modelo.
<img width="960" height="540" alt="idea_gendata" src="https://github.com/user-attachments/assets/26e0d2db-c9fd-42b8-859c-77095800284c" />

### 2. Arquitectura del Modelo
Se ha seleccionado la arquitectura **U-Net**, el estándar de la industria para segmentación semántica, debido a su capacidad para recuperar detalles finos gracias a sus *skip connections*. 
*   **Backbone:** Como codificador (encoder) se emplea **ResNet-18** con pesos pre-entrenados en ImageNet (`encoder_weights="imagenet"`), lo cual acelera la convergencia y compensa la falta de datos específicos.
*   **Pérdida (Loss):** Se utilizó una función híbrida combinando un 50% de *Dice Loss* (coherencia global) y un 50% de *BCEWithLogitsLoss* (precisión a nivel de píxel) para lidiar con el fuerte desbalanceo de clases.
*   **Optimizador:** Adam (Learning rate 1e-4)
<img width="960" height="540" alt="idea_unet" src="https://github.com/user-attachments/assets/cbeda6a1-fd0f-4f7a-bcb7-0dfe89cdb6a5" />

## Resultados y Rendimiento

<img width="499" height="465" alt="resultados_precision" src="https://github.com/user-attachments/assets/5497f581-e9df-4b74-b511-5df657068649" />
<img width="486" height="478" alt="resultados_loss" src="https://github.com/user-attachments/assets/81e5f452-b486-4e3f-bd03-ecb9dad0cbbf" />

El modelo alcanzó una gran precisión durante el entrenamiento con datos sintéticos (Dice Score > 0.96 e IoU > 0.94). 
Para frescos reales la inferencia se realiza mediante la técnica de **ventana deslizante (sliding window)** en parches de 256x256 con un *stride* de 128 píxeles. En su validación el sistema demostró ser altamente efectivo, obteniendo un **Dice Score global de 0.8394 y un IoU de 0.7406**.
<img width="943" height="263" alt="resultados_finales3" src="https://github.com/user-attachments/assets/5d84e592-186a-40ea-8526-5575eece7944" />
<img width="960" height="540" alt="resultados_finales2" src="https://github.com/user-attachments/assets/3125876b-aecf-4e58-bce6-07f5eee7f273" />
<img width="960" height="540" alt="resultados_finales1" src="https://github.com/user-attachments/assets/757d2942-3e0b-4506-960a-aec1aff3edec" />

**Experimento de Fine-Tuning:**
Se exploró una estrategia avanzada de *fine-tuning* para intentar adaptar y refinar el conocimiento del modelo a las condiciones exactas de los frescos reales. Partiendo de los pesos obtenidos con los datos sintéticos, se entrenó la red durante 50 épocas adicionales utilizando imágenes de frescos reales y máscaras creadas manualmente.
Sin embargo, tras una evaluación objetiva, **la U-Net entrenada exclusivamente con datos sintéticos superó al modelo con fine-tuning** (Dice de 0.9102 frente a 0.8823, e IoU de 0.8391 frente a 0.7941 en las métricas de prueba) . Esto se debió principalmente a la escasez de datos reales disponibles (sólo 697 recortes útiles procedentes de 7 imágenes) y a que las máscaras generadas manualmente en GIMP contenían imprecisiones humanas que introdujeron ruido en el aprendizaje. Por lo tanto, la solución final y más robusta del proyecto prescinde del *fine-tuning*.
<img width="960" height="540" alt="comparacion_precisa" src="https://github.com/user-attachments/assets/7908ccb2-ca91-4030-8cf8-27f409a2ff55" />

## Propiedad de imágenes

Los datos originales utilizados para el entrenamiento y fine-tuning de este modelo (imágenes de frescos reales) han sido proporcionados por profesionales de la restauración. Por motivos de propiedad intelectual, este repositorio no incluye las imágenes reales ni los pesos finales del modelo entrenado sobre ellas.

## Estructura del Repositorio

```text
TFG/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── colores/ [© PI] Carpeta con imágenes de recortes de frescos de colores variados (10).
│   │   ├── frescos_gimp/ [© PI] Frescos reales con los que se miden las métricas de la prueba de validación. (8)
│   │   ├── mascaras_gimp/ [© PI] Máscaras hechas con el software GIMP de los frescos reales, usadas para la prueba. (8)
│   │   ├── pinturas/ [© PI] Carpeta con imágenes de frescos bien conservados (10).
│   │   └── texturas/ [© PI] Carpeta con imágenes de texturas de fondo, por ejemplo yeso (8).
│   ├── processed/
│   │   ├── dataset/ [© PI] Contiene imagenes de frescos y sus mascaras sinteticas realizadas con gen_data_manual.py
│   │   │   ├── X/ [© PI] Frescos con desperfectos sintéticos generados (7100).
│   │   │   └── Y/ [© PI] Máscaras correspondientes a esos frescos con desperfectos (7100).
│   │   ├── dataset_validacion_final/ [© PI] Contiene imagenes de frescos reales y sus mascaras hechas con gimp para evaluar el modelo Unet.py
│   │   │   ├── dataset_frescos_evalfinal/ [© PI] ventanas de 256x256 de frescos realizadas de frescos_gimp/ (865)
│   │   │   └── dataset_mascaras_evalfinal/ [© PI] ventanas de 256x256 de máscaras para validación de mascaras gimp/ (865)
│   │   └── dataset_entrenamiento_finetuning/: [© PI] Contiene dataset_frescos y dataset_mascaras con imágenes reales preparadas para la segunda fase de entrenamiento
│   │       ├── dataset_frescos/: [© PI] ventanas de 256x256 de frescos (697)
│   │       └── dataset_mascaras/: [© PI] ventanas de 256x256 de máscaras (697)
│   └── test/
│       ├── pruebas/ [© PI] Ejemplos de frescos sin procesar que el modelo debe intentar restaurar (8).
│       ├── dataset_comparacion/: [© PI] imágenes reservadas para comparar objetivamente el modelo base y el modelo con fine-tuning.
│       │   ├── validacion_frescos/: [© PI] ventanas de 256x256 de frescos (3)
│       │   └── validacion_mascaras/: [© PI] ventanas de 256x256 de máscaras (3)
│       └── pruebas_finetuning/: [© PI] Ejemplos de prueba específicos que procesa el script de inferencia del modelo refinado. (1)
├── models/
│   ├── pesos.pth [Local] Archivo generado tras el entrenamiento con los pesos guardados de la red U-Net.
│   └── pesos_finetuning: [Local] Pesos actualizados tras ejecutar la segunda fase de entrenamiento (Fine-Tuning) con imágenes reales.
├── outputs/
│   ├── resultados/ [© PI] Resultados de la predicción, con las correspondientes máscaras y una imagen de solapamiento para ver lo detectado. (16)
│   ├── metricas_entrenamiento.png Imagen generada con las métricas de Validation loss, Dice e IoU obtenidas durante el entrenamiento.
│   ├── resultados_evaluacion.txt Archivo de texto generado por evaluacion_modelo_real.py con los resultados numéricos finales de Dice e IoU.
│   ├── metricas_entrenamiento_finetuning.png Gráficas de Loss, Dice e IoU de la segunda fase de entrenamiento
│   ├── resultados_finetuning/: [© PI] Carpeta con los resultados y solapamientos generados por la inferencia del modelo refinado. (2)
│   ├── comparativa_UnetVSFinetuning.png: [© PI] Representación gráfica y visual de la comparativa de rendimiento entre la U-Net original y la versión refinada.
│   ├── comparativa_UnetVSFinetuning.txt: Archivo generado con los resultados numéricos de comparar el desempeño de ambos modelos.
│   └── visualizacion_inferencia.png [© PI] Ejemplo generado por evaluacion_modelo_real.py más detallado a nivel de imágenes individuales para ilustrar los resultados 
└── src/
    ├── data/
    │   ├── gen_data_manual.py Script que crea las imágenes sintéticas a partir de las carpetas de colores/, pinturas/ y texturas/.
    │   └── recortar.py Realiza los recortes de 256x256 para probar los resultados. Recorta las imágenes originales de frescos_gimp/ y mascaras_gimp/.
    ├── models/
    │   ├── Unet.py : Script principal que ejecuta el bucle de entrenamiento utilizando las imágenes de la carpeta dataset/.
    │   └── fine_tuning.py: Segunda U-Net para refinar el entrenamiento. Parte de los pesos sintéticos y entrena sobre el dataset real.
    └── scripts/
        ├── evaluacion_modelo_real.py Realiza el cálculo de las métricas de Dice e IoU con los recortes de máscaras de 256x256 de frescos reales que hay en dataset_validacion_final.
        ├── inference.py Archivo que realiza la predicción cargando los pesos de pesos.pth sobre las imágenes de la carpeta pruebas/, almacenando el resultado final en la carpeta resultados/.
        ├── inference_finetuning.py: Realiza las predicciones con los pesos_finetuning.pth sobre las imágenes de pruebas_finetuning/.
        └── comparativa_UnetVSFinetuning.py: Ejecuta la evaluación en paralelo para comparar numéricamente el rendimiento entre el modelo base y el sometido a Fine-Tuning, generando los archivos de reporte.

*(Nota: Los elementos marcados con `[© PI]` no se incluyen en este repositorio público por motivos de Propiedad Intelectual de los datos originales proporcionados por los restauradores. Los elementos marcados con `[Local]` son generados por el código pero omitidos también por Propiedad Intelectual*
````
## Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/gosema/TFG.git
    cd TFG
    ```
2.  **Instalar dependencias:**
    Se recomienda el uso de un entorno virtual. Ejecuta:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecutar inferencia:**
    El código en `src/` permite ejecutar la evaluación. Carga una imagen en la carpeta de `data/`, asegúrate de tener los pesos del modelo en `models/` y los resultados se guardarán en `outputs/`.

