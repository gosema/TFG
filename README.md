# TFG
Trabajo de Fin de Grado del curso 2025-2026 del Grado de Ingeniería Informática de la UGR.
El presente Trabajo de Fin de Grado trata de dar solución al problema de los restauradores de frescos para generar máscaras de daños de obras históricas destruidas por el paso del tiempo. Para ello se trata de implementar una U-Net capaz de detectar en imágenes con frescos los distintos desconchones, grietas odesperfectos que se hayan producido y se generará una imagen binaria de máscara de daños con las zonas en mal estado en blanco y las bien conservadas en negro.
Además como no existe ningún conjunto de datos de frescos y máscaras de daños, se recurrirá a algoritmos de generación de daños sintéticos, creando así un dataset propio.

## Propiedad de imágenes

Los datos originales utilizados para el entrenamiento y fine-tuning de este modelo (imágenes de frescos reales) han sido proporcionados por profesionales de la restauración. Por motivos de propiedad intelectual, este repositorio no incluye las imágenes reales ni los pesos finales del modelo entrenado sobre ellas.

## Archivos

TFG/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── colores/ Carpeta con imágenes de recortes de frescos de colores variados (10).
│   │   ├── frescos_gimp/ Frescos reales con los que se miden las métricas de la prueba de validación. (8)
│   │   ├── mascaras_gimp/ Máscaras hechas con el software GIMP de los frescos reales, usadas para la prueba. (8)
│   │   ├── pinturas/ Carpeta con imágenes de frescos bien conservados (10).
│   │   └── texturas/ Carpeta con imágenes de texturas de fondo, por ejemplo yeso (8).
│   ├── processed/
│   │   ├── dataset/ Contiene imagenes de frescos y sus mascaras sinteticas realizadas con gen_data_manual.py
│   │   │   ├── X/ Frescos con desperfectos sintéticos generados (7100).
│   │   │   └── Y/ Máscaras correspondientes a esos frescos con desperfectos (7100).
│   │   ├── dataset_validacion_final/ Contiene imagenes de frescos reales y sus mascaras hechas con gimp para evaluar el modelo Unet.py
│   │   │   ├── dataset_frescos_evalfinal/ ventanas de 256x256 de frescos realizadas de frescos_gimp/ (865)
│   │   │   └── dataset_mascaras_evalfinal/ ventanas de 256x256 de máscaras para validación de mascaras gimp/ (865)
│   │   └── dataset_entrenamiento_finetuning/: Contiene dataset_frescos y dataset_mascaras con imágenes reales preparadas para la segunda fase de entrenamiento
│   │       ├── dataset_frescos/: ventanas de 256x256 de frescos (697)
│   │       └── dataset_mascaras/: ventanas de 256x256 de máscaras (697)
│   └── test/
│       ├── pruebas/ Ejemplos de frescos sin procesar que el modelo debe intentar restaurar (8).
│       ├── dataset_comparacion/: imágenes reservadas para comparar objetivamente el modelo base y el modelo con fine-tuning.
│       │   ├── validacion_frescos/: ventanas de 256x256 de frescos (3)
│       │   └── validacion_mascaras/: ventanas de 256x256 de máscaras (3)
│       └── pruebas_finetuning/: Ejemplos de prueba específicos que procesa el script de inferencia del modelo refinado. (1)
├── models/
│   ├── pesos.pth Archivo generado tras el entrenamiento con los pesos guardados de la red U-Net.
│   └── pesos_finetuning: Pesos actualizados tras ejecutar la segunda fase de entrenamiento (Fine-Tuning) con imágenes reales.
├── outputs/
│   ├── resultados/ Resultados de la predicción, con las correspondientes máscaras y una imagen de solapamiento para ver lo detectado. (16)
│   ├── metricas_entrenamiento.png Imagen generada con las métricas de Validation loss, Dice e IoU obtenidas durante el entrenamiento.
│   ├── resultados_evaluacion.txt Archivo de texto generado por evaluacion_modelo_real.py con los resultados numéricos finales de Dice e IoU.
│   ├── metricas_entrenamiento_finetuning.png; Gráficas de Loss, Dice e IoU de la segunda fase de entrenamiento
│   ├── resultados_finetuning/: Carpeta con los resultados y solapamientos generados por la inferencia del modelo refinado. (2)
│   ├── comparativa_UnetVSFinetuning.png: Representación gráfica y visual de la comparativa de rendimiento entre la U-Net original y la versión refinada.
│   ├── comparativa_UnetVSFinetuning.txt: Archivo generado con los resultados numéricos de comparar el desempeño de ambos modelos.
│   └── visualizacion_inferencia.png Ejemplo generado por evaluacion_modelo_real.py más detallado a nivel de imágenes individuales para ilustrar los resultados 
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



