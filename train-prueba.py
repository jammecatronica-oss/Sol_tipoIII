#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:40:06 2026
-Vamos a entrenar la red neuronal -yolo26n.pt- con 995 imágenes generadas de ráfagas tipo III (la llamaremos 0) y IIIb (la llamaremos 1).
- De las 1000 imágenes disponibles (creadas sintéticamente), vamos a dedicar 100 para validar y 100 para test.
@author: jam
"""

from ultralytics import YOLO
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # Load a pretrained YOLO26n model
# model = YOLO("yolo26n.pt")

# # Train the model on COCO8
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
# # results = model("imagen-prueba.jpeg", show=True)

# %%
# ============================================================
# CONFIGURACIÓN — edita estos valores según tu caso
# ============================================================
 
# Ruta al archivo data.yaml que describe tu dataset sintético
DATA_YAML = "cook.yaml"          # <-- EDITA la ruta si está en otro lugar
 
# Punto de partida del entrenamiento:
#   - "yolo26n.pt"   -> fine-tuning desde pesos preentrenados en COCO
#                       (RECOMENDADO incluso si dominio -espectrogramas
#                       solares- es muy distinto de COCO: converge más rápido
#                       y suele generalizar mejor con datasets sintéticos
#                       limitados, porque parte de capas que ya saben extraer
#                       bordes/texturas/formas en vez de pesos aleatorios).
#   - "yolo26n.yaml" -> arquitectura sin pesos, entrenamiento 100% desde cero
#                       ("from scratch" real). Necesita muchas más
#                       imágenes sintéticas y épocas para converger a un
#                       resultado comparable.

# Modelo inicial, cuando todo empieza
MODEL_INIT = "yolo26n.pt"

# # si se interrumpió el entrenamiento, uso este path:
# MODEL_INIT = "runs/detect/runs_solar/yolo26n_synth_v1/weights/last.pt"
 
EPOCHS = 150
IMG_SIZE = 640
BATCH = 16            # baja este valor si te quedas sin memoria (GPU/CPU)
PATIENCE = 30         # early stopping: épocas sin mejora antes de detener
PROJECT = "runs_solar"
NAME = "yolo26n_synth_v1"
 
# %%
# ============================================================
# ENTRENAMIENTO
# ============================================================
 
model = YOLO(MODEL_INIT)
 
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    patience=PATIENCE,
    project=PROJECT,
    name=NAME,
    plots=True,        # guarda curvas de pérdida, PR curve, matriz de confusión, etc.
)
 
# %%
# ============================================================
# VALIDACIÓN — evalúa el mejor checkpoint sobre el set de validación
# ============================================================
 
metrics = model.val()
print("mAP50-95:", metrics.box.map)
print("mAP50:   ", metrics.box.map50)
 
# %%
# ============================================================
# El mejor modelo queda guardado en:
#   {PROJECT}/{NAME}/weights/best.pt
#
# Para usarlo luego en inferencia, igual que en tu script original:
#   model = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
#   results = model.predict(source="test.jpg", conf=0.55, save=True,
#                            save_txt=True, save_conf=True)
# ============================================================