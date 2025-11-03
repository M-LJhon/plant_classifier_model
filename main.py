from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io
from PIL import Image

# Inicializar la app
app = FastAPI(title="API Clasificación de Plantas")
# Cargar el modelo entrenado
MODEL_PATH = "plant_classifier_model.h5"
model = load_model(MODEL_PATH)

# Definir las clases.
class_names = ['cactus', 'fern', 'rose', 'sunflower', 'tulip']

# Cargar cuidados desde JSON
with open("cuidados.json", "r", encoding="utf-8") as f:
    cuidados_data = json.load(f)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir luego a tu dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_plant(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la clase de planta y sus cuidados.
    """
    try:
        # Leer imagen desde la subida
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))  # Tamaño que usó el modelo

        # Convertir a array y normalizar
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        confidence = float(np.max(pred) * 100)
        plant_name = class_names[class_index]

        # Buscar cuidados
        cuidados = cuidados_data.get(plant_name, "No hay información disponible.")

        # Respuesta
        return {
            "planta": plant_name,
            "confianza": confidence,
            "cuidados": cuidados
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
