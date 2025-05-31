from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("malaria_custom_model.h5")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = preprocess_image(contents)
        prediction = model.predict(img)[0][0]
        class_label = "Parasitic" if prediction > 0.5 else "Uninfected"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        return JSONResponse({
            "prediction": class_label,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})
