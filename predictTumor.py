import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('brain_tumor_detector.h5')

def predictTumor(image):
    try:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Aplicar umbralización
        thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        # Encontrar contornos
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if not cnts:
            return "No tumor detected"

        # Obtener el contorno más grande
        c = max(cnts, key=cv.contourArea)

        # Encontrar los extremos
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # Recortar la imagen según los extremos
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        # Redimensionar y normalizar la imagen
        image_resized = cv.resize(new_image, (240, 240)) / 255.0
        image_resized = image_resized.reshape((1, 240, 240, 3))

        # Realizar predicción
        res = model.predict(image_resized)

        return "Tumor Detected" if res[0][0] > 0.5 else "No Tumor"
    except Exception as e:
        return f"Error processing image: {str(e)}"

