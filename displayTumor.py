import numpy as np
import cv2 as cv

class DisplayTumor:
    def __init__(self):
        self.curImg = None
        self.Img = None

    def readImage(self, img):
        self.Img = np.array(img)
        self.curImg = np.array(img)
        gray = cv.cvtColor(self.Img, cv.COLOR_BGR2GRAY)
        _, self.thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    def getImage(self):
        return self.curImg

    def removeNoise(self):
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(self.thresh, cv.MORPH_OPEN, kernel, iterations=2)
        self.curImg = opening

    def displayTumor(self):
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv.dilate(self.curImg, kernel, iterations=3)

        # Transformación de distancia
        dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Región desconocida
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Etiquetado de marcadores
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Aplicar watershed
        markers = cv.watershed(self.Img, markers)
        self.Img[markers == -1] = [255, 0, 0]

        self.curImg = cv.cvtColor(self.Img, cv.COLOR_RGB2BGR)


