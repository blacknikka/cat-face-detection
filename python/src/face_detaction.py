import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class FaceDetection:
    SF=1.05
    N=3

    def __init__(self, haar):
        self.haar = haar

    def getFaceRectangles(self, picturePath):
        image = cv2.imread(picturePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load the cat detector Haar cascade, then detect cat faces
        # in the input image
        detector = cv2.CascadeClassifier(self.haar)
        rects = detector.detectMultiScale(gray, scaleFactor=self.SF, minNeighbors=self.N, minSize=(75, 75))

        results = []
        for (i, (x, y, w, h)) in enumerate(rects):
            results.append(image[y:y+h, x:x+w])

        return results
