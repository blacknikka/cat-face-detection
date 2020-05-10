import cv2
import os
import face_detaction
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def main():
    pictureBase = pathlib.Path('../data').resolve()
    srcBase = pathlib.Path('./').resolve()

    # get path
    fileName = 'sample-picture.jpg'
    fileNameWithoutExt = os.path.splitext(os.path.basename(fileName))[0]
    fileExt = os.path.splitext(fileName)[1]

    haarPath = os.path.join(srcBase, 'haarcascade_frontalcatface.xml')
    picturePath = os.path.join(pictureBase, 'train', fileName)

    detector = face_detaction.FaceDetection(haarPath)
    rects = detector.getFaceRectangles(picturePath)

    for (i, rect) in enumerate(rects):
        fileNo = str(i) if i != 0 else ''
        outFile = os.path.join(pictureBase, 'out', fileNameWithoutExt + fileNo + fileExt)
        cv2.imwrite(outFile, rect)

if __name__ == "__main__":
    main()

