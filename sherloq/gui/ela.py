import cv2 as cv
import numpy as np
from odonto.sherloq.gui.jpeg import compress_jpg

class ElaWidget():
    def __init__(self):
        pass
        
    def preprocess(self, image, qm):
        compressed = compress_jpg(image, qm)
        return compressed

    def process(self, image, qm):
        image = np.array(image)
        original = image.astype(np.float32) / 255
        compressed = None
        difference = cv.absdiff(original, self.preprocess(image, qm).astype(np.float32) / 255)
        ela = cv.convertScaleAbs(cv.sqrt(difference) * 255, None, 50 / 20)
        return ela