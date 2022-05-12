from time import time
import cv2 as cv
import numpy as np
from odonto.sherloq.gui.utility import create_lut, norm_mat, equalize_img, elapsed_time

class GradientWidget():
    def __init__(self, image):

        image = np.array(image)

        self.intensity_spin = 90
        self.blue_combo = 0
        self.invert_check = False
        self.equalize_check = True
        self.image = image
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype('uint8')
        #_, self.gray = cv.threshold(gray,128,255,cv.THRESH_BINARY)
        self.dx, self.dy = cv.spatialGradient(self.gray)

    def process(self):
    
        intensity = int(self.intensity_spin / 100 * 127)
        invert = self.invert_check
        equalize = self.equalize_check
       
        blue_mode = self.blue_combo
        if invert:
            dx = (-self.dx).astype(np.float32)
            dy = (-self.dy).astype(np.float32)
        else:
            dx = (+self.dx).astype(np.float32)
            dy = (+self.dy).astype(np.float32)
        dx_abs = np.abs(dx)
        dy_abs = np.abs(dy)
        red = ((dx / np.max(dx_abs) * 127) + 127).astype(np.uint8)
        green = ((dy / np.max(dy_abs) * 127) + 127).astype(np.uint8)
        if blue_mode == 0:
            blue = np.zeros_like(red)
        elif blue_mode == 1:
            blue = np.full_like(red, 255)
        elif blue_mode == 2:
            blue = norm_mat(dx_abs + dy_abs)
        elif blue_mode == 3:
            blue = norm_mat(np.linalg.norm(cv.merge((red, green)), axis=2))
        else:
            blue = None
        gradient = cv.merge([blue, green, red])
        if equalize:
            gradient = equalize_img(gradient)
        elif intensity > 0:
            gradient = cv.LUT(gradient, create_lut(intensity, intensity))
        
        return gradient