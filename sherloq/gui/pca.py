import cv2 as cv
import numpy as np
#from odonto.sherloq.gui.tools import ToolWidget
from odonto.sherloq.gui.utility import norm_mat, modify_font, norm_img, equalize_img

class PcaWidget():
    def __init__(self, image, parent=None, equalize=True, invert=False, distance_radio=True, project_radio=False, crossprod_radio=False, component_combo=2):
        self.distance_radio = distance_radio
        self.project_radio = project_radio
        self.crossprod_radio = crossprod_radio
        self.distance_radio = distance_radio
        self.last_radio = self.distance_radio
        self.invert_check = invert
        self.equalize_check = equalize
        self.component_combo = component_combo

        image = np.array(image)

        rows, cols, chans = image.shape
        x = np.reshape(image, (rows * cols, chans)).astype(np.float32)
        mu, ev, ew = cv.PCACompute2(x, np.array([]))
        p = np.reshape(cv.PCAProject(x, mu, ev), (rows, cols, chans))
        x0 = image.astype(np.float32) - mu
        self.output = []
        for i, v in enumerate(ev):
            cross = np.cross(x0, v)
            distance = np.linalg.norm(cross, axis=2) / np.linalg.norm(v)
            project = p[:, :, i]
            self.output.extend([norm_mat(distance, to_bgr=True), norm_mat(project, to_bgr=True), norm_img(cross)])

        table_data = [
            [mu[0, 2], mu[0, 1], mu[0, 0]],
            [ev[0, 2], ev[0, 1], ev[0, 0]],
            [ev[1, 2], ev[1, 1], ev[1, 0]],
            [ev[2, 2], ev[2, 1], ev[2, 0]],
            [ew[2, 0], ew[1, 0], ew[0, 0]],
        ]

    def process(self):
        index = 3 * self.component_combo
        if self.distance_radio:
            output = self.output[index]
            self.last_radio = self.distance_radio
            if self.equalize_check:
                output = equalize_img(output)
            if self.invert_check:
                output = cv.bitwise_not(output)

        elif self.project_radio:
            output = self.output[index + 1]
            self.last_radio = self.project_radio
            if self.equalize_check:
                output = equalize_img(output)
            if self.invert_check:
                output = cv.bitwise_not(output)

        elif self.crossprod_radio:
            output = self.output[index + 2]
            self.last_radio = self.crossprod_radio
            if self.equalize_check:
                output = equalize_img(output)
            if self.invert_check:
                output = cv.bitwise_not(output)

        return output
        
        
