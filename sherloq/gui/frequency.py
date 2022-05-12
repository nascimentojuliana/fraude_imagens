import cv2 as cv
import numpy as np
from odonto.sherloq.gui.utility import norm_mat

class FrequencyWidget():
    def __init__(self, image, parent=None):

        self.split_spin=15
       
        self.smooth_spin=25
        
        self.thr_spin=0

        self.filter_spin=0

        self.image = np.array(image)
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        height = cv.getOptimalDFTSize(rows)
        width = cv.getOptimalDFTSize(cols)
        padded = cv.copyMakeBorder(gray, 0, height - rows, 0, width - cols, cv.BORDER_CONSTANT)
        self.dft = np.fft.fftshift(cv.dft(padded.astype(np.float32), flags=cv.DFT_COMPLEX_OUTPUT))
        self.magnitude0, self.phase0 = cv.cartToPolar(self.dft[:, :, 0], self.dft[:, :, 1])
        self.magnitude0 = cv.normalize(cv.log(self.magnitude0), None, 0, 255, cv.NORM_MINMAX)
        self.phase0 = cv.normalize(self.phase0, None, 0, 255, cv.NORM_MINMAX)
        self.magnitude = self.phase = None

    def process(self):
        rows, cols, _ = self.dft.shape
        mask = np.zeros((rows, cols), np.float32)
        half = np.sqrt(rows ** 2 + cols ** 2) / 2
        radius = int(half * self.split_spin / 100)
        mask = cv.circle(mask, (cols // 2, rows // 2), radius, 1, cv.FILLED)
        kernel = 2 * int(half * self.smooth_spin / 100) + 1
        mask = cv.GaussianBlur(mask, (kernel, kernel), 0)
        mask /= np.max(mask)
        threshold = int(self.thr_spin / 100 * 255)
        if threshold > 0:
            mask[self.magnitude0 < threshold] = 0
            zeros = (mask.size - np.count_nonzero(mask)) / mask.size * 100
        else:
            zeros = 0
        mask2 = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

        rows0, cols0, _ = self.image.shape
        low = cv.idft(np.fft.ifftshift(self.dft * mask2), flags=cv.DFT_SCALE)
        low = norm_mat(cv.magnitude(low[:, :, 0], low[:, :, 1])[:rows0, :cols0], to_bgr=True)
        
        high = cv.idft(np.fft.ifftshift(self.dft * (1 - mask2)), flags=cv.DFT_SCALE)
        high = norm_mat(cv.magnitude(high[:, :, 0], high[:, :, 1]), to_bgr=True)
        high = (np.copy(high[: self.image.shape[0], : self.image.shape[1]]))
        self.magnitude = (self.magnitude0 * mask).astype(np.uint8)
        self.phase = (self.phase0 * mask).astype(np.uint8)
        return high

    def postprocess(self):
        kernel = 2 * self.filter_spin + 1
        if kernel >= 3:
            magnitude = cv.GaussianBlur(self.magnitude, (kernel, kernel), 0)
            phase = cv.GaussianBlur(self.phase, (kernel, kernel), 0)
            # phase = cv.medianBlur(self.phase, kernel)
        else:
            magnitude = self.magnitude
            phase = self.phase
        self.mag_viewer.update_original(cv.cvtColor(magnitude, cv.COLOR_GRAY2BGR))
        self.phase_viewer.update_original(cv.cvtColor(phase, cv.COLOR_GRAY2BGR))

        