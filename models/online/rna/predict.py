import cv2
import os,sys
import imageio
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from odonto.models.online.rna.pipeline import Pipeline
from skimage.color import gray2rgb, rgb2gray, label2rgb 

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

limiar = 0.8
image_path = ''

pipeline = Pipeline(method='RGB', mode='inception',  dimension=299)

image = pipeline.pre_process(image_path)

predict = pipeline.predict(image)
