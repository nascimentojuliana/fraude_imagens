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

explainer = lime_image.LimeImageExplainer()

image = pipeline.pre_process(image_path)

predict = pipeline.predict(image)

if predict[0][1] >= limiar:
    predito = 1
else:
    predito=0 

if predito == 1:
	explanation = explainer.explain_instance(image[0].astype('double'), pipeline.predict, top_labels=10, hide_color=0, num_samples=1000)
	temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
	image = mark_boundaries(temp, mask)

	output = ''
	
	imageio.imwrite(output, image)
