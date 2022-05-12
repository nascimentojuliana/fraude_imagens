import os, sys
import numpy as np
import pandas as pd

import tensorflow as tf

from joblib import load

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class Pipeline():
	def __init__(self, method='RGB', mode='inception', dimension=None):
		self.method = method
		self.mode = mode
		self.dimension = dimension
		
		#self.model1 = self.load_model(self.path_model1)
		self.path_model = ''

		if self.path_model:
			self.model = load_model(self.path_model)

	def pre_process(self, IMAG_PATH):
		img = image.load_img(IMAG_PATH, target_size=(self.dimension , self.dimension))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = imagenet_utils.preprocess_input(x=x, mode='tf')
		return x

	def predict(self, image):
		predictions = self.model.predict(image)
		return predictions
