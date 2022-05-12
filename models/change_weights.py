from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow import keras
#from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras_preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import EarlyStopping as Stop
from tensorflow.keras.callbacks import ModelCheckpoint as Point
from tensorflow.keras.models import load_model 
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, concatenate

input_channel = 6

# Expand weights dimension to match new input channels
def multify_weights(kernel, out_channels):
  
  mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
  tiled = np.tile(mean_1d, (out_channels, 1))
  return(tiled)


# Loop through layers of both original model 
# and custom model and copy over weights 
# layer_modify refers to first convolutional layer
def weightify(model_orig, custom_model, layer_modify):
  layer_to_modify = [layer_modify]

  conf = custom_model.get_config()
  layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))]

  for layer in model_orig.layers:
    if layer.name in layer_names:
      if layer.get_weights() != []:
        target_layer = custom_model.get_layer(layer.name)

        if layer.name in layer_to_modify: 
          kernels = layer.get_weights()[0]

          test=multify_weights(kernels, input_channel - 3)

          kernels_extra_channel = np.concatenate((kernels,
                                                  test),
                                                  axis=-2)

          print(kernels_extra_channel.shape)
                                                  
          target_layer.set_weights([kernels_extra_channel])
          target_layer.trainable = False

        else:
          target_layer.set_weights(layer.get_weights())
          target_layer.trainable = False

input_size=(299, 299, 6)
inputs = Input(input_size)

# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False)

config = base_model.get_config()
    
# Change input shape to new dimensions
config["layers"][0]["config"]["batch_input_shape"] = (None, None, None, 6)
config["layers"][1]["config"]["strides"] = (1, 1)

# # Create new model with config
base_model_new = tf.keras.models.Model.from_config(config)

modify_name = config["layers"][1]["config"]["name"]

weightify(base_model, base_model_new, modify_name)
#conv2d = base_model_new(inputs)

base_model_new.save('../../data/models/xception_6_channels.h5') 
# #base_model = load_model('inception.h5')

