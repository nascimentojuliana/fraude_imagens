import cv2
import numpy as np
import tensorflow as tf
from odonto.sherloq.gui.ela import ElaWidget
from odonto.sherloq.gui.pca import PcaWidget
from odonto.sherloq.gui.frequency import FrequencyWidget
from odonto.sherloq.gui.gradient import GradientWidget
from odonto.sherloq.gui.jpeg import loss_curve
from tensorflow.keras.applications.vgg16 import preprocess_input


class PreProcessing():
    def __init__(self, dimension):
        self.dimension = dimension
        
    def transform_ela_rgb(self, image):
        ela = ElaWidget()
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        image_ela = ela.process(image_redimensionada, qm)
        result = np.concatenate((image_ela, image_redimensionada), axis=2) 
        return result

    def transform_pca_rgb(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        pca = PcaWidget(image_redimensionada)
        image_pca = pca.process()
        result = np.concatenate((image_pca, image_redimensionada), axis=2) 
        return result

    def transform_pca_ela(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        
        pca = PcaWidget(image_redimensionada)
        image_pca = pca.process()

        ela = ElaWidget()
        image_ela = ela.process(image_redimensionada)
        
        result = np.concatenate((image_pca, image_ela), axis=2) 
        return result

    def transform_pca(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        pca = PcaWidget(image_redimensionada)
        image_pca = pca.process()
        return image_pca

    def transform_ela(self, image):
        qm=75

        ela = ElaWidget()
        #image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        image_ela = ela.process(image, qm=qm)
        return image_ela

    def transform_rgb(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        image_redimensionada = np.array(image_redimensionada)
        return image_redimensionada


    def transform_gradient(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        gra = GradientWidget(image_redimensionada)
        image_gra = gra.process()
        return image_gra

    def transform_frequency(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        freq = FrequencyWidget(image_redimensionada)
        image_freq = freq.process()
        return image_freq

    def transform_pca_gra(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        gra = GradientWidget(image_redimensionada)
        image_gra = gra.process()

        pca = PcaWidget(image_redimensionada)
        image_pca = pca.process()

        result = np.concatenate((image_gra, image_pca), axis=2) 
        return result

    def transform_ela_gra(self, image):
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        gra = GradientWidget(image_redimensionada)
        image_gra = gra.process()

        ela = ElaWidget()
        image_redimensionada = tf.image.resize_with_pad(image, target_height=self.dimension, target_width=self.dimension, method=tf.image.ResizeMethod.BILINEAR,antialias=False)
        image_ela = ela.process(image_redimensionada, qm=90)

        result = np.concatenate((image_gra, image_ela), axis=2) 
        return result