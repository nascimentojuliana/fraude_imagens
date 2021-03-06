import os, sys
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from odonto.models.DataGenerator import DataGenerator
from odonto.models.DataGenerator_predict import DataGeneratorPredict

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

#base_model = ResNet50V2(weights='imagenet', include_top=False)
#base_model.save('../data/models/resnet50v2.h5')


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class RNA():
	def __init__(self,path_model_base=None,path_model=None, dimension=None):
		self.path_model_base = path_model_base
		self.path_model = path_model
		self.dimension = dimension
		    
		if self.path_model:
			self.model = self.load_model()

	def load_model(self):
		model = load_model('{}'.format(self.path_model))
		return model

	def fit(self, df_test, df_validate, df_train, method, model_base, epochs=1000, batch_size=24):

		print('###########################{}##########################'.format(method))

		# create the base pre-trained models
		base_model = Xception(weights='imagenet', include_top=False)
		#base_model.save('../data/models/inception.h5')
		#base_model = load_model(self.path_model_base)

		checkpoint_filepath = 'model.h5'

		callback = [EarlyStopping(monitor='loss', patience=20), 
					ModelCheckpoint(filepath=checkpoint_filepath, 
					monitor="val_loss", verbose=0, 
					save_best_only=True, save_weights_only=False,
					mode="auto", save_freq="epoch", options=None)]


		#csv_logger = CSVLogger('../data/models/evaluate/model_{}_unfreeze_log_{}.csv'.format(model_base, method), append=True, separator=';')

		# add a global spatial average pooling layer
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- let's say we have 200 classes
		outputs = Dense(2, activation='softmax')(x)

		# this is the model we will train
		model = Model(inputs=base_model.input, outputs=outputs)

		#base_model.trainable = True

		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional InceptionV3 layers
		#for layer in base_model.layers:
		#    layer.trainable = True

		# compile the model (should be done *after* setting layers to non-trainable)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		data_train = DataGenerator(df=df_train, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=True) 

		data_validate = DataGenerator(df=df_validate, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=True) 

		data_test = DataGenerator(df=df_test, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=True) 

		#model.fit(data_train, epochs=epochs, validation_data=(data_validate), callbacks=[callback, csv_logger])
		model.fit(data_train, epochs=epochs, validation_data=(data_validate), callbacks=[callback])

		#score = model.evaluate(data_test, batch_size=batch_size)

		#print(score)

		os.remove(checkpoint_filepath)

		return model

	def predict(self, df, batch_size, method):
		data = DataGeneratorPredict(df=df, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=False) 
		predictions = self.model.predict(data) 

		scores = pd.DataFrame(predictions, columns = ['normal', 'alterada'])
		scores_y = scores[['alterada']].values

		submission_results =self.seleciona_classe_threshold(scores)

		submission_results.insert(0, 'image', df['image_name'])

		submission_results.insert(0, 'label', df['label'])

		#submission_results.insert(0, 'texto', df['texto'])

		return submission_results

	def seleciona_classe_threshold(self, df):
		df.columns = ['normal',  'alterada']
		df['predito'] = ''
		for index, row in df.iterrows():
			normal = row.normal
			alterada = row.alterada
			if alterada >= 0.9:
				df.at[index, 'predito'] = 1
			if alterada < 0.9:
				df.at[index,'predito'] = 0
		return df


	def evaluate(self, df, batch_size, method):

		submission_results = self.predict(df, batch_size, method) 

		submission_results.insert(1, 'real', df['label'])

		submission_results['real'] = submission_results[['real']].applymap(lambda x: int(x))
		
		Y = label_binarize(submission_results['real'], classes=[0, 1])

		n_classes = Y.shape[1]

		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(Y[:, i], scores_y[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), scores_y.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		submission_results['real'] = submission_results[['real']].applymap(lambda x: str(x))
		submission_results['predito'] = submission_results[['predito']].applymap(lambda x: str(x))

		matrix = confusion_matrix(submission_results['real'], submission_results['predito'])

		accuracy = accuracy_score(submission_results['real'], submission_results['predito'], normalize=False)

		recall = recall_score(submission_results['real'], submission_results['predito'], average='macro')

		f1 = f1_score(submission_results['real'], submission_results['predito'], average='macro')

		precision = precision_score(submission_results['real'], submission_results['predito'], average='macro')

		dict_result = {'accuracy_score': accuracy,
						'recall_score': recall,
						'f1_score': f1,
						'confusion_matrix': matrix,
						'precision': precision,
						'auc': roc_auc["micro"]}

		return dict_result
