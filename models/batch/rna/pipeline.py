import pandas as pd
from odonto.models.page_type.page_type import Page_Type
from odonto.models.rna.neural_network import RNA


class Pipeline():
	def __init__(self, method='RGB', mode='inception', dimension=None):
		self.method = method
		self.mode = mode
		self.dimension = dimension


	def predict(self, df, limiar1, limiar2):
		
		path_model3 = '../data/models/page_type/model_xception_RGB.h5'

		model_page_type = Page_Type(path_model=path_model3, dimension=self.dimension)
		
		predictions = model_page_type.predict(df=df, method=self.method, batch_size=24, limiar=limiar1)

		predictions = predictions[(predictions.predito_pagina == 1)].reset_index(drop=True)

		path_model = '../data/models/unfreeze/model_{}_{}.h5'.format(mode, method)

    	model = RNA(path_model=path_model, dimension=299)

    	submission_results = model.predict(df=predictions, batch_size=1, method=method) 
		
		return submission_results
