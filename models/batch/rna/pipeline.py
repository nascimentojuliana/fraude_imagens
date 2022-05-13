import pandas as pd
from odonto.models.page_type.page_type import Page_Type
from odonto.models.rna.neural_network import RNA


class Pipeline():
	def __init__(self, method='RGB', mode='xception', dimension=None):
		self.method = method
		self.mode = mode
		self.dimension = dimension


	def predict(self, df, limiar1, limiar2):
		
		path_model = ''

    	model = RNA(path_model=path_model, dimension=299)

    	submission_results = model.predict(df=df, batch_size=1, method=method) 
		
		return submission_results
