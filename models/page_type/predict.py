import pandas as pd
from odonto.models.sklearn.sklearn_model import Sklearn
from odonto.models.DataGenerator import DataGenerator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

methods = ['RGB']ssss

df = pd.read_csv('')

#explainer = lime_image.LimeImageExplainer()

for method in methods:
	
	path_model3 = ''

	model_page_type = Page_Type(path_model=path_model3, dimension=self.dimension)
	
	df_test = df[(df.use == 'test')].reset_index(drop=True)
	
	predictions = model_page_type.predict(df=df_test, method=self.method, batch_size=1, limiar=0.5)

