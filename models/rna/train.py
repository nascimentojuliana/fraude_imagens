import pandas as pd
from odonto.models.rna.neural_network import RNA

class Train():


    def train(self):
        methods = ['RGB'] 

        model_base = 'xception'

        df = pd.read_csv('')
        df_train = df[(df.use == 'train')].reset_index(drop=True)

        df_validate = df[(df.use == 'validate')].reset_index(drop=True)

        df_test = df[(df.use == 'test')].reset_index(drop=True)

        for method in methods:

            path_model_base = ''.format(model_base)

            model = RNA(path_model_base=path_model_base, dimension = 299)

            model = model.fit(df_test=df_test, 
        			    	  df_validate=df_validate, 
        			    	  df_train=df_train,
        			    	  method = method,
        			    	  model_base = model_base,
        			    	  batch_size=32,
        			    	  epochs=1000)

            model_json = model.to_json()
            with open('', "w") as json_file:
            	json_file.write(model_json)

            model.save(''.format(model_base, method))
            print("Saved model to disk")
