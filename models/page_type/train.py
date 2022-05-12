import pandas as pd
from odonto.models.page_type.page_type import Page_Type

methods = ['RGB'] 

model_base = 'xception'

df = pd.read_csv('../data/data/tipo_imagem/dataset_final.csv')

df_train = df[(df.use == 'TRAIN')].reset_index(drop=True)

df_validate = df[(df.use == 'VALIDATE')].reset_index(drop=True)

df_test = df[(df.use == 'TEST')].reset_index(drop=True)

for method in methods:

    path_model_base = '../data/models/{}.h5'.format(model_base)

    model = Page_Type(path_model_base=path_model_base, dimension = 299)

    model = model.fit(df_test=df_test, 
                      df_validate=df_validate, 
                      df_train=df_train,
                      method = method,
                      model_base = model_base,
                      batch_size=32,
                      epochs=4000)

    model_json = model.to_json()
    
    with open('../data/models/page_type/model_{}_{}.json'.format(model_base, method), "w") as json_file:
    	json_file.write(model_json)

    model.save('../data/models/page_type/model_{}_{}.h5'.format(model_base, method))
    print("Saved model to disk")