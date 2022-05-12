import pandas as pd
from odonto.models.rna.neural_network import RNA
from odonto.models.DataGenerator import DataGenerator

methods = ['RGB']

mode = 'inception'

df = pd.read_csv('../data/data/dataset.csv')

df_test = df[(df.use == 'test')].reset_index(drop=True)

for method in methods:

    path_model = '../data/models/unfreeze/model_{}_{}.h5'.format(mode, method)

    model = RNA(path_model=path_model, dimension=299)

    dict_scores = model.evaluate(df=df_test, batch_size=1, method=method) 

    with open("../data/models/evaluate/model_{}_unfreeze_{}_completed.json".format(mode, method), "w") as json_file:
        json_file.write(str(dict_scores))

    print('finalizado {}'.format(method))
