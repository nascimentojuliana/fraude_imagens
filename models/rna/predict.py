import pandas as pd
from odonto.models.rna.neural_network import RNA

methods = ['RGB']

mode = 'inception'

df = pd.read_csv('../data/extração/gmail-dados/teste_ja_analisadas_rede_neural.csv')

df_test = df[(df.use == 'test')].reset_index(drop=True)

for method in methods:

    path_model = '../data/models/unfreeze/model_{}_{}.h5'.format(mode, method)

    model = RNA(path_model=path_model, dimension=299)

    submission_results = model.predict(df=df, batch_size=1, method=method) 

    submission_results.to_csv('../data/models/predictions/model_{}_{}_teste_ja_analisadas_09_11_2021.csv'.format(mode, method))

    print('finalizado {}'.format(method))
