import pandas as pd
from odonto.models.page_type.page_type import Page_Type

methods = ['RGB']

mode = 'xception'

df_test = pd.read_csv('')

for method in methods:

    path_model = ''

    model = Page_Type(path_model=path_model, dimension=299)

    dict_scores = model.evaluate(df=df_test, batch_size=1, method=method, limiar=0.5) 

    with open("".format(mode, method), "w") as json_file:
        json_file.write(str(dict_scores))

    print('finalizado {}'.format(method))
