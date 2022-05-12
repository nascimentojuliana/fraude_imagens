import joblib
import pandas as pd
from joblib import load
from odonto.models.batch.pipeline import Pipeline

pipeline = Pipeline(method='RGB', mode='inception',  dimension=299)

df = pd.read_csv('../data/extração/gmail-dados/terceira_leva.csv')
predictions = pipeline.predict(df, limiar1=0.5, limiar2=0.9)
predictions.to_csv('../data/models/predictions/model_{}_{}_{}_15_20_21.csv'.format('vgg16', 'logistic', 'RGB'))
