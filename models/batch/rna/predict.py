import joblib
import pandas as pd
from joblib import load
from odonto.models.batch.pipeline import Pipeline

pipeline = Pipeline(method='RGB', mode='xception',  dimension=299)

df = pd.read_csv('')
predictions = pipeline.predict(df, limiar1=0.5, limiar2=0.9)
predictions.to_csv('')
