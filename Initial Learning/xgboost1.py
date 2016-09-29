import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np


df = pd.read_csv('train2.csv', header=0)
print df
print df.columns
print df.describe()
print df.head()