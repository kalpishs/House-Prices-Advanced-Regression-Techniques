#a regularized linear regression model. 
#Surprisingly it does really well with very little feature engineering. 
#The key point is to to log_transform the numeric variables since most of them are skewed.


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print train.head()
