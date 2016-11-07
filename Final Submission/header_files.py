#**********************************************************************************************************
"""===========================================================================================================

Team Name 	- Persistence
Team Member1 	- Megha Agarwal (201506511)
Team Member2 	- Kalpish Singhal (201505513)

=========================================================================================================="""
#******************************************Header Files Used**********************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score


from operator import itemgetter
import itertools
import xgboost as xgb
from sklearn.svm import SVC
from itertools import product, chain

#**********************************************************************************************************

