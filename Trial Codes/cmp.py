import pandas as pd
import numpy as np
import sys

train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

train["SalePrice"]=train["SalePrice"]-test["SalePrice"]

def neg(x):
	if x<0:
		x=-1*x
	return x

train["SalePrice"]=train["SalePrice"].apply(neg)


# solution["SalePrice"]=solution.sort_values(["SalePrice"], ascending=[False])
train.to_csv("sol3.csv", index = False)
print (train["SalePrice"].sum())