import pandas as pd

data = pd.read_csv("1.benign.csv")

print (data.isnull().sum())