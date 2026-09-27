import pandas as pd

data=pd.read_csv("datasets\heart.csv")
print("HEART DATA")
print(data.head(1))

data=pd.read_csv("datasets\diabetes.csv")
print("DIABETES DATA")
print(data.head(1))

data=pd.read_csv("datasets\kidney_disease.csv")
print("KIDNEY DISEASE DATA")
print(data.head(1))