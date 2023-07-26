import pandas as pd

df = pd.read_csv("WineQuality.csv")
print(df.head())

y = df.quality
X = df.drop('Type', axis='columns')

print(X.head())