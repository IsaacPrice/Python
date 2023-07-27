import pandas as pd
import numpy as np
from HMAI import *

df = pd.read_csv("Salary_Data.csv")
df.dropna(inplace=True)

# This splits the dataframe into the ratio we needs
train_df, test_df = split_dataframe(df, train_ratio=.8)

model = Sequential (train_df, test_df, {'Target' : 'Salary', 'Type' : 'softmax'})
model.add(Dense(input_dim=200, output_dim=32, activation='relu'))
model.add(Dense(input_dim=32, output_dim=4, activation='softmax'))

model.fit(10000)
print(model.evaluate())