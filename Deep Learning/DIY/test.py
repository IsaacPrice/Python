import pandas as pd
import numpy as np
from HMAI import *

df = pd.read_csv("WineQuality.csv")
df.dropna(inplace=True)

model = Sequential (df, {'Target' : 'quality', 'Type' : 'softmax'})
model.add(Dense(input_dim=32485, output_dim=32, activation='relu'))
model.add(Dense(input_dim=32, output_dim=4, activation='softmax'))

model.run(10000)
