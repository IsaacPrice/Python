import pandas as pd
import numpy as np
from HMAI import *

df = pd.read_csv("WineQuality.csv")

model = Sequential (
    df, 
    [
    "OneHotEncode",
    "Standarize", 
    "Dense", 
    ],
    {"Target" : 'quality', "Type" : "Classification"}
)
