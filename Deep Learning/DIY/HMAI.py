import numpy as np
import pandas as pd

# This function will take the data and replace the values with the one hot encoded version
def one_hot_encode(df, values):
    data = pd.get_dummies()
    print(data.head())

class Sequential:
    def __init__(self, df, layers, output):
        self.df = df # This is the dataframe for the model
    
        # This will loop through the list of hidden layers and make a list of commands
        self.hidden = []
        for layer_name in layers[0:-1]:
            if layer_name == "OneHotEncode": 
                self.hidden.append(self.OneHotEncode)
            elif layer_name == "Standardize":
                self.hidden.append(self.Standardize)
            elif layer_name == "Dense":
                self.hidden.append(self.Dense)
        
        # Sets the settings for the output layer
        self.target = output['Target']
        self.type = output['Type']

        print("Successfully created model")

    # This will find all of the dummy variables and one hot encode them
    def OneHotEncode(self):
        Dummies = []
        for column_names in self.df.columns:
            if self.df[column_names].dtype == 'object': # The object typically means that it contains a string
                Dummies.append(column_names)
        
        self.df = pd.concat([pd.get_dummies(self.df[Dummies]), self.df.drop(Dummies, axis='columns')], axis='columns')

    # This will take all of the full values and set them to floats so that the mean is 0 and the standard deviation is 1
    # It will also set the booleans to the correct values
    def Standardize(self):
        result = self.df.copy()
        for feature_name in self.df.columns:
            if self.df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                mean_value = self.df[feature_name].mean()
                std_value = self.df[feature_name].std()
                result = [feature_name] = (self.df[feature_name - mean_value]) / std_value
            elif self.df[feature_name].dtype == 'bool':
                result[feature_name] = self.df[feature_name].astype(float)
        return result

    def Dense(self):
        # First we have to assign the
    
    def Run(self):
        for command in self.hidden:
            command()