import numpy as np
import pandas as pd

def split_dataframe(df, train_ratio=0.8, seed=None):
    if seed:
        np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(df))
    train_size = int(len(df) * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    return df.iloc[train_indices], df.iloc[test_indices]

class Dense:
    def __init__(self, input_dim, output_dim, activation, learning_rate=0.01):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)
        self.learning_rate = learning_rate
        self.activation = activation

    def forward(self, X):
        self.X = X # Input for current layer
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self.activation_forward(self.Z)
        return self.A

    def backward(self, dA):
        print("Shapes:", self.X.shape, self.X.T.shape, dZ.shape, self.weights.shape)
        dZ = self.activation_backward(dA)
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)

        # update parameters
        self.weights -= self.learning_rate * dW
        self.biases -= self.learning_rate * db
        return dX

    def activation_forward(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation == "softmax":
            e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))  # prevent overflow
            return e_Z / np.sum(e_Z, axis=-1, keepdims=True) 

    def activation_backward(self, dA):
        if self.activation == "relu":
            return dA * (self.Z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = self.sigmoid(self.Z)
            return dA * s * (1 - s)
        elif self.activation == "softmax":
            return dA

class Sequential:
    def __init__(self, train_df, test_df, output, layers=[]):
        self.train_df = train_df 
        self.test_df = test_df
        self.layers = layers
        self.target = output['Target']
        self.type = output['Type']
        print("Successfully created model")

    def OneHotEncode(self):
        Dummies = [] # OneHotEncodes the train dataframe
        for column_names in self.train_df.columns:
            if self.train_df[column_names].dtype == 'object': # The object typically means that it contains a string
                Dummies.append(column_names)
        self.train_df = pd.concat([pd.get_dummies(self.train_df[Dummies]), self.train_df.drop(Dummies, axis='columns')], axis='columns')
        Dummies = [] # OneHotEncodes the test dataframe
        for column_names in self.test_df.columns:
            if self.test_df[column_names].dtype == 'object': # The object typically means that it contains a string
                Dummies.append(column_names)
        self.test_df = pd.concat([pd.get_dummies(self.test_df[Dummies]), self.test_df.drop(Dummies, axis='columns')], axis='columns')


    def Standardize(self):
        for feature_name in self.train_df.columns: # The train data
            if self.train_df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                mean_value = self.train_df[feature_name].mean()
                std_value = self.train_df[feature_name].std()
                self.train_df[feature_name] = (self.train_df[feature_name] - mean_value) / std_value
            elif self.train_df[feature_name].dtype == 'bool':
                self.train_df[feature_name] = self.train_df[feature_name].astype(float)

        for feature_name in self.test_df.columns: # The test data
            if self.test_df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                mean_value = self.test_df[feature_name].mean()
                std_value = self.test_df[feature_name].std()
                self.test_df[feature_name] = (self.test_df[feature_name] - mean_value) / std_value
            elif self.test_df[feature_name].dtype == 'bool':
                self.test_df[feature_name] = self.test_df[feature_name].astype(float)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def compute_loss(self, A, y):
        if self.type == 'sigmoid':  # binary classification, use binary cross entropy
            epsilon = 1e-7
            y_pred = np.clip(A, epsilon, 1 - epsilon)
            loss = -1 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
            dA = - (np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))
            return loss, dA

        elif self.type == 'softmax':  # TODO: This needs fixed. dunno really whats wrong with it
            y = pd.get_dummies(y)
            m = y.shape[0]
            loss = -np.log(A[np.arange(m), y.argmax(axis=1)]).mean()
            A[np.arange(m), y.argmax(axis=1)] -= 1
            dA = A / m
            return loss, dA

        elif self.type == 'relu':  # regression problem, use mean squared error
            loss = ((y - A)**2).mean()
            dA = -2 * (y - A)
            return loss, dA

    def fit(self, epochs, X=[], y=[]):
        self.OneHotEncode()
        self.Standardize()
        y = self.train_df[self.target]
        X = self.train_df.drop(self.target, axis='columns')

        # Dynamically update input_dim of first layer to match input DataFrame
        if len(self.layers) > 0 and isinstance(self.layers[0], Dense):
            self.layers[0].input_dim = X.shape[1]

        for i in range(epochs):
            A = self.forward(X)
            loss, dA = self.compute_loss(A, y)
            print(f'Epoch {i}, loss: {loss}')
            self.backward(dA)
    
    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        y = pd.get_dummies(Y)
        y = self.test_df[self.target]
        X = self.test_df.drop(self.target, axis='columns')
        A = self.predict(X)
        return(self.compute_loss(A, y))
