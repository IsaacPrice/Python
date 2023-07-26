import numpy as np
import pandas as pd

# This class is just becasue of how complicated the Dense layer is compared to the others
class Dense:
    def __init__(self, input_dim, output_dim, activation, learning_rate=0.01):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)
        self.learning_rate = learning_rate

        # This assigns the activation function based on whats given
        if activation == "relu":
            self.activation_forward = self.relu
            self.activation_backward = self.relu_backward
        elif activation == "sigmoid":
            self.activation_forward = self.sigmoid
            self.activation_backward = self.sigmoid_backward
        elif activation == "softmax":
            self.activation_forward = self.softmax
            self.activation_backward = self.softmax_cross_entropy_loss_backward
    
    def forward(self, X):
        self.X = X # Needs to be stored for the backpropogation
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self.activation_forward(self.Z)
        return self.A
    
    def backward(self, dZ):
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0)
        dX = np.dot(dZ, self.weights.T)

        # update parameters
        self.weights -= self.learning_rate * dW
        self.biases -= self.learning_rate * db

        return dX

    # Here are all of the forward and backword activation functions
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_backward(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_backward(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))  # prevent overflow
        return e_Z / np.sum(e_Z, axis=-1, keepdims=True) 
    
    def softmax_cross_entropy_loss(self, A, Y):
        m = Y.shape[0]
        loss = -np.log(A[np.arange(m), Y.argmax(axis=1)]).mean()
        return loss

    def softmax_cross_entropy_loss_backward(self, A, Y):
        m = Y.shape[0]
        A[np.arange(m), Y.argmax(axis=1)] -= 1
        return A/m
    

class Sequential:
    def __init__(self, df, output, layers=[]):
        self.df = df # This is the dataframe for the model
        self.layers = layers
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
    def Standardize(self):
        for feature_name in self.df.columns:
            if self.df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                mean_value = self.df[feature_name].mean()
                std_value = self.df[feature_name].std()
                self.df[feature_name] = (self.df[feature_name] - mean_value) / std_value
            elif self.df[feature_name].dtype == 'bool':
                self.df[feature_name] = self.df[feature_name].astype(float)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def Run(self, epochs):
        self.OneHotEncode()
        self.Standardize()

        X = self.df.drop(self.target, axis=1).values
        y = self.df[self.target].values
        
        for i in range(epochs):
            # forward propagation
            A = self.forward(X)
            
            # compute loss
            # Here, we should define some loss function. For now, let's assume you have defined a function `compute_loss` which computes loss between predicted (A) and true values (y).
            loss = self.compute_loss(A, y)
            print(f'Epoch {i}, loss: {loss}')
            
            # backward propagation
            self.backward(loss)
