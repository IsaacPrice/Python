import numpy as np
import pandas as pd

# Softmax function (This is the alternate to the sigmoid, as this is multi-class)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0) + 1e-9)

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def manual_train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    test_indices = np.random.choice(indices, size=int(X.shape[0]*test_size), replace=False)
    train_indices = np.array(list(set(indices) - set(test_indices)))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if df[feature_name].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


class multi_classification:
    def __init__(self, weight_decay=0.001):
        self.weight_decay = weight_decay
        print("Successfully created empty model")
    
    def fit(self, X_train, y_train):
        self.classes = len(y_train.columns)

        np.random.seed(1)

        # Define the architecture of the model
        input_nodes = X_train.shape[1]
        hidden_nodes = 4  # You can change this as per your requirements

        # Initialize weights randomly with mean 0
        self.synaptic_weights_1 = 2 * np.random.random((input_nodes, hidden_nodes)) - 1
        self.synaptic_weights_2 = 2 * np.random.random((hidden_nodes, self.classes)) - 1

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        for iteration in range(10000):
            # Forward propagation
            self.layer_0 = X_train
            self.layer_1 = 1 / (1 + np.exp(-(np.dot(self.layer_0, self.synaptic_weights_1)))) 
            self.logits = np.dot(self.layer_1, self.synaptic_weights_2)
            self.layer_2 = softmax(self.logits)

            # Calculate error (actual - predicted)
            self.layer_2_error = y_train - self.layer_2

            if iteration % 1000 == 0:
                print("Error:" + str(np.mean(np.abs(self.layer_2_error))))

            # Back propagation
            layer_2_delta = self.layer_2_error  # Gradient for softmax + cross-entropy
            layer_1_error = layer_2_delta.dot(self.synaptic_weights_2.T)
            layer_1_delta = layer_1_error * sigmoid_derivative(self.layer_1)

            # Update weights with weight decay
            self.synaptic_weights_2 += self.layer_1.T.dot(layer_2_delta) - self.weight_decay * self.synaptic_weights_2
            self.synaptic_weights_1 += self.layer_0.T.dot(layer_1_delta) - self.weight_decay * self.synaptic_weights_1
        
        return "Finished Fitting Successfully"

    def predict(self, X):
        # This will predict the given data
        X = X.to_numpy()
        new_layer_0 = X
        new_layer_1 = 1 / (1 + np.exp(-(np.dot(new_layer_0, self.synaptic_weights_1))))  # sigmoid
        new_layer_2 = softmax(np.dot(new_layer_1, self.synaptic_weights_2))

        return new_layer_2


    def score(self, X_test, y_test):
        # This will go through and predict the values for each value and compare to the answers and get the average to return
        new_layer_0 = X_test.to_numpy()
        new_layer_1 = 1 / (1 + np.exp(-(np.dot(new_layer_0, self.synaptic_weights_1))))  # sigmoid
        new_layer_2 = softmax(np.dot(new_layer_1, self.synaptic_weights_2))

        # find the class with the  highest probability for each example in the predictions
        predictions = np.argmax(new_layer_2, axis=1)

        # find the actual class for each example in the test set
        actual = np.argmax(y_test.to_numpy(), axis=1)

        # calculate the accuracy: the proportion of predictions that exactly match the actual classes
        accuracy = np.mean(predictions == actual)

        return accuracy



# Creating the data frame
df = pd.read_csv("WineQuality.csv")

# Sorting the data into whats neccisary
df = df.drop('Unnamed: 0', axis='columns')
dummies = pd.get_dummies(df.Type)
df = df.drop('Type', axis='columns')
df.dropna(inplace=True)
#df = pd.concat([df, dummies], axis='columns')

# Seperating the key from the values
X = df.drop('quality', axis='columns')
y = pd.get_dummies(df.quality)

X = normalize(X)

# splitting the data into train and test
X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

# Finally creating the model
model = multi_classification()

# Fit the data to the model
model.fit(X_train, y_train)

# get the score
print(model.score(X_test, y_test))