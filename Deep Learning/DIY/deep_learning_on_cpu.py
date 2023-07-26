import numpy as np

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def manual_train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    test_indices = np.random.choice(indices, size=int(X.shape[0]*test_size), replace=False)
    train_indices = np.array(list(set(indices) - set(test_indices)))

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def normalize(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

def one_hot_encode(y):
    classes = np.unique(y)
    y_encoded = np.zeros((len(y), len(classes)))

    for i, label in enumerate(y):
        y_encoded[i, label] = 1

    return y_encoded

class multi_classification:
    def __init__(self, weight_decay=0.001):
        self.weight_decay = weight_decay
        print("Successfully created empty model")
    
    def fit(self, X_train, y_train):
        self.classes = y_train.shape[1]

        np.random.seed(1)

        # Define the architecture of the model
        input_nodes = X_train.shape[1]
        hidden_nodes = 4  # You can change this as per your requirements

        # Initialize weights randomly with mean 0
        self.synaptic_weights_1 = 2 * np.random.random((input_nodes, hidden_nodes)) - 1
        self.synaptic_weights_2 = 2 * np.random.random((hidden_nodes, self.classes)) - 1

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
        new_layer_0 = X
        new_layer_1 = 1 / (1 + np.exp(-(np.dot(new_layer_0, self.synaptic_weights_1))))  # sigmoid
        new_layer_2 = softmax(np.dot(new_layer_1, self.synaptic_weights_2))
        return new_layer_2

    def score(self, X_test, y_test):
        new_layer_0 = X_test
        new_layer_1 = 1 / (1 + np.exp(-(np.dot(new_layer_0, self.synaptic_weights_1))))  # sigmoid
        new_layer_2 = softmax(np.dot(new_layer_1, self.synaptic_weights_2))

        predictions = np.argmax(new_layer_2, axis=1)
        actual = np.argmax(y_test, axis=1)

        accuracy = np.mean(predictions == actual)
        return accuracy

data = np.genfromtxt('WineQuality.csv', delimiter=',', skip_header=1)

X = data[:, 1:-1]  # Assuming the first column is an index like 'Unnamed: 0' and the last column is the target
y = data[:, -1].astype(int)

X = normalize(X)
y = one_hot_encode(y)

X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

model = multi_classification()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
