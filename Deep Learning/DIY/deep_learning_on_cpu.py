import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]])

# Output dataset            
y = np.array([[0,1,1,0]]).T

# Seed the random number generator
np.random.seed(1)

# Initialize weights randomly with mean 0
synaptic_weights_1 = 2 * np.random.random((3,4)) - 1
synaptic_weights_2 = 2 * np.random.random((4,1)) - 1

for iteration in range(10000):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synaptic_weights_1))
    layer_2 = sigmoid(np.dot(layer_1, synaptic_weights_2))
    
    # Calculate error (actual - predicted)
    layer_2_error = y - layer_2

    if iteration % 1000 == 0:
        print("Error:" + str(np.mean(np.abs(layer_2_error))))
    
    # Back propagation
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synaptic_weights_2.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    synaptic_weights_2 += layer_1.T.dot(layer_2_delta)
    synaptic_weights_1 += layer_0.T.dot(layer_1_delta)

print("Output after training:")
print(layer_2)
