import numpy as np

# read csv file into np array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

# calculate the value using sigmoid
def sigmoid(int):
    return 1 / (1 + np.exp(-int))

# calculate the value of the derivative of sigmoid (input val from sigmoid)
def sigmoid_derivative(int):
    return int * (1 - int)

def forward_feed(weights, inputs, bias):
    # each output is summation of each input val * corresponding weight
    # output should be 1d array
    result = np.matmul(weights, inputs)
    if (type(result) == 'numpy.ndarray'):
        output = np.zeros(len(result))
        for i in range(len(result)):
            output[i] = (sigmoid(result[i]) + bias)
    else:
        output = sigmoid(result) + bias
    return output

def test(test_data, layer1, layer2):
    total = len(test_data)
    correct = 0

    for row in test_data:
        actual = row[0]
        prediction = 0
        inputs = np.array(row[1:])
        # First, run the network (forward feed)
        # Get output value of all hidden nodes first
        hidden_val_output = forward_feed(layer1, inputs, 0)

        #Feed output forward to calculate output of neural net
        output_val = forward_feed(layer2, hidden_val_output.T, 0)

        # Round the final answer for the prediction
        if (output_val > .5):
            prediction = 1
        else:
            prediction = 0

        # Check if the prediction is equal to the actual value
        if (prediction == actual):
            correct += 1

    return correct / total

test_set = read_csv('./data/mnist_test_0_1.csv')

layer1_weights = np.loadtxt('layer1_weights.csv', delimiter=',')
layer2_weights = np.loadtxt('layer2_weights.csv', delimiter=',')

accuracy = test(test_set, layer1_weights, layer2_weights)
print('Accuracy of the model is: ')
print(accuracy)
