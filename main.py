#Author: Jordan Menchen
import numpy as np
import random

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

# returns arr of trained weight values
def get_model(data, learn_rate, epochs, hidden_nodes, b1, b2):
    # n = num hidden nodes, m = num inputs
    n = len(hidden_nodes)
    m = len(data[0]) - 1

    # layer1 between input and hidden is n x m matrix
    # n x m (50 x 784) so you can multiply by the num of inputs
    layer1_weights = np.random.rand(n, m)
    for i in range(n):
        for j in range(m):
            randnum = random.random()
            if (randnum > .5):
                layer1_weights[i][j] *= -1
    # layer2 between hidden and output is n x 1 matrix
    layer2_weights = np.random.rand(n)
    for i in range(len(layer2_weights)):
        randnum = random.random()
        if (randnum > .5):
            layer2_weights[i] *= -1

    # Begin looping through each epoch
    for epoch in range(1, epochs+1):
        print('Epoch: ' + str(epoch))
        row_count = 1
        # Loop through every training example
        for row in data:
            if (row_count % 1000 == 0):
                print('Row: ' + str(row_count) + ' of ' + str(len(data)))
            actual = row[0]
            inputs = np.array(row[1:])
            # First, run the network (forward feed)
            # Get output value of all hidden nodes first
            hidden_val_output = forward_feed(layer1_weights, inputs, b1)

            #Feed output forward to calculate output of neural net
            output_val = forward_feed(layer2_weights, hidden_val_output.T, b2)

            #Now back propogate through to update weights
            delta_j = sigmoid_derivative(output_val) * (actual - output_val)
            delta_vals = hidden_val_output
            for i in range(len(hidden_val_output)):
                delta_vals[i] = sigmoid_derivative(hidden_val_output[i]) * layer2_weights[i] * delta_j

            #Update weights
            # TODO: try and add stopping criteria for weight changes
            weight_change = list()
            temp = layer2_weights
            for i in range(len(layer2_weights)):
                temp[i] = layer2_weights[i] + (learn_rate * hidden_val_output[i] * delta_j)
                if (temp[i] != layer2_weights[i]):
                    percent_change = (abs(temp[i] - layer2_weights[i]) / ((temp[i] + layer2_weights[i]) / 2)) * 100
                    weight_change.append(percent_change)
            layer2_weights = temp

            temp = layer1_weights
            for i in range(len(layer1_weights)):
                for j in range(len(layer1_weights[0])):
                    temp[i][j] = layer1_weights[i][j] + (learn_rate * inputs[j] * delta_vals[i])
                    if (temp[i][j] != layer1_weights[i][j]):
                        percent_change = (abs(temp[i][j] - layer1_weights[i][j]) / ((temp[i][j] + layer1_weights[i][j]) / 2)) * 100
                        weight_change.append(percent_change)
            layer1_weights = temp
            row_count+=1
            #Calculate the avg percent change for all weights. If the change is < .1%, hit stopping criteria
            if (len(weight_change) != 0):
                avg_change = np.mean(np.array(weight_change))
            else:
                avg_change = 1
            if ((avg_change < .1 and row_count > 500) or (avg_change < .1 and epoch > 1)):
                print('row_count is: ' + str(row_count))
                print('we are on epoch: ' + str(epoch))
                print('Reached stopping criteria - weight change < 0.1%')
                return layer1_weights, layer2_weights
    return layer1_weights, layer2_weights

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

    return correct / total * 100

def main():
    np.set_printoptions(precision = 5)

    #read in the train and test files to np arrays
    test_set = read_csv('./data/mnist_test_0_1.csv')
    train_set = read_csv('./data/mnist_train_0_1.csv')

    #set constants
    b1 = -.3
    b2 = .03
    learn_rate = 0.5
    epochs = 3
    hidden_nodes = np.zeros(100)

    layer1_weights, layer2_weights = get_model(train_set, learn_rate, epochs, hidden_nodes, b1, b2)
    np.savetxt('layer1_weights.csv', layer1_weights, delimiter=',')
    np.savetxt('layer2_weights.csv', layer2_weights, delimiter=',')

    accuracy = test(test_set, layer1_weights, layer2_weights)
    print('Accuracy (%) of the model is: ')
    print(accuracy)

if __name__== "__main__": main()
