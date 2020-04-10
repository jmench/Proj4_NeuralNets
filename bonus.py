#Author: Jordan Menchen
import numpy as np
import random

# read csv file into np array
def read_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

# gets user input to determine what datasets to use
def get_dataset():
    while(1):
        print('Which dataset do you want to use?')
        print('Enter 1 for 0-1 OR 2 for 0-4\n')
        choice = input('What is your choice? ')
        if (choice == '1'):
            #read in the train and test files to np arrays
            test_set = read_csv('./data/mnist_test_0_1.csv')
            train_set = read_csv('./data/mnist_train_0_1.csv')
            outputs = 2
            return train_set, test_set, outputs
        elif (choice == '2'):
            #read in the train and test files to np arrays
            test_set = read_csv('./data/mnist_test_0_4.csv')
            train_set = read_csv('./data/mnist_train_0_4.csv')
            outputs = 5
            return train_set, test_set, outputs
        else:
            print('ERR: INVALID CHOICE.\n')

# calculate the value using sigmoid
def sigmoid(num):
    return 1 / (1 + np.exp(-num * 1.0))

# calculate the value of the derivative of sigmoid (input val from sigmoid)
def sigmoid_derivative(num):
    return num * (1 - num)

# runs the network to get output of next layer
def forward_feed(weights, inputs, bias):
    # outputs holds all output values for Weights x Inputs
    outputs = np.matmul(weights, inputs)
    # Run activation function and add bias for each output value
    for i in range(len(outputs)):
        outputs[i] = (sigmoid(outputs[i]) + bias)
    return outputs

# returns an array with summation values for layer2 weight
def get_summation(layer2_weights, delta_j_vals):
    # need summation value for every hidden node
    summations = np.zeros(len(layer2_weights[0]))
    # For every delta value per output node
    for i in range(len(delta_j_vals)):
        # For every hidden node
        for j in range(len(summations)):
            # Add to the summation the product of the layer2_weight at the hidden_node to the output and the delta_j value at the output
            summations[j] += layer2_weights[i][j] * delta_j_vals[i]
    return summations

# returns numpy arrays with dimension x,y that have random values between -1 and 1
def get_random_weights(x, y):
    weight_arr = np.random.rand(x, y)
    '''for i in range(x):
        for j in range(y):
            randnum = random.random()
            if (randnum > .5):
                weight_arr[i][j] *= -1
    '''
    return weight_arr

# returns updated weight values
def update_weights(weights, learn_rate, node_output_val, delta_vals, x, y):
    updated_weights = weights
    for i in range(x):
        for j in range(y):
            updated_weights[i][j] = weights[i][j] + (learn_rate * node_output_val[j] * delta_vals[i])
    return updated_weights

# returns arr of trained weight values
def get_model(data, learn_rate, epochs, hidden_nodes, num_outputs, b1, b2):
    num_inputs = len(data[0]) - 1
    # Get random weights
    layer1_weights = get_random_weights(hidden_nodes, num_inputs)
    layer2_weights = get_random_weights(num_outputs, hidden_nodes)
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
            hidden_layer_output = forward_feed(layer1_weights, inputs, b1)

            #Feed output from hidden layer forward to calculate output of neural net
            output_vals = forward_feed(layer2_weights, hidden_layer_output, b2)

            #Now back propogate through to update weights
            # First task is to get delta values for output nodes
            delta_j_vals = np.zeros(num_outputs)
            for i in range(num_outputs):
                delta_j_vals[i] = sigmoid_derivative(output_vals[i]) * (actual - output_vals[i])

            # Use delta j values to get all delta values for hidden nodes
            delta_i_vals = np.zeros(hidden_nodes)
            # summations hold the summation value for Weight(hidden,output)*delta_j(output)
            summations = get_summation(layer2_weights, delta_j_vals)
            for i in range(hidden_nodes):
                delta_i_vals[i] = sigmoid_derivative(hidden_layer_output[i]) * summations[i]

            #Update weights
            layer2_weights = update_weights(layer2_weights, learn_rate, hidden_layer_output, delta_j_vals, num_outputs, hidden_nodes)
            layer1_weights = update_weights(layer1_weights, learn_rate, inputs, delta_i_vals, hidden_nodes, num_inputs)

            row_count+=1
        # next row in data set
    # after all epochs, return final weight values
    return layer1_weights, layer2_weights

# Tests the model against the test dataset to determine accuracy
def test(test_data, layer1, layer2):
    total = len(test_data)
    correct = 0
    for row in test_data:
        actual = row[0]
        inputs = np.array(row[1:])
        # Get output value of all hidden nodes first without bias
        hidden_layer_output = forward_feed(layer1, inputs, 0)
        #Feed output forward to calculate output of neural net without bias
        output_vals = forward_feed(layer2, hidden_layer_output, 0)
        print(output_vals)
        # Prediction is the index of the output node with largest value
        prediction = np.argmax(output_vals)
        print('Prediction: ' + str(prediction))
        print('Actual: ' + str(actual) + '\n')
        # Check if prediction is correct
        if (prediction == actual):
            correct += 1
    return correct / total * 100

# driver for the program
def main():
    train_set, test_set, outputs = get_dataset()
    print('Training could take a while... check back in an hour or two for progress')

    #set constants
    b1 = random.random()
    b2 = random.random()
    learn_rate = 0.001
    epochs = 10
    hidden_nodes = 100

    layer1_weights, layer2_weights = get_model(train_set, learn_rate, epochs, hidden_nodes, outputs, b1, b2)

    np.savetxt('layer1_weights_1.csv', layer1_weights, delimiter=',')
    np.savetxt('layer2_weights_2.csv', layer2_weights, delimiter=',')

    #layer1_weights = np.loadtxt('layer1_weights_BONUS.csv', delimiter=',')
    #layer2_weights = np.loadtxt('layer2_weights_BONUS.csv', delimiter=',')

    print(b1)
    print(b2)

    accuracy = test(test_set, layer1_weights, layer2_weights)
    print('Accuracy (%) of the model is: ')
    print(accuracy)

if __name__== "__main__": main()
