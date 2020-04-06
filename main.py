#Author: Jordan Menchen
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

# returns arr of trained weight values
def get_model(data, learn_rate, epochs, hidden_nodes, bias):
    # n = num hidden nodes, m = num inputs
    n = hidden_nodes
    m = len(data[0]) - 1

    # layer1 between input and hidden is n x m matrix
    # n x m so you can multiply by the num of inputs
    layer1 = np.random.rand(m, n)
    # layer2 between hidden and output is n x 1 matrix
    layer2 = np.random.rand(n)

    # Begin looping through each epoch
    for i in range(epochs):
        print('Epoch: ' + str(i+1))
        # Loop through every training example
        for row in data:
            actual = row[0]
            pixels = row[1:]
            # First, run the network
            # Get value of all hidden nodes first
            for
    return layer1, layer2

def main():
    np.set_printoptions(precision = 5)

    #read in the train and test files to np arrays
    test_set = read_csv('./data/mnist_test_0_1.csv')
    train_set = read_csv('./data/mnist_train_0_1.csv')

    #set constants
    bias = 1
    learn_rate = 0.01
    epochs = 10
    hidden_nodes = np.zeros(50)

    layer1, layer2 = get_model(train_set, learn_rate, epochs, hidden_nodes, bias)

if __name__== "__main__": main()
