from scipy.misc import imresize
import numpy as np


def resize_data(img):
    M = imresize(img, (8, 8)).flatten()
    for i in range(64):
        if (M[i] > 0):
            M[i] = 1
    return M


# read the next image in the image file
def read_next_image(file):
    img = []
    line = file.readline().strip()
    if (line == ""):
        return None, None

    while (len(line) > 1):
        img_line = []
        for ch in line:
            img_line.append(float(ch))
        img.append(img_line)
        line = file.readline().strip()
    return np.array(img), float(line)


# read the next image in the test file
def read_next_test(file):
    img = []
    line = file.readline().strip()
    if (line == ""):
        return None, None

    while (len(line) > 1):
        img_line = []
        for ch in line:
            img_line.append(float(ch))
        img.append(img_line)
        line = file.readline().strip()
    return np.array(img), float(line)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    hidden_layer = np.random.random((n_inputs, n_hidden)) - 1
    output_layer = np.random.random((n_hidden, n_outputs)) - 1
    return hidden_layer, output_layer


# Transfer neuron activation
def transfer(weight):
    return 1.0 / (1.0 + np.exp(-weight))


# Calculate the derivative of an neuron output
def transfer_derivative(weight):
    return transfer(weight) * (1 - transfer(weight))


# train the network
def train_network(hidden_layer_weight, output_layer_weight, X_train, l_rate, n_epoch):
    for epoch in range(n_epoch):
        total_error = []
        for X, Y in zip(X_train, Y_train):
            # feed forward
            X = np.array(X)
            layer1 = transfer(np.dot(X, hidden_layer_weight))  # calculation of f(net_j)
            layer2 = transfer(np.dot(layer1, output_layer_weight))  # calculation of f(net_k)
            error = np.subtract(Y, layer2)  # calculation of error (t_k - z_k)
            total_error.append(np.mean(np.abs(error)))

            # back propagation
            layer2_delta = error * transfer_derivative(layer2)  # calculation of del(w_jk)
            layer1_error = np.dot(layer2_delta, output_layer_weight.T)
            layer1_delta = layer1_error * transfer_derivative(layer1)  # calculation of del(w_ij)

            # update
            hidden_layer_weight_update = l_rate * (np.dot(np.matrix(X).T, np.matrix(layer1_delta)))
            hidden_layer_weight += hidden_layer_weight_update
            output_layer_weight_update = l_rate * np.dot(np.matrix(layer1).T, np.matrix(layer2_delta))
            output_layer_weight += output_layer_weight_update
        # print average error
        print epoch + 1, np.mean(total_error)

    return hidden_layer_weight, output_layer_weight


# Make a prediction with a network
def predict(layer1, layer2, x):
    x = np.array(x)
    layer1 = transfer(np.dot(X, layer1))  # calculation of f(net_j)
    layer2 = transfer(np.dot(layer1, layer2))  # calculation of f(net_k)
    return np.argmax(layer2)


np.random.seed(1)

training_file = open("Dataset1/optdigits-orig.tra", "r")
testing_file = open("Dataset1/optdigits-orig-sample.tra", "r")
N = 33

# skip first 21 lines
for i in range(21):
    training_file.readline()
    testing_file.readline()

X_train = []
Y_train = []
X_test = []
Y_test = []

# read and pre-process Train dataset
img, label = read_next_image(training_file)
while (img is not None):
    X_train.append(resize_data(img))
    temp = [0.0] * 10
    temp[int(label)] = 1.0
    Y_train.append(temp)
    img, label = read_next_image(training_file)

# read and pre-process Train dataset
img, label = read_next_test(testing_file)
while (img is not None):
    X_test.append(resize_data(img))
    Y_test.append(label)
    img, label = read_next_test(testing_file)

n_inputs = 64
n_outputs = 10
n_hidden = 50
learning_rate = 0.5
n_epoch = 20
hidden_layer_weight, output_layer_weight = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(hidden_layer_weight, output_layer_weight, X_train, learning_rate, n_epoch)
for X, Y in zip(X_test, Y_test):
    prediction = predict(hidden_layer_weight, output_layer_weight, X)
    print('Expected = %d, Got = %d' % (Y, prediction))

