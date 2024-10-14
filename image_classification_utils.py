import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

from PIL import Image

np.random.seed(1)


def convert_image(png_path, org_size):
    image_file = Image.open(png_path).convert("RGB").resize([org_size[0], org_size[1]])
    image = np.array(image_file)

    converted_image = image.reshape(np.prod(org_size), 1)
    converted_image = converted_image / 255.0
    return image, converted_image


def save_model(parameters, file_path="./models/model_64_64_3.pkl"):
    with open(file_path, "wb") as file:
        pickle.dump(parameters, file)


def load_model(file_path="./models/model_64_64_3.pkl"):
    with open(file_path, "rb") as file:
        parameters = pickle.load(file)

    return parameters


def load_data(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters_deep(layer_dims):
    """
    Args:
    layer_dims - dimensions of each layer in our network

    Returns:
    parameters - python dictionary containing:
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
    """

    layers_num = len(layer_dims)  # number of layers in the network
    parameters = {}

    for l in range(1, layers_num):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layer_dims[l], 1)

    return parameters


def sigmoid(Z):
    """
    Sigmoid activation

    Args:
    Z - numpy array

    Returns:
    A - output of sigmoid(Z)
    cache -- returns Z for backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    assert A.shape == Z.shape

    return A, cache


def relu(Z):
    """
    ReLU activation

    Args:
    Z - Output of the linear layer

    Returns:
    A - output of ReLU(Z)
    cache - return Z for backpropagation
    """

    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    """
    Backward propagation for a single ReLU unit.

    Args:
    dA - post-activation gradient
    cache - 'Z' stored for backward propagation efficiently

    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ


def sigmoid_backward(dA, cache):
    """
    Backward propagation for a single sigmoid unit.

    Args:
    dA - post-activation gradient
    cache - 'Z' stored for computing backward propagation efficiently

    Returns:
    dZ - Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert dZ.shape == Z.shape

    return dZ


def linear_forward(A, W, b):
    """
    Linear part of a layer's forward propagation.

    Args:
    A - activations from previous layer
    W - weights matrix
    b - bias vector

    Returns:
    Z - the input of the activation function
    cache -- a python dictionary containing "A", "W" and "b" for computing the backward pass
    """

    Z = W.dot(A) + b

    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer

    Args:
    A_prev - activations from previous layer
    W - weights matrix
    b - bias vector
    activation - the activation to be used in this layer, text string: "sigmoid" or "relu"

    Returns:
    A - the output of the activation function
    cache - a python dictionary containing "linear_cache" and "activation_cache" for computing the backward pass
    """

    if activation.lower() == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation.lower() == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID computation

    Args:
    X - input data( input size, number of examples)
    parameters - output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    layer_number = len(parameters) // 2

    for l in range(1, layer_number):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A,
        parameters["W" + str(layer_number)],
        parameters["b" + str(layer_number)],
        activation="sigmoid",
    )
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


def linear_backward(dZ, cache):
    """
    Linear portion of backward propagation for a single layer (layer l)

    Args:
    dZ - Gradient of the cost with respect to the linear output of layer l
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the layer l

    Returns:
    dA_prev - Gradient of the cost with respect to the activation of the layer l-1
    dW - Gradient of the cost with respect to W layer l
    db - Gradient of the cost with respect to b layer l
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.

    Args:
    dA - post-activation gradient for current layer l
    cache - tuple of values (linear_cache, activation_cache) for computing backward propagation
    activation - the activation to be used in this layer, as text string: "sigmoid" or "relu"

    Returns:
    dA_prev - Gradient of the cost with respect to the activation of the layer l-1
    dW - Gradient of the cost with respect to W layer l
    db - Gradient of the cost with respect to b layer l
    """

    linear_cache, activation_cache = cache

    if activation.lower() == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation.lower() == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Args:
    AL - probability vector, output of the forward propagation (L_model_forward())
    Y - true "label" vector (0 if non-cat, 1 if cat)
    caches - list of caches containing:
        every cache of linear_activation_forward() with "relu"
        the cache of linear_activation_forward() with "sigmoid"

    Returns:
    grads -- A dictionary with the gradients
         grads["dA" + str(l)]
         grads["dW" + str(l)]
         grads["db" + str(l)]
    """

    grads = {}
    layers_number = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[layers_number - 1]
    (
        grads["dA" + str(layers_number - 1)],
        grads["dW" + str(layers_number)],
        grads["db" + str(layers_number)],
    ) = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(layers_number - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Args:
    parameters - dictionary containing parameters
    grads - dictionary containing gradients, output of L_model_backward

    Returns:
    parameters - dictionary containing updated parameters:
          parameters["W" + str(l)]
          parameters["b" + str(l)]
    """

    layers_number = len(parameters) // 2

    for l in range(layers_number):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def compute_cost(AL, Y):
    """
    Cost function

    Args:
    AL - probability vector corresponding to label predictions
    Y - true "label" vector

    Returns:
    cost - cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss
    cost = (1.0 / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)
    assert cost.shape == ()

    return cost


def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=1000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Args:
    X - input data
    Y - true "label" vector (0 if cat, 1 if non-cat)
    layers_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations times 100")
    plt.title("Cost function")
    plt.show()

    return parameters


def predict(X, y, parameters):
    """
    Predict the results of a L-layer neural network.

    Ars:
    X - input data
    parameters - parameters of the trained model

    Returns:
    p - predictions for the given dataset X
    """

    m = X.shape[1]

    predicts, caches = L_model_forward(X, parameters)
    predicts[0, :] = predicts[0, :] > 0.5

    # print("Accuracy: " + str(np.sum((predicts == y) / m)))

    return predicts
