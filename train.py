import numpy as np
from image_classification_utils import load_data, L_layer_model, save_model


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data(
        "datasets/train_catvnoncat.h5", "datasets/test_catvnoncat.h5"
    )

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    nn_input_size = np.prod(train_x_orig.shape[1:])
    layers_dims = [nn_input_size, 20, 7, 5, 1]  #  4-layer model

    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=True)

    save_model(
        parameters,
        file_path=f"./models/model_{train_x_orig.shape[1]}_{train_x_orig.shape[2]}_{train_x_orig.shape[3]}.pkl",
    )
