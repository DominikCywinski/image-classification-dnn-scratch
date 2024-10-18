from image_classification_utils import load_data, predict, load_model


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data(
        "datasets/train_catvnoncat.h5", "datasets/test_catvnoncat.h5"
    )

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    model_path = "./models/model_64_64_3.pkl"
    parameters = load_model(model_path)

    print("Train accuracy: ")
    predict(train_x, parameters, train_y)

    print("Test accuracy: ")
    predict(test_x, parameters, test_y)
