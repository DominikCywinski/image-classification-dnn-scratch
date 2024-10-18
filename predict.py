import numpy as np
import os
import matplotlib.pyplot as plt
import re

from image_classification_utils import predict, load_model, load_data, convert_image


if __name__ == "__main__":
    folder_path = "predict_data"
    model_path = "./models/model_64_64_3.pkl"

    _, _, _, _, classes = load_data("datasets/train_catvnoncat.h5", "datasets/test_catvnoncat.h5")
    parameters = load_model(model_path)
    org_size = [int(num) for num in re.findall(r"\d+", model_path)]

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            png_path = os.path.join(folder_path, file_name)
            image, converted_image = convert_image(png_path, org_size)

            my_predicted_image = predict(converted_image, parameters)

            plt.imshow(image)
            plt.title(classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"))
            plt.show()
