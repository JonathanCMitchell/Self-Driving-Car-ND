from sklearn import preprocessing
import numpy as np
def normalize(image_data):
    """
    Max Min normalization
    :param image_data:
    :return: image_data_normalized
    """
    a = np.float32(-0.5)
    b = np.float32(0.5)
    fill_data = np.zeros(image_data.shape)
    x_max = np.max(image_data, axis = 0)
    x_min = np.min(image_data, axis = 0)
    fill_data = a + (image_data - x_min)*(b-a)/(x_max - x_min)
    return fill_data

def preprocess_data(X, y):
    """

    :param X: input image data
    :param y: input labels
    :return: (input image data normalized, labels_one_hot)
    """
    X_normalized = normalize(X)

    lb = preprocessing.LabelBinarizer()

    y_one_hot = lb.fit_transform(y)

    return (X_normalized, y_one_hot)
