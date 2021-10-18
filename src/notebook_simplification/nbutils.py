import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random


def discretize_predictions(predictions):
    """
    Transform a Series of predictions with shape [number_of_predictions x possible_prediction_outcomes] to a simplified
    series of predictions with shape [number_of_predictions].

    :example:
    prediction_1 = [0, 0, 0, 1]                          # 4 possible outcomes
    prediction_2 = [1, 0, 0, 0]
    all_predictions = [prediction_1, prediction_2]       # shape = [2 x 4]
    discretize_predictions(all_predictions)
    > [3, 0]                                             # shape = [2]

    :param predictions: matrix (numpy.ndarray or similar)
    """
    return np.argmax(predictions, axis=1)


def plot_random_image(images, labels=None, predictions=None):
    """
    Grab a random image and plot it. If labels or predictions are given then those are in the title.

    :param images: list of images
    :param labels: optional list of labels, same length as images
    :param predictions: optional list of predictions, same length as images
    """
    n = np.random.randint(0, len(images))
    plot_specific_image(n, images, labels, predictions)


def plot_specific_image(n, images, labels=None, predictions=None):
    """
    Grab a specific image and plot it. If labels or predictions are given then those are in the title.

    :param n: the index of the image to plot
    :param images: list of images
    :param labels: optional list of labels, same length as images
    :param predictions: optional list of predictions, same length as images
    """
    plt.figure()
    plt.imshow(images[n], cmap='gray')
    title = f"Image id: {n}"
    if labels is not None:
        title += f" - Truth: {labels[n]}"
    if predictions is not None:
        title += f" - Prediction: {predictions[n]}"
    plt.title(title)


def explain(variable, name=""):
    """
    Show a brief overview of a variable, including type, size and a sample.

    :param variable: any variable
    :param name: optional name of the variable used in the title
    """
    print()
    print(f"Explanation of variable {name}")
    print("===============================")
    print(f"The type of this variable is {type(variable)}")

    try:
        print(f"The dimensions of this variable are {variable.shape}")
    except AttributeError:
        try:
            print(f"The length of this variable is '{len(variable)}'")
        except TypeError:
            print(f"The dimensions of this variable are unknown, meaning it is either 0 dimensional or complex")

    sample = str(variable)
    if len(sample) > 77:
        sample = sample[:77] + "..."
    print("Sample of the data:")
    print(sample)
    print()


def help():
    """
    This function explains the current module.

    :example:
    help()
    """
    print(
        "\n"
        "WELCOME!\n"
        "This library contains various functions to help you keep your notebooks straightforward. It has been developed"
        " for a specific training and functions might not be as generic as you like.\n"
        "Below is a list of helper-functions available:"
    )
    for var, value in globals().items():
        if not str(var).startswith("__") and callable(value):
            print(f" - {var}")
    print("\n\nFor more information run `function_name?` in seperate cell. For example `help?`")


def __sample_by_label(x, y, label, ratio):
    indexes = np.flatnonzero(y == label)
    count = int(len(indexes) * ratio)
    sampled_indexes = np.random.choice(indexes, count, replace=False)
    return x[sampled_indexes], y[sampled_indexes]


def sample(x, y, target: dict):
    """
    Given a dataset and labels, sample ratio's of specific classes given a dictionary.

    :param x: dataset
    :param y: labels
    :param target: dict with {label: ratio} e.g. {3: 0.99}
    :return: new x, new y
    """
    rebalanced_x, rebalanced_y = [], []
    for label, ratio in target.items():
        sampled_x, sampled_y = __sample_by_label(x, y, label, ratio)
        rebalanced_x.append(sampled_x)
        rebalanced_y.append(sampled_y)
    return np.concatenate(rebalanced_x), np.concatenate(rebalanced_y)


def histogram(classes):
    """
    Draw a histogram for the MNIST dataset

    :param classes: labels
    """
    plt.figure(figsize=(10, 5))
    plt.title('Value counts')
    plt.hist(classes, bins=range(10), rwidth=0.8, align='left')
    plt.xticks(range(10))


def replace(classes, old, new):
    """
    Replace a value in an array or list

    :param classes: array or list of labels
    :param old: value to replace
    :param new: value to replace with
    :return: new array
    """
    return np.where(classes == old, new, classes)


def plot_confusion_matrix(truths, predictions, labels=None):
    """
    Plot a confusion matrix

    :param truths: list of should-be values
    :param predictions: list of predicted values
    :param labels: names of classes to write on the side
    """
    cm = confusion_matrix(truths, predictions)
    if labels:
        xticklabels = labels
        yticklabels = labels
    else:
        xticklabels = "auto"
        yticklabels = "auto"
    ax = sn.heatmap(cm, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, fmt="d")
    ax.set(xlabel='Prediction', ylabel='Truth')
    ax.set_title("Confusion Matrix")


def zero_pad(image, new_x=56, new_y=56):
    """
    Increase the size of an image by padding to zeros to match new size

    :param image: an image (for example 28x28 MNIST)
    :param new_x: new x size in pixels
    :param new_y: new y size in pixels
    :return: enlarged image
    """
    old_x, old_y = image.shape

    pos_x = random.randint(0, new_x - old_x)
    pos_y = random.randint(0, new_y - old_y)

    z = np.zeros((new_x, new_y))
    z[pos_x: pos_x + old_x, pos_y: pos_y + old_y] = image

    return z


def plot_image(image, label=None, prediction=None):
    """
    Plot a single image

    :param image: the image
    :param label: optional label
    :param prediction: optional prediction
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    title = ""
    if label is not None:
        title += f" - Truth: {label}"
    if prediction is not None:
        title += f" - Prediction: {prediction}"
    plt.title(title)


def get_filters(layer):
    """
    Return a list of the filters of a convolutional layer

    :param layer: the layer (model.layers[n])
    :return: a list of filters
    """
    filters, biases = layer.get_weights()
    return [filters[:, :, 0, f] for f in range(filters.shape[3])]
