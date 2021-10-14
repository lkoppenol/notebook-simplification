import numpy as np
import matplotlib.pyplot as plt


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
        title += f" - Truth: {predictions[n]}"
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
