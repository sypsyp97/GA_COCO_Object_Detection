# calculate the fitness
from math import pi

import numpy as np


def calculate_fitness(iou, inference_time):
    """
    This function calculates the fitness value for a model based on intersection-over-union (IoU) and inference time.

    Parameters:
    ----------
    iou : float
        Intersection-over-union value of the model on a given dataset. This is a measure of the accuracy of the model.
    inference_time : float
        The time taken by the model to perform inference on the Edge TPU.

    Returns:
    -------
    fitness : float
        The calculated fitness value of the model.

    Note: The fitness is calculated as a weighted combination of IoU and inference time where a higher IoU and lower
    inference time lead to a higher fitness. The arctan function is used to smooth the influence of inference time,
    and the factor 200 is a scaling factor that can be adjusted.
    """
    print("IOU:", iou)
    print("Inference Time:", inference_time)
    fitness = (1 - np.arctan(inference_time / 200) / (pi / 2)) * iou
    print("Fitness:", fitness)
    return fitness
