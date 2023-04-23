# calculate the fitness
import numpy as np
from math import pi


def calculate_fitness(iou, inference_time):

    fitness = (1 - np.arctan(inference_time / 500) / (pi / 2)) * iou

    return fitness
