# calculate the fitness
import numpy as np
from math import pi


def calculate_fitness(iou, inference_time):
    print("IOU:", iou)
    print("Inference Time:", inference_time)
    fitness = (1 - np.arctan(inference_time / 200) / (pi / 2)) * iou
    print("Fitness:", fitness)
    return fitness
