import os
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTHONHASHSEED'] = "1234"

from src.Evolutionary_Algorithm import start_evolution
from dataset.Dataset import train_ds, val_ds, test_ds
import tensorflow as tf
import gc
from datetime import datetime
import numpy as np
import random

random.seed(1234)
tf.random.set_seed(1234)
np.random.seed(1234)


with open('arrays/first_population_array.pkl', 'rb') as f:
    data = pickle.load(f)
    f.close()


if __name__ == '__main__':
    gc.enable()
    now = datetime.now()
    formatted_date = now.strftime("%d%m%Y%H%M%S")

    print(train_ds, val_ds, test_ds)

    population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        generations=20,
        population=20,
        num_classes=4,
        epochs=10,
        population_array=data,
        time=formatted_date)
