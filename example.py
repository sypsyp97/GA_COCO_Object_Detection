import os
import pickle

# Set environment variables to reduce TensorFlow logging and set a consistent hash seed.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "1234"

import gc
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from dataset.Dataset import test_ds, train_ds, val_ds
# Import necessary modules.
from src.Evolutionary_Algorithm import start_evolution

# Set random seeds for consistent output.
random.seed(1234)
tf.random.set_seed(1234)
np.random.seed(1234)

# Load previously evolved population from a pickle file.
with open("results_25042023190939/next_population_array.pkl", "rb") as f:
    data = pickle.load(f)
    f.close()

# Main execution.
if __name__ == "__main__":
    # Enable garbage collector to free up memory.
    gc.enable()

    # Get current time for file naming.
    now = datetime.now()
    formatted_date = now.strftime("%d%m%Y%H%M%S")

    print(train_ds, val_ds, test_ds)

    # Start the evolutionary algorithm with the loaded population.
    (
        population_array,
        max_fitness_history,
        average_fitness_history,
        best_models_arrays,
    ) = start_evolution(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        generations=5,
        population=20,
        num_classes=4,
        epochs=70,
        population_array=data,
        time=formatted_date,
    )
