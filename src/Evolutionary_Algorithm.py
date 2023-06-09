import gc
import os
import pickle

import numpy as np
from src.Compile_Edge_TPU import compile_edgetpu
from src.Create_Model import create_model, train_model
from src.Fitness_Function import calculate_fitness
from src.Inference_Speed_TPU import inference_time_tpu
from src.Model_Checker import model_has_problem
from src.TFLITE_Converter import convert_to_tflite


def create_first_population(population, num_classes=5):
    """Generate the initial population of models for a genetic algorithm.

    Parameters:
    population : int
        The number of models to generate.
    num_classes : int, optional
        The number of output classes in the model, defaults to 5.

    Returns:
    np.ndarray
        A 3D numpy array representing the initial population of models.
    """

    # Generate a 3D numpy array of random binary digits,
    # where the first dimension is the number of models,
    # and the second and third dimensions are the characteristics of each model.
    first_population_array = np.random.randint(0, 2, (population, 9, 18))

    # Loop over each model in the population
    for i in range(population):
        # Create a model based on the characteristics encoded in the array
        model = create_model(first_population_array[i], num_classes=num_classes)

        # If the model has a problem, delete it and create a new one with different random characteristics
        while model_has_problem(model):
            del model
            first_population_array[i] = np.random.randint(0, 2, (9, 18))
            model = create_model(first_population_array[i], num_classes=num_classes)

        # Delete the model to free up memory
        del model

    return first_population_array


def select_models(
    train_ds,
    val_ds,
    test_ds,
    time,
    population_array,
    generation,
    epochs=30,
    num_classes=34,
):
    """Trains models defined by the population_array, evaluates them, selects the best
    models based on fitness and saves the model information and fitness stats.

    Parameters:
    ----------
    train_ds : tf.data.Dataset
        The training dataset.

    val_ds : tf.data.Dataset
        The validation dataset.

    test_ds : tf.data.Dataset
        The testing dataset.

    time : str
        Timestamp used in the generation of the result directories.

    population_array : array-like
        The array containing the model architectures to be trained and evaluated.

    generation : int
        The current generation number in the evolutionary process.

    epochs : int, optional (default=30)
        Number of epochs for training each model.

    num_classes : int, optional (default=34)
        Number of classes for the classification task.

    Returns:
    -------
    best_models_arrays : list
        List of the architectures of the best models.

    max_fitness : float
        The maximum fitness score in the current generation.

    average_fitness : float
        The average fitness score in the current generation.
    """
    # Initialize fitness, inference time and IOU lists
    fitness_list = []
    tpu_time_list = []
    iou_list = []

    # Define directories for saving results
    result_dir = f"results_{time}"
    generation_dir = result_dir + f"/generation_{generation}"
    best_models_arrays_dir = generation_dir + "/best_model_arrays.pkl"
    fitness_list_dir = generation_dir + "/fitness_list.pkl"
    iou_list_dir = generation_dir + "/iou_list.pkl"
    tpu_time_list_dir = generation_dir + "/tpu_time_list.pkl"

    # Create directories if they don't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    # Iterate over each model architecture in the population array
    for i in range(population_array.shape[0]):
        # Create and train model
        model = create_model(population_array[i], num_classes=num_classes)
        model, history = train_model(train_ds, val_ds, model=model, epochs=epochs)
        # Evaluate model
        iou = np.max(history.history["val_meaniou"])

        # Convert model to TFLite and measure TPU inference time
        try:
            tflite_model, tflite_name = convert_to_tflite(
                keras_model=model, generation=generation, i=i, time=time
            )
            edgetpu_name = compile_edgetpu(tflite_name)
            tpu_time = inference_time_tpu(edgetpu_model_name=edgetpu_name)
        except:
            tpu_time = 9999

        # Calculate model fitness
        fitness = calculate_fitness(iou, tpu_time)
        iou_list.append(iou)
        fitness_list.append(fitness)
        tpu_time_list.append(tpu_time)

        # Save fitness, IOU, and TPU time information
        with open(fitness_list_dir, "wb") as f:
            pickle.dump(fitness_list, f)
        with open(iou_list_dir, "wb") as f:
            pickle.dump(iou_list, f)
        with open(tpu_time_list_dir, "wb") as f:
            pickle.dump(tpu_time_list, f)

        gc.collect()

    # Calculate max and average fitness
    max_fitness = np.max(fitness_list)
    average_fitness = np.average(fitness_list)

    # Select best models based on fitness
    best_models_indices = sorted(
        range(len(fitness_list)), key=lambda j: fitness_list[j], reverse=True
    )[:5]
    best_models_arrays = [population_array[k] for k in best_models_indices]

    # Save architectures of the best models
    with open(best_models_arrays_dir, "wb") as f:
        pickle.dump(best_models_arrays, f)

    return best_models_arrays, max_fitness, average_fitness


def crossover(parent_arrays):
    """Performs crossover operation on a list of parent arrays to generate a child
    array.

    Parameters:
    parent_arrays : list of np.ndarray
        A list of parent arrays.

    Returns:
    np.ndarray
        A child array that is a combination of the parent arrays.
    """

    # Generate a same-sized array filled with random integers between 0 and 4 (inclusive),
    # which will be used as indices to select elements from the parent arrays
    parent_indices = np.random.randint(0, 5, size=parent_arrays[0].shape)

    # Use the indices array to select elements from the parent arrays and form a new child array
    child_array = np.choose(parent_indices, parent_arrays)

    return child_array


def mutate(model_array, mutate_prob=0.05):
    """Performs mutation operation on a given model array.

    Parameters:
    model_array : np.ndarray
        The model array to be mutated.
    mutate_prob : float, optional
        The probability of mutation for each element in the array, defaults to 0.05.

    Returns:
    np.ndarray
        The mutated model array.
    """

    # Generate a same-sized array filled with random floats between 0 and 1 (inclusive)
    prob = np.random.uniform(size=(9, 18))

    # Perform mutation operation: if the randomly generated number for a position is less than mutation probability,
    # flip the bit at that position in the model array; else, keep the original bit
    mutated_array = np.where(
        prob < mutate_prob, np.logical_not(model_array), model_array
    )

    return mutated_array


def create_next_population(parent_arrays, population=20, num_classes=5):
    """Creates the next generation of model arrays by performing crossover and mutation
    operations.

    Parameters:
    parent_arrays : list of np.ndarray
        A list of parent arrays.
    population : int, optional
        The size of the population to be generated, defaults to 20.
    num_classes : int, optional
        The number of classes for the model, defaults to 5.

    Returns:
    np.ndarray
        The next generation of model arrays.
    """

    # Initialize the next generation with random integers between 0 and 1
    next_population_array = np.random.randint(0, 2, (population, 9, 18))

    # For each individual in the population
    for individual in range(population):
        # Perform crossover operation using parent arrays
        next_population_array[individual] = crossover(parent_arrays)
        # Perform mutation operation with a mutation probability of 0.03
        next_population_array[individual] = mutate(
            next_population_array[individual], mutate_prob=0.03
        )

    # For each individual in the population
    for individual in range(population):
        # Create a model using the individual's model array
        model = create_model(next_population_array[individual], num_classes=num_classes)
        # If the model has a problem
        while model_has_problem(model):
            # Delete the model
            del model
            # Perform crossover operation using parent arrays
            next_population_array[individual] = crossover(parent_arrays)
            # Perform mutation operation with a mutation probability of 0.03
            next_population_array[individual] = mutate(
                next_population_array[individual], mutate_prob=0.03
            )
            # Create a new model using the updated individual's model array
            model = create_model(
                next_population_array[individual], num_classes=num_classes
            )
        # Delete the model after checking
        del model

    # Return the next generation of model arrays
    return next_population_array


def start_evolution(
    train_ds,
    val_ds,
    test_ds,
    generations,
    population,
    num_classes,
    epochs,
    population_array=None,
    time=None,
):
    """This function starts the evolutionary process to generate the optimal model
    architecture.

    Parameters:
    ----------
    train_ds : Tensorflow Dataset
        The training dataset.
    val_ds : Tensorflow Dataset
        The validation dataset.
    test_ds : Tensorflow Dataset
        The testing dataset.
    generations : int
        The number of generations to run the evolution process.
    population : int
        The number of individuals (model architectures) in each generation.
    num_classes : int
        The number of classes in the dataset.
    epochs : int
        The number of epochs for which each model is trained.
    population_array : array, default=None
        The initial population. If None, the function creates the first population.
    time : str, default=None
        String to append to result directories for unique identification.

    Returns:
    -------
    population_array : array
        The final population array after all generations.
    max_fitness_history : list
        List of the maximum fitness value of each generation.
    average_fitness_history : list
        List of the average fitness value of each generation.
    best_models_arrays : list
        List of best model architectures in the final generation.
    """

    max_fitness_history = []
    average_fitness_history = []

    # If no initial population is given, create the first population
    if population_array is None:
        population_array = create_first_population(
            population=population, num_classes=num_classes
        )

    result_dir = f"results_{time}"

    # Create the result directory if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Run evolution for the given number of generations
    for generation in range(generations):
        best_models_arrays, max_fitness, average_fitness = select_models(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            time=time,
            population_array=population_array,
            generation=generation,
            epochs=epochs,
            num_classes=num_classes,
        )
        # Create the next population based on the best models from the current generation
        population_array = create_next_population(
            parent_arrays=best_models_arrays,
            population=population,
            num_classes=num_classes,
        )

        max_fitness_history.append(max_fitness)
        average_fitness_history.append(average_fitness)

        # Save the next population, max fitness history, average fitness history and best models arrays
        next_population_array_dir = result_dir + "/next_population_array.pkl"
        max_fitness_history_dir = result_dir + "/max_fitness_history.pkl"
        average_fitness_history_dir = result_dir + "/average_fitness_history.pkl"
        best_model_arrays_dir = result_dir + "/best_model_arrays.pkl"

        with open(next_population_array_dir, "wb") as f:
            pickle.dump(population_array, f)
        with open(max_fitness_history_dir, "wb") as f:
            pickle.dump(max_fitness_history, f)
        with open(average_fitness_history_dir, "wb") as f:
            pickle.dump(average_fitness_history, f)
        with open(best_model_arrays_dir, "wb") as f:
            pickle.dump(best_models_arrays, f)

    return (
        population_array,
        max_fitness_history,
        average_fitness_history,
        best_models_arrays,
    )
