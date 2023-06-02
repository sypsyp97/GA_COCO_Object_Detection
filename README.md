# GA_Object_Detection

![GitHub Code License](https://img.shields.io/github/license/sypsyp97/Genetic_NAS)
![GitHub last commit](https://img.shields.io/github/last-commit/sypsyp97/Genetic_NAS)
![GitHub pull request](https://img.shields.io/badge/PRs-not_welcome-red)


Welcome to the GA_Object_Detection repository, the home of Yipeng Sun's Master's Thesis project. This project explores the use of genetic algorithms to search for adaptable models specifically designed for Edge TPU. We aim to leverage the capabilities of the Edge TPU to enhance inference speed while maintaining a high level of accuracy in obejct detection tasks.

## Overview

The project's objective is to explore the potential of genetic algorithms for creating models for object detection tasks that are adaptable for the Edge TPU. These models aim to leverage the capabilities of the Edge TPU to enhance inference speed while maintaining a high level of IoU.


## Prerequisites

To get the most out of this project, you should have:

- Familiarity with Python 3.9 and above
- Basic understanding of neural networks
- Some knowledge about genetic algorithms

## Environment and Installation

This project is developed in Python 3.9 environment with TensorFlow 2.11 being the major library used. To set up the environment, follow these steps:

1. Clone the repository to your local machine：
 ```bash
# Clone the repository
git clone https://github.com/sypsyp97/Genetic_NAS.git
cd Genetic_NAS
```
2. Set up a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) environment and install the required packages:
```bash
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt

```

## Repository Structure

The repository is structured as follows:

- `src/`: This directory contains the main source code and utility scripts that aid in various tasks for the project.
- `get_datasets/`: This directory includes scripts for data acquisition.
- `example.py`: A Python script that is used for testing the application with the Caltech101 dataset.
- `requirements.txt`: Specifies the libraries and their respective versions required for this project.



## Usage Example
Before running the NAS process, please ensure that you have an Edge TPU device available. You will also need to install the necessary libraries and dependencies for working with the Edge TPU. Instructions for setting up the Edge TPU can be found in the [Coral documentation](https://coral.ai/docs/accelerator/get-started/).

Here's an example of how you can use the `start_evolution` function to initiate the process of NAS:

```python
from src.Evolutionary_Algorithm import start_evolution

population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        generations=4,
        population=20,
        num_classes=5,
        epochs=30,
        time=formatted_date
    )
```

You can also easily start the NAS process by running the `example.py` script. This script has been designed to make running the NAS process simpler by predefining certain parameters and steps.
```bash
python example.py
```

## Additional Note

For detailed documentation of functions, please refer to our [Genetic_NAS project](https://github.com/sypsyp97/Genetic_NAS.git).

## License

This project is licensed under the terms of the [MIT License](LICENSE). 

## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{Genetic_NAS,
  title = {Genetic Neural Architecture Search for Edge TPU},
  author = {Yipeng Sun},
  howpublished = {\url{https://github.com/sypsyp97/Genetic_NAS.git}},
  year = {2023}
}
```
