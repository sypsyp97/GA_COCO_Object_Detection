import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
import scipy.io

# Paths to images and corresponding annotations
path_images = "101_ObjectCategories/airplanes/"
path_annot = "Annotations/Airplanes_Side_2/"

# Get list of paths to images and annotations
image_paths = [f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))]
annot_paths = [f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]

image_paths.sort()
annot_paths.sort()

image_size = 256  # Target size for input images

images, targets = [], []

# Iterate over annotations and corresponding images
for i in range(0, len(annot_paths)):
    # Load bounding box coordinates
    annot = scipy.io.loadmat(path_annot + annot_paths[i])["box_coord"][0]
    top_left_x, top_left_y = annot[2], annot[0]
    bottom_right_x, bottom_right_y = annot[3], annot[1]

    # Load, resize, and convert image to array
    image = keras.utils.load_img(path_images + image_paths[i])
    (w, h) = image.size[:2]
    image = image.resize((image_size, image_size))
    images.append(keras.utils.img_to_array(image))

    # Scale bounding box coordinates relative to image size and append to targets
    targets.append(
        (
            float(top_left_x) / w,
            float(top_left_y) / h,
            float(bottom_right_x) / w,
            float(bottom_right_y) / h,
        )
    )

# Convert list to numpy arrays and split data into training, validation, and testing sets
x_train = np.asarray(images[: int(len(images) * 0.8)])
y_train = np.asarray(targets[: int(len(targets) * 0.8)])
x_val = np.asarray(images[int(len(images) * 0.8) : int(len(images) * 0.9)])
y_val = np.asarray(targets[int(len(targets) * 0.8) : int(len(targets) * 0.9)])
x_test = np.asarray(images[int(len(images) * 0.9) :])
y_test = np.asarray(targets[int(len(targets) * 0.9) :])

# Create tensorflow datasets for training, validation, and testing
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
