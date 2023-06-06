import tensorflow as tf
import tensorflow_addons as tfa
from keras import layers
from tensorflow import keras

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block


def create_model(model_array, input_shape=(256, 256, 3), num_classes=4):
    """
    This function constructs a Keras model based on a given array encoding the structure of the model.

    Parameters:
    ----------
    model_array : list
        A list of integers representing the structure of the model.
    input_shape : tuple, optional
        The shape of the input tensor. Defaults to (256, 256, 3).
    num_classes : int, optional
        The number of classes for the classification task. Defaults to 4.

    Returns:
    -------
    model : keras.Model
        A compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=64, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    x = layers.GlobalAvgPool2D()(x)
    bounding_box = layers.Dense(num_classes)(x)

    model = keras.Model(inputs, bounding_box)

    return model


class meaniou(tf.keras.metrics.Metric):
    """
    This class is an implementation of a custom metric called mean intersection-over-union (mean IoU).

    Intersection-over-union (IoU) is a common evaluation metric for object detection tasks. This class calculates
    mean IoU for batch of predictions and ground truth bounding boxes.

    The metric is updated using `update_state` method, and `result` method computes mean IoU. `reset_states` resets
    the internal variables for the next computation.

    """

    def __init__(self, name="meaniou", **kwargs):
        super(meaniou, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Get the intersection coordinates of the bounding boxes
        top_left = tf.maximum(y_true[:, :2], y_pred[:, :2])
        bottom_right = tf.minimum(y_true[:, 2:], y_pred[:, 2:])

        # Compute the intersection area
        intersection_dims = tf.maximum(bottom_right - top_left, 0)
        intersection_area = intersection_dims[:, 0] * intersection_dims[:, 1]

        # Compute the area of the ground truth and predicted bounding boxes
        y_true_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
        y_pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

        # Compute the union area
        union_area = y_true_area + y_pred_area - intersection_area

        # Update the intersection and union sums
        self.intersection.assign_add(tf.reduce_sum(intersection_area))
        self.union.assign_add(tf.reduce_sum(union_area))

    def result(self):
        return self.intersection / (self.union + tf.keras.backend.epsilon())

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)


def model_summary(model):
    """
    This function prints the summary of the Keras model along with the total number of trainable weights.

    Parameters:
    ----------
    model : keras.Model
        The model for which the summary is to be printed.

    Returns:
    -------
    None
    """
    model.summary()
    print("Number of trainable weights = {}".format(len(model.trainable_weights)))


def train_model(
    train_ds, val_ds, model, epochs=100, checkpoint_filepath="checkpoints/checkpoint"
):
    """
    This function trains the Keras model on the given datasets and saves the weights of the best performing model
    using the validation dataset.

    Parameters:
    ----------
    train_ds : tensorflow.data.Dataset
        The training dataset.
    val_ds : tensorflow.data.Dataset
        The validation dataset.
    model : keras.Model
        The Keras model to train.
    epochs : int, optional
        The number of epochs to train the model. Defaults to 100.
    checkpoint_filepath : str, optional
        The file path to save the model weights. Defaults to "checkpoints/checkpoint".

    Returns:
    -------
    model : keras.Model
        The trained Keras model.
    history : History
        A History object. Its `history` attribute is a record of training loss values and metrics values at
        successive epochs, as well as validation loss values and validation metrics values.
    """
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_meaniou",
        save_best_only=True,
        save_weights_only=True,
    )

    loss_fn = keras.losses.MeanSquaredError()

    opt = tfa.optimizers.LazyAdam(learning_rate=0.004)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    metrics = [meaniou()]

    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[
                checkpoint_callback,
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
            ],
        )

        model.load_weights(checkpoint_filepath)
    except Exception as e:
        history = None
        print(e)
    return model, history
