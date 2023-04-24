from keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf

from src.Decode_Block import decoded_block
from src.Gene_Pool import conv_block

# if tf.config.list_physical_devices('GPU'):
#     strategy = tf.distribute.MirroredStrategy()
# else:  # Use the Default Strategy
#     strategy = tf.distribute.get_strategy()

'''This function takes in 3 inputs, model_array, num_classes and input_shape. The function creates a keras model by defining the layers in it.

It starts by creating an input layer with the shape specified by the input_shape variable. Then it applies a 
rescaling layer with a scale factor of 1/255 to the input. It then applies a convolutional block with a kernel size 
of 2, 16 filters and a stride of 2 to the input.

It then enters a for loop that iterates 9 times. On each iteration, it applies a decoded_block function to the 
current output, passing in the current element of the model_array.

After the for loop, it applies another convolutional block with 320 filters, a kernel size of 1 and a stride of 1 to 
the output. It then applies a global average pooling layer and a dropout layer with a rate of 0.5. Finally, 
it adds a dense layer with num_classes number of units and returns the model.'''


def create_model(model_array, input_shape=(256, 256, 3), num_classes=4):
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
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


def train_model(train_ds, val_ds,
                model, epochs=100,
                checkpoint_filepath="checkpoints/checkpoint"):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_meaniou",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    loss_fn = keras.losses.MeanSquaredError()

    opt = tfa.optimizers.LazyAdam(learning_rate=0.004)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    metrics=[meaniou()]

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)

    try:
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback,
                                       keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

        model.load_weights(checkpoint_filepath)
    except Exception as e:
        history = None
        print(e)
    return model, history
