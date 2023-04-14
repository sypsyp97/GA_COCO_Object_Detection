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


def get_spatial_dimensions(layer):
    output_shape = layer.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]  # Use the first item in the list

    if len(output_shape) == 4:  # Check if the layer has spatial dimensions
        return output_shape[1:3]  # Return only the spatial

    return None  # Return None if no spatial dimensions


def find_layers_with_downsampling(model):
    layers_with_downsampling = []

    for i in range(len(model.layers) - 1):
        current_layer = model.layers[i]
        next_layer = model.layers[i + 1]

        current_shape = get_spatial_dimensions(current_layer)
        next_shape = get_spatial_dimensions(next_layer)

        if current_shape is None or next_shape is None:  # Skip layers without spatial dimensions
            continue

        if current_shape[0] > next_shape[0] and current_shape[1] > next_shape[1]:
            if "rescaling" not in current_layer.name:
                layers_with_downsampling.append(current_layer)

    return layers_with_downsampling


def create_base_model(model_array, input_shape=(160, 160, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_block(x, kernel_size=2, filters=64, strides=2)

    for i in range(9):
        x = decoded_block(x, model_array[i])

    model = keras.Model(inputs, x)

    return model


def create_model(model_array, input_shape=(160, 160, 3), num_classes=34):
    base_model = create_base_model(model_array, input_shape)

    # Find the layers with downsampling in the base model
    downsampling_layers = find_layers_with_downsampling(base_model)

    # Get output of the downsampling layers for skip connections
    skip_outputs = [layer.output for layer in downsampling_layers]

    # Encoder
    encoder_output = base_model.output

    # Decoder
    x = encoder_output
    for i in range(len(skip_outputs) - 1, -1, -1):
        skip_output = skip_outputs[i]
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        x = tf.keras.layers.Conv2D(skip_output.shape[-1], kernel_size=3, padding='same', activation=tf.nn.relu)(x)
        x = tf.image.resize(x, skip_output.shape[1:3])
        x = tf.keras.layers.Concatenate()([x, skip_output])
        x = tf.keras.layers.Conv2D(filters=skip_output.shape[-1], kernel_size=3, padding='same', activation=tf.nn.relu)(
            x)

    # Final upsampling layer and output
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=3, padding='same', activation=tf.nn.relu)(x)
    output = tf.keras.layers.Conv2D(num_classes, kernel_size=1, activation=tf.nn.softmax)(x)
    output = tf.image.resize(output, base_model.input.shape[1:3])

    # Create and compile the U-Net model
    unet = tf.keras.Model(inputs=base_model.input, outputs=output)

    return unet


def model_summary(model):
    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))


def train_model(train_ds, val_ds,
                model, epochs=30,
                checkpoint_filepath="checkpoints/checkpoint"):
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True)

    loss_fn = keras.losses.CategoricalCrossentropy()

    opt = tfa.optimizers.LazyAdam(learning_rate=0.004)
    opt = tfa.optimizers.MovingAverage(opt)
    opt = tfa.optimizers.Lookahead(opt)

    metrics = ["accuracy", tf.keras.metrics.MeanIoU(num_classes=34, name='mean_io_u')]

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=metrics)

    try:
        history = model.fit(train_ds,
                            epochs=epochs,
                            validation_data=val_ds,
                            callbacks=[checkpoint_callback])

        model.load_weights(checkpoint_filepath)
    except Exception as e:
        history = None
        print(e)
    return model, history
