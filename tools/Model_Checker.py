"""This function takes a model as an input and checks if the model has any MultiHeadAttention layers with an output
size greater than 1024. If it finds such a layer, it returns True, otherwise it returns False. It does this by
iterating through all the layers of the model, checking if the string 'multi_head_attention' is present in the string
representation of the layer. If it is, it gets the output shape of the layer and checks the size of the second
dimension. If it's greater than 1024, it returns True. Otherwise, it continues to check the next layer. If it doesn't
find any such layer, it returns False."""

from tools.TFLITE_Converter import convert_to_tflite
from tools.Compile_Edge_TPU import compile_edgetpu
from dataset.Dataset import test_ds, train_ds, val_ds

import os
import gc


def is_edge_tpu_compatible(model):
    try:
        # Convert the Keras model to a TFLite model
        _, tflite_path = convert_to_tflite(model)

        # Try to compile the TFLite model for the Edge TPU
        edgetpu_model_name = compile_edgetpu(tflite_path)

        # Check if the compilation was successful
        if os.path.exists(edgetpu_model_name):
            model_size = os.path.getsize(edgetpu_model_name)
            print(f"Model size: {model_size / 1024 / 1024} mb")
            if model_size > 8 * 1024 * 1024:
                compatible = False
            else:
                compatible = True
        else:
            compatible = False

        # Clean up the temporary files
        os.remove(tflite_path)
        if os.path.exists(edgetpu_model_name):
            os.remove(edgetpu_model_name)

        return compatible
    except Exception as e:
        print(f"Error during Edge TPU compatibility check: {e}")
        return False


def model_has_attention(model):
    contains_multi_head_attention = False
    for layer in model.layers:
        if 'multi_head_attention' in str(layer):
            contains_multi_head_attention = True
            break

    if contains_multi_head_attention:
        for layer in model.layers:
            if 'multi_head_attention' in str(layer):
                output_shape = layer.output.shape
                size = output_shape[1]
                if size > 400:
                    return False
        return True

    else:
        return False


def model_has_problem(model):
    if model_has_attention(model):
        if is_edge_tpu_compatible(model):
            gc.collect()
            return False
        else:
            gc.collect()
            return True
    else:
        gc.collect()
        return True
