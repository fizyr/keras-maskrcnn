import keras.backend
import keras.layers

class Shape(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return keras.backend.shape(inputs)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)
