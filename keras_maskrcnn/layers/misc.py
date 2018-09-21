"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
import keras.layers
import numpy as np


class Shape(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return keras.backend.shape(inputs)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)


class ConcatenateBoxes(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        boxes, other = inputs

        boxes_shape = keras.backend.shape(boxes)
        other_shape = keras.backend.shape(other)
        other = keras.backend.reshape(other, (boxes_shape[0], boxes_shape[1], -1))

        return keras.backend.concatenate([boxes, other], axis=2)

    def compute_output_shape(self, input_shape):
        boxes_shape, other_shape = input_shape
        return boxes_shape[:2] + (np.prod([s for s in other_shape[2:]]) + 4,)
