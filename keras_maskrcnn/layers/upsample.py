
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

import keras
import keras_retinanet.backend

import numpy as np


class Upsample(keras.layers.Layer):
    def __init__(self, target_size, *args, **kwargs):
        self.target_size = target_size
        super(Upsample, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return keras_retinanet.backend.resize_images(inputs, (self.target_size[0], self.target_size[1]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(self.target_size) + (input_shape[-1],)

    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({
            'target_size': self.target_size,
        })

        return config
