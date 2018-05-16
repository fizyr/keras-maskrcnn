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


def log2(x):
    return keras.backend.log(x) / keras.backend.log(2.0)


def overlap(a, b):
    """ Computes the IoU overlap of boxes in a and b.

    Args
        a: np.array of shape (N, 4) of boxes.
        b: np.array of shape (K, 4) of boxes.

    Returns
        A np.array of shape (N, K) of overlap between boxes from a and b.
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], axis=1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], axis=1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = keras.backend.maximum(iw, 0)
    ih = keras.backend.maximum(ih, 0)

    ua = keras.backend.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = keras.backend.maximum(ua, keras.backend.epsilon())

    intersection = iw * ih

    return intersection / ua
