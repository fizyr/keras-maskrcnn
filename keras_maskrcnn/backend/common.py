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

import tensorflow.keras.backend


def log2(x):
    return tensorflow.keras.backend.log(x) / tensorflow.keras.backend.log(tensorflow.keras.backend.cast(2.0, x.dtype))


def overlap(a, b):
    """ Computes the IoU overlap of boxes in a and b.

    Args
        a: np.array of shape (N, 4) of boxes.
        b: np.array of shape (K, 4) of boxes.

    Returns
        A np.array of shape (N, K) of overlap between boxes from a and b.
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = tensorflow.keras.backend.minimum(tensorflow.keras.backend.expand_dims(a[:, 2], axis=1), b[:, 2]) - tensorflow.keras.backend.maximum(tensorflow.keras.backend.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = tensorflow.keras.backend.minimum(tensorflow.keras.backend.expand_dims(a[:, 3], axis=1), b[:, 3]) - tensorflow.keras.backend.maximum(tensorflow.keras.backend.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = tensorflow.keras.backend.maximum(iw, 0)
    ih = tensorflow.keras.backend.maximum(ih, 0)

    ua = tensorflow.keras.backend.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = tensorflow.keras.backend.maximum(ua, tensorflow.keras.backend.epsilon())

    intersection = iw * ih

    return intersection / ua
