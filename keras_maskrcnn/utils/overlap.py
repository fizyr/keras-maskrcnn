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

import numpy as np


def compute_overlap(a, b):
    """
    Args
        a: (N, H, W) ndarray of float
        b: (K, H, W) ndarray of float
    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    intersection = np.zeros((a.shape[0], b.shape[0]))
    union        = np.zeros((a.shape[0], b.shape[0]))
    for index, mask in enumerate(a):
        intersection[index, :] = np.sum(np.count_nonzero(b & mask, axis=1), axis=1)
        union[index, :]        = np.sum(np.count_nonzero(b + mask, axis=1), axis=1)

    return intersection / union
