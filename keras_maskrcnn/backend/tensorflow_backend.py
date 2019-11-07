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

import tensorflow


def crop_and_resize(image, boxes, box_indices, crop_size, **kwargs):
    return tensorflow.cast(
        tensorflow.image.crop_and_resize(
            image=image,
            boxes=tensorflow.cast(boxes, tensorflow.float32),
            box_indices=box_indices,
            crop_size=crop_size,
            **kwargs
        ),
        image.dtype
    )


def floor(*args, **kwargs):
    return tensorflow.floor(*args, **kwargs)


def split(*args, **kwargs):
    return tensorflow.split(*args, **kwargs)


def transpose(*args, **kwargs):
    return tensorflow.transpose(*args, **kwargs)


def cond(*args, **kwargs):
    return tensorflow.cond(*args, **kwargs)
