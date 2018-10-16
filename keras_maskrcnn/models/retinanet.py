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

import keras.models
import keras_retinanet.layers
import keras_retinanet.models.retinanet

from ..layers.roi import RoiAlign
from ..layers.upsample import Upsample
from ..layers.misc import Shape, ConcatenateBoxes


def default_mask_model(
    num_classes,
    pyramid_feature_size=256,
    mask_feature_size=256,
    roi_size=(14, 14),
    mask_size=(28, 28),
    name='mask_submodel'
):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'activation'         : 'relu',
    }

    inputs  = keras.layers.Input(shape=(None, roi_size[0], roi_size[1], pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters=mask_feature_size,
            **options
        ), name='roi_mask_{}'.format(i))(outputs)

    # perform upsampling + conv instead of deconv as in the paper
    # https://distill.pub/2016/deconv-checkerboard/
    outputs = keras.layers.TimeDistributed(
        Upsample(mask_size),
        name='roi_mask_upsample')(outputs)
    outputs = keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=mask_feature_size,
        **options
    ), name='roi_mask_features')(outputs)

    outputs = keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='sigmoid'
    ), name='roi_mask')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_roi_submodels(num_classes):
    return [
        ('masks', default_mask_model(num_classes)),
    ]


def retinanet_mask(
    inputs,
    num_classes,
    retinanet_model=None,
    anchor_params=None,
    nms=True,
    class_specific_filter=True,
    name='retinanet-mask',
    roi_submodels=None,
    **kwargs
):
    """ Construct a RetinaNet mask model on top of a retinanet bbox model.

    This model uses the retinanet bbox model and appends a few layers to compute masks.

    # Arguments
        inputs          : List of keras.layers.Input. The first input is the image, the second input the blob of masks.
        num_classes     : Number of classes to classify.
        retinanet_model : keras_retinanet.models.retinanet model, returning regression and classification values.
        anchor_params   : Struct containing anchor parameters. If None, default values are used.
        name            : Name of the model.
        *kwargs         : Additional kwargs to pass to the retinanet bbox model.
    # Returns
        Model with inputs as input and as output the output of each submodel for each pyramid level and the detections.

        The order is as defined in submodels.
        ```
        [
            regression, classification, other[0], other[1], ..., boxes_masks, boxes, scores, labels, masks, other[0], other[1], ...
        ]
        ```
    """
    if anchor_params is None:
        anchor_params = keras_retinanet.utils.anchors.AnchorParameters.default

    if roi_submodels is None:
        roi_submodels = default_roi_submodels(num_classes)

    image = inputs
    image_shape = Shape()(image)

    if retinanet_model is None:
        retinanet_model = keras_retinanet.models.retinanet.retinanet(inputs=image, num_classes=num_classes, **kwargs)

    # parse outputs
    regression     = retinanet_model.outputs[0]
    classification = retinanet_model.outputs[1]
    other          = retinanet_model.outputs[2:]
    features       = [retinanet_model.get_layer(name).output for name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    # build boxes
    anchors = keras_retinanet.models.retinanet.__build_anchors(anchor_params, features)
    boxes = keras_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = keras_retinanet.layers.ClipBoxes(name='clipped_boxes')([image, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = keras_retinanet.layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        max_detections        = 100,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # split up in known outputs and "other"
    boxes  = detections[0]
    scores = detections[1]

    # get the region of interest features
    rois = RoiAlign()([image_shape, boxes, scores] + features)

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output]) for (name, _), output in zip(roi_submodels, maskrcnn_outputs)]

    # reconstruct the new output
    outputs = [regression, classification] + other + trainable_outputs + detections + maskrcnn_outputs

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
