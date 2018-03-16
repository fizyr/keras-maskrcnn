import keras.models
from keras_retinanet.models import retinanet
from keras_retinanet.layers import NonMaximumSuppression

from ..layers.roi import RoiAlign
from ..layers.upsample import Upsample
from ..layers.misc import Shape, ConcatenateBoxesMasks
from .. import losses

custom_objects = retinanet.custom_objects
custom_objects.update({
    'RoiAlign'              : RoiAlign,
    'Upsample'              : Upsample,
    'Shape'                 : Shape,
    'ConcatenateBoxesMasks' : ConcatenateBoxesMasks,
    'mask_'                 : losses.mask(),
})


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
            **options,
        ), name='roi_mask_{}'.format(i))(outputs)

    # perform upsampling + conv instead of deconv as in the paper
    # https://distill.pub/2016/deconv-checkerboard/
    outputs = keras.layers.TimeDistributed(
        Upsample(mask_size),
        name='roi_mask_upsample')(outputs)
    outputs = keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=mask_feature_size,
        **options,
    ), name='roi_mask_features')(outputs)

    outputs = keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='sigmoid',
    ), name='roi_mask')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_roi_submodels(num_classes):
    return [
        ('mask', default_mask_model(num_classes)),
    ]


def retinanet_mask(
    inputs,
    num_classes,
    nms=True,
    name='retinanet-mask',
    roi_submodels=None,
    **kwargs
):
    """ Construct a RetinaNet mask model on top of a retinanet bbox model.

    This model uses the retinanet bbox model and appends a few layers to compute masks.

    # Arguments
        inputs      : List of keras.layers.Input. The first input is the image, the second input the blob of masks.
        num_classes : Number of classes to classify.
        name        : Name of the model.
        *kwargs     : Additional kwargs to pass to the retinanet bbox model.
    # Returns
        Model with inputs as input and as output the output of each submodel for each pyramid level and the detections.

        The order is as defined in submodels. Using default values the output is:
        ```
        [
            regression, classification, boxes_masks, boxes, masks
        ]
        ```
    """
    if roi_submodels is None:
        roi_submodels = default_roi_submodels(num_classes)

    image = inputs
    image_shape = Shape()(image)

    bbox_model = retinanet.retinanet_bbox(inputs=image, num_classes=num_classes, nms=False, **kwargs)

    # parse outputs
    regression     = bbox_model.outputs[0]
    classification = bbox_model.outputs[1]
    other          = bbox_model.outputs[2:-1]
    boxes          = bbox_model.outputs[-1]
    fpn            = [bbox_model.get_layer(name).output for name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    # get the region of interest features
    top_boxes, top_classification, rois = RoiAlign()([image_shape, boxes, classification] + fpn)

    # optionally apply non maximum suppression
    if nms:
        top_classification = NonMaximumSuppression(name='nms')([top_boxes, top_classification])

    # estimate masks
    # TODO: Change this so that it iterates over roi_submodels
    masks = roi_submodels[0][1](rois)

    # concatenate boxes and masks together
    boxes_masks = ConcatenateBoxesMasks(name='boxes_masks')([top_boxes, masks])

    # reconstruct the new output
    outputs = [regression, classification] + other + [boxes_masks, top_boxes, top_classification, masks]

    # construct the model
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
