import keras.models
from keras_retinanet.models import retinanet

from ..layers.roi import RoiAlign
from ..layers.upsample import Upsample
from ..layers.mask_loss import MaskLoss
from ..layers.misc import Shape

custom_objects = retinanet.custom_objects
custom_objects.update({
    'RoiAlign' : RoiAlign,
    'Upsample' : Upsample,
    'MaskLoss' : MaskLoss,
    'Shape'    : Shape,
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
            regression_P3, regression_P4, regression_P5, regression_P6, regression_P7,
            classification_P3, classification_P4, classification_P5, classification_P6, classification_P7,
            boxes_P3, boxes_P4, boxes_P5, boxes_P6, boxes_P7,
            detections
        ]
        ```
    """
    if roi_submodels is None:
        roi_submodels = default_roi_submodels(num_classes)

    image, annotations, gt_masks = inputs
    image_shape = Shape()(image)

    bbox_model = retinanet.retinanet_bbox(inputs=image, num_classes=num_classes, output_fpn=True, nms=False, **kwargs)

    # parse outputs
    regression     = bbox_model.outputs[:5]
    classification = bbox_model.outputs[5:10]
    other          = bbox_model.outputs[10:-6]
    fpn            = bbox_model.outputs[-6:-1]
    detections     = bbox_model.outputs[-1]

    # get the region of interest features
    detections, rois = RoiAlign()([image_shape, detections] + fpn)

    # estimate masks
    masks = roi_submodels[0][1](rois)

    # compute mask loss
    mask_loss = MaskLoss(name='mask_loss')([detections, masks, annotations, gt_masks])

    # reconstruct the new output
    outputs = regression + classification + other + [mask_loss, detections, masks]

    # construct the model
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
