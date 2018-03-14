import keras.backend
import keras.layers

import keras_retinanet.backend

from .. import backend


class MaskLoss(keras.layers.Layer):
    def __init__(self, iou_threshold=0.5, **kwargs):
        self.iou_threshold = iou_threshold

        super(MaskLoss, self).__init__(**kwargs)

    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    def overlap(self, a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], axis=1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], axis=1), b[:, 0])
        ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], axis=1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], axis=1), b[:, 1])

        iw = keras.backend.maximum(iw, 0)
        ih = keras.backend.maximum(ih, 0)

        ua = keras.backend.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = keras.backend.maximum(ua, keras.backend.epsilon())

        intersection = iw * ih

        return intersection / ua

    def call(self, inputs, **kwargs):
        detections, masks, annotations, masks_target = inputs

        # TODO: Fix batch_size > 1
        detections   = detections[0]
        masks        = masks[0]
        annotations  = annotations[0]
        masks_target = masks_target[0]

        # compute overlap of detections with annotations
        iou                  = self.overlap(detections, annotations)
        argmax_overlaps_inds = keras.backend.argmax(iou, axis=1)
        max_iou              = keras.backend.max(iou, axis=1)

        # filter those with IoU > 0.5
        indices              = keras_retinanet.backend.where(keras.backend.greater_equal(max_iou, self.iou_threshold))
        detections           = keras_retinanet.backend.gather_nd(detections, indices)
        masks                = keras_retinanet.backend.gather_nd(masks, indices)
        argmax_overlaps_inds = keras.backend.cast(keras_retinanet.backend.gather_nd(argmax_overlaps_inds, indices), 'int32')

        shape = keras.backend.shape(masks_target)[1:]
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        boxes = keras.backend.stack([
            y1 / keras.backend.cast(shape[0], dtype=keras.backend.floatx()),
            x1 / keras.backend.cast(shape[1], dtype=keras.backend.floatx()),
            y2 / keras.backend.cast(shape[0], dtype=keras.backend.floatx()),
            x2 / keras.backend.cast(shape[1], dtype=keras.backend.floatx()),
        ], axis=1)

        # crop and resize masks_target
        masks_target = keras.backend.expand_dims(masks_target, axis=3)
        masks_target = backend.crop_and_resize(
            masks_target,
            boxes,
            argmax_overlaps_inds,
            keras.backend.int_shape(masks)[1:3]
        )
        masks_target = masks_target[:, :, :, 0]

        # gather the predicted masks
        masks = backend.transpose(masks, (0, 3, 1, 2))
        argmax_overlaps_inds = keras.backend.stack([
            keras.backend.arange(keras.backend.shape(argmax_overlaps_inds)[0]),
            argmax_overlaps_inds
        ], axis=1)
        masks = keras_retinanet.backend.gather_nd(masks, argmax_overlaps_inds)

        # compute mask loss
        mask_loss = masks - masks_target
        mask_loss = keras.backend.abs(mask_loss)
        divisor = keras.backend.shape(masks)[0] * keras.backend.shape(masks)[1] * keras.backend.shape(masks[2])
        mask_loss = keras.backend.sum(mask_loss) / keras.backend.maximum(keras.backend.cast(divisor, keras.backend.floatx()), 1)

        return mask_loss

    def compute_output_shape(self, input_shape):
        return ()

    def get_config(self):
        config = super(MaskLoss, self).get_config()
        config.update({
            'iou_threshold': self.iou_threshold,
        })

        return config
