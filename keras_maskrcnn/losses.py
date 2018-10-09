import keras.backend
import keras_retinanet.backend
from . import backend


def mask(iou_threshold=0.5, mask_size=(28, 28)):
    def _mask_conditional(y_true, y_pred):
        # if there are no masks annotations, return 0; else, compute the masks loss
        loss = backend.cond(
            keras.backend.any(keras.backend.equal(keras.backend.shape(y_true), 0)),
            lambda: 0.0,
            lambda: _mask(y_true, y_pred, iou_threshold=iou_threshold, mask_size=mask_size)
        )
        return loss

    def _mask(y_true, y_pred, iou_threshold=0.5, mask_size=(28, 28)):
        # split up the different predicted blobs
        boxes = y_pred[:, :, :4]
        masks = y_pred[:, :, 4:]

        # split up the different blobs
        annotations  = y_true[:, :, :5]
        width        = keras.backend.cast(y_true[0, 0, 5], dtype='int32')
        height       = keras.backend.cast(y_true[0, 0, 6], dtype='int32')
        masks_target = y_true[:, :, 7:]

        # reshape the masks back to their original size
        masks_target = keras.backend.reshape(masks_target, (keras.backend.shape(masks_target)[0], keras.backend.shape(masks_target)[1], height, width))
        masks        = keras.backend.reshape(masks, (keras.backend.shape(masks)[0], keras.backend.shape(masks)[1], mask_size[0], mask_size[1], -1))

        # TODO: Fix batch_size > 1
        boxes        = boxes[0]
        masks        = masks[0]
        annotations  = annotations[0]
        masks_target = masks_target[0]

        # compute overlap of boxes with annotations
        iou                  = backend.overlap(boxes, annotations)
        argmax_overlaps_inds = keras.backend.argmax(iou, axis=1)
        max_iou              = keras.backend.max(iou, axis=1)

        # filter those with IoU > 0.5
        indices              = keras_retinanet.backend.where(keras.backend.greater_equal(max_iou, iou_threshold))
        boxes                = keras_retinanet.backend.gather_nd(boxes, indices)
        masks                = keras_retinanet.backend.gather_nd(masks, indices)
        argmax_overlaps_inds = keras.backend.cast(keras_retinanet.backend.gather_nd(argmax_overlaps_inds, indices), 'int32')
        labels               = keras.backend.cast(keras.backend.gather(annotations[:, 4], argmax_overlaps_inds), 'int32')

        # make normalized boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        boxes = keras.backend.stack([
            y1 / (keras.backend.cast(height, dtype=keras.backend.floatx()) - 1),
            x1 / (keras.backend.cast(width, dtype=keras.backend.floatx()) - 1),
            (y2 - 1) / (keras.backend.cast(height, dtype=keras.backend.floatx()) - 1),
            (x2 - 1) / (keras.backend.cast(width, dtype=keras.backend.floatx()) - 1),
        ], axis=1)

        # crop and resize masks_target
        masks_target = keras.backend.expand_dims(masks_target, axis=3)  # append a fake channel dimension
        masks_target = backend.crop_and_resize(
            masks_target,
            boxes,
            argmax_overlaps_inds,
            mask_size
        )
        masks_target = masks_target[:, :, :, 0]  # remove fake channel dimension

        # gather the predicted masks using the annotation label
        masks = backend.transpose(masks, (0, 3, 1, 2))
        label_indices = keras.backend.stack([
            keras.backend.arange(keras.backend.shape(labels)[0]),
            labels
        ], axis=1)
        masks = keras_retinanet.backend.gather_nd(masks, label_indices)

        # compute mask loss
        mask_loss  = keras.backend.binary_crossentropy(masks_target, masks)
        normalizer = keras.backend.shape(masks)[0] * keras.backend.shape(masks)[1] * keras.backend.shape(masks)[2]
        normalizer = keras.backend.maximum(keras.backend.cast(normalizer, keras.backend.floatx()), 1)
        mask_loss  = keras.backend.sum(mask_loss) / normalizer

        return mask_loss

    return _mask_conditional
