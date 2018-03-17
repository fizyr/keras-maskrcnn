import keras.backend
import keras.layers
import keras_retinanet.backend

from .. import backend


class RoiAlign(keras.layers.Layer):
    def __init__(self, top_k=500, crop_size=(14, 14), **kwargs):
        self.crop_size = crop_size
        self.top_k = top_k

        super(RoiAlign, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        image_shape    = inputs[0]

        # TODO: Support batch_size > 1
        boxes          = inputs[1][0]
        classification = inputs[2][0]
        fpn            = [i[0] for i in inputs[3:]]

        # compute best scores for each detection
        scores = keras.backend.max(classification, axis=1)

        # select the top k for mask ROI computation
        _, indices     = keras_retinanet.backend.top_k(scores, k=self.top_k, sorted=False)
        boxes          = keras.backend.gather(boxes, indices)
        classification = keras.backend.gather(classification, indices)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        normed_boxes = keras.backend.stack([
            y1 / keras.backend.cast(image_shape[1], dtype=keras.backend.floatx()),
            x1 / keras.backend.cast(image_shape[2], dtype=keras.backend.floatx()),
            y2 / keras.backend.cast(image_shape[1], dtype=keras.backend.floatx()),
            x2 / keras.backend.cast(image_shape[2], dtype=keras.backend.floatx()),
        ], axis=1)

        # process each pyramid independently
        rois = None
        for f in fpn:
            # compute rois for this feature level
            level_rois = backend.crop_and_resize(
                keras.backend.expand_dims(f, axis=0),
                normed_boxes,
                keras.backend.zeros((keras.backend.shape(normed_boxes)[0],), dtype='int32'),
                self.crop_size
            )

            # concatenate the rois on the channel axis
            if rois is None:
                rois = level_rois
            else:
                rois = keras.backend.concatenate([rois, level_rois], axis=-1)

        return [keras.backend.expand_dims(boxes, axis=0), keras.backend.expand_dims(classification, axis=0), keras.backend.expand_dims(rois, axis=0)]

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[1][0], None, input_shape[1][2]),
            (input_shape[2][0], None, input_shape[2][2]),
            (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], sum(i[-1] for i in input_shape[3:]))
        ]

    def compute_mask(self, inputs, mask=None):
        return 3 * [None]

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
            'top_k'     : self.top_k,
        })

        return config
