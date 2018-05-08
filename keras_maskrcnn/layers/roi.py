import keras.backend
import keras.layers
import keras_retinanet.backend

from .. import backend


class RoiAlign(keras.layers.Layer):
    def __init__(self, top_k=500, crop_size=(14, 14), **kwargs):
        self.crop_size = crop_size
        self.top_k = top_k

        super(RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, boxes, canonical_size=224, canonical_level=1, min_level=0, max_level=4):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = keras.backend.sqrt(w * h)

        levels = backend.floor(canonical_level + backend.log2(size / canonical_size + keras.backend.epsilon()))
        levels = keras.backend.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        # TODO: Support batch_size > 1
        image_shape    = keras.backend.cast(inputs[0], keras.backend.floatx())
        boxes          = keras.backend.stop_gradient(inputs[1][0])
        classification = keras.backend.stop_gradient(inputs[2][0])
        fpn            = [keras.backend.stop_gradient(i[0]) for i in inputs[3:]]

        # compute best scores for each detection
        scores = keras.backend.max(classification, axis=1)

        # select the top k for mask ROI computation
        _, indices     = keras_retinanet.backend.top_k(scores, k=keras.backend.minimum(self.top_k, keras.backend.shape(boxes)[0]), sorted=False)
        boxes          = keras.backend.gather(boxes, indices)
        classification = keras.backend.gather(classification, indices)

        # compute from which level to get features from
        target_levels = self.map_to_level(boxes)

        # process each pyramid independently
        rois                   = []
        ordered_boxes          = []
        ordered_classification = []
        for i in range(len(fpn)):
            # select the boxes and classification from this pyramid level
            indices = keras_retinanet.backend.where(keras.backend.equal(target_levels, i))

            level_boxes          = keras_retinanet.backend.gather_nd(boxes, indices)
            level_classification = keras_retinanet.backend.gather_nd(classification, indices)

            ordered_boxes.append(level_boxes)
            ordered_classification.append(level_classification)

            fpn_shape = keras.backend.cast(keras.backend.shape(fpn[i]), dtype=keras.backend.floatx())

            # convert to expected format for crop_and_resize
            x1 = level_boxes[:, 0]
            y1 = level_boxes[:, 1]
            x2 = level_boxes[:, 2]
            y2 = level_boxes[:, 3]
            level_boxes = keras.backend.stack([
                (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1),
            ], axis=1)

            # append the rois to the list of rois
            rois.append(backend.crop_and_resize(
                keras.backend.expand_dims(fpn[i], axis=0),
                level_boxes,
                keras.backend.zeros((keras.backend.shape(level_boxes)[0],), dtype='int32'),
                self.crop_size
            ))

        # reassemble the boxes in a different order
        boxes          = keras.backend.concatenate(ordered_boxes, axis=0)
        classification = keras.backend.concatenate(ordered_classification, axis=0)

        # concatenate rois to one blob
        rois = keras.backend.concatenate(rois, axis=0)
        return [keras.backend.expand_dims(boxes, axis=0), keras.backend.expand_dims(classification, axis=0), keras.backend.expand_dims(rois, axis=0)]

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[1][0], None, input_shape[1][2]),
            (input_shape[2][0], None, input_shape[2][2]),
            (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])
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
