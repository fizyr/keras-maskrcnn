import keras.backend
import keras.layers
import keras_retinanet.backend

from .. import backend


class RoiAlign(keras.layers.Layer):
    def __init__(self, crop_size=(14, 14), **kwargs):
        self.crop_size = crop_size

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
        image_shape = keras.backend.cast(inputs[0], keras.backend.floatx())
        boxes       = keras.backend.stop_gradient(inputs[1][0])
        scores      = keras.backend.stop_gradient(inputs[2][0])
        fpn         = [keras.backend.stop_gradient(i[0]) for i in inputs[3:]]

        # compute from which level to get features from
        target_levels = self.map_to_level(boxes)

        # process each pyramid independently
        rois           = []
        ordered_indices = []
        for i in range(len(fpn)):
            # select the boxes and classification from this pyramid level
            indices = keras_retinanet.backend.where(keras.backend.equal(target_levels, i))
            ordered_indices.append(indices)

            level_boxes = keras_retinanet.backend.gather_nd(boxes, indices)
            fpn_shape   = keras.backend.cast(keras.backend.shape(fpn[i]), dtype=keras.backend.floatx())

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

        # concatenate rois to one blob
        rois = keras.backend.concatenate(rois, axis=0)

        # reorder rois back to original order
        indices = keras.backend.concatenate(ordered_indices, axis=0)
        rois    = keras_retinanet.backend.scatter_nd(indices, rois, keras.backend.cast(keras.backend.shape(rois), 'int64'))

        return keras.backend.expand_dims(rois, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
        })

        return config
