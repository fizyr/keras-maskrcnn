import keras.backend
import keras.layers
import keras_retinanet.backend

from .. import backend


class RoiAlign(keras.layers.Layer):
    def __init__(self, top_k=1000, crop_size=(14, 14), **kwargs):
        self.crop_size = crop_size
        self.top_k = top_k

        super(RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, detections, canonical_size=224, canonical_level=1, min_level=0, max_level=4):
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = keras.backend.sqrt(w * h)

        levels = backend.floor(canonical_level + backend.log2(size / canonical_size + keras.backend.epsilon()))
        levels = keras.backend.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        # TODO: Support batch_size > 1
        detections = inputs[0][0]
        fpn        = [i[0] for i in inputs[1:]]

        # compute best scores for each detection
        classification = detections[:, 4:]
        scores         = keras.backend.max(classification, axis=1)

        # select the top k for mask ROI computation
        _, indices = backend.top_k(scores, k=self.top_k, sorted=False)
        detections = keras.backend.gather(detections, indices)

        # compute from which level to get features from
        target_levels = self.map_to_level(detections)

        # process each pyramid independently
        rois = []
        for i in range(len(fpn)):
            # select the detections from this pyramid
            indices = keras_retinanet.backend.where(keras.backend.equal(target_levels, i))
            level_detections = keras_retinanet.backend.gather_nd(detections, indices)

            # convert to expected format for crop_and_resize
            shape = keras.backend.shape(fpn[i])
            x1 = level_detections[:, 0]
            y1 = level_detections[:, 1]
            x2 = level_detections[:, 2]
            y2 = level_detections[:, 3]
            level_boxes = keras.backend.stack([
                y1 / keras.backend.cast(shape[0] - 1, dtype=keras.backend.floatx()),
                x1 / keras.backend.cast(shape[1] - 1, dtype=keras.backend.floatx()),
                y2 / keras.backend.cast(shape[0] - 1, dtype=keras.backend.floatx()),
                x2 / keras.backend.cast(shape[1] - 1, dtype=keras.backend.floatx()),
            ], axis=1)

            # append the rois to the list of rois
            rois.append(backend.crop_and_resize(
                keras.backend.expand_dims(fpn[i], axis=0),
                level_boxes,
                keras.backend.zeros((keras.backend.shape(level_boxes)[0],), dtype='int32'),
                self.crop_size
            ))

        # concatenate to one big blob
        rois = keras.backend.concatenate(rois, axis=0)
        return keras.backend.expand_dims(rois, axis=0)

    def compute_output_shape(self, input_shape):
       return (None, None, self.crop_size[0], self.crop_size[1], input_shape[1][-1])

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
            'top_k'     : self.top_k,
        })

        return config
