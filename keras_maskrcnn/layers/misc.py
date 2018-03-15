import keras.backend
import keras.layers


class Shape(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return keras.backend.shape(inputs)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)


class ConcatenateBoxesMasks(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        detections, masks = inputs
        boxes = detections[:, :, :4]

        boxes_shape = keras.backend.shape(boxes)
        masks_shape = keras.backend.shape(masks)
        masks = keras.backend.reshape(masks, (masks_shape[0], boxes_shape[1], -1))

        return keras.backend.concatenate([boxes, masks], axis=2)

    def compute_output_shape(self, input_shape):
        detections_shape, masks_shape = input_shape
        return masks_shape[:2] + (masks_shape[2] * masks_shape[3],)
