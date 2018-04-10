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
import numpy as np
import cv2

import keras

from keras_retinanet.preprocessing.coco import CocoGenerator
from keras_retinanet.utils.anchors import bbox_transform
from keras_retinanet.utils.image import (
    adjust_transform_for_image,
    apply_transform,
)
from keras_retinanet.utils.transform import transform_aabb


class CocoGeneratorMask(CocoGenerator):
    def load_annotations(self, image_index):
        # get image info
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)

        # outputs
        annotations = np.zeros((0, 5))
        masks       = []

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            # we need an annotation to compute targets properly... make an impossible annotation
            annotations = np.array([[0, 0, 1, 1, 0]], dtype=keras.backend.floatx())
            masks = [np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)]
            return annotations, masks

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if 'segmentation' not in a:
                raise ValueError('Expected \'segmentation\' key in annotation, got: {}'.format(a))

            box = a['bbox']

            # some annotations have basically no width / height, skip them
            if box[2] < 1 or box[3] < 1:
                continue

            mask = np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)
            for seg in a['segmentation']:
                points = np.array(seg).reshape((len(seg) // 2, 2)).astype(int)

                # draw mask
                cv2.fillPoly(mask, [points.astype(int)], (1,))

            masks.append(mask.astype(float))

            # gather everything into one blob
            annotation        = np.zeros((1, 5))
            annotation[0, :4] = box
            annotation[0, -1] = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations, masks

    def random_transform_group_entry(self, image, annotations, masks):
        # randomly transform both image and annotations
        if self.transform_generator:
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

            for i in range(len(masks)):
                masks[i] = apply_transform(transform, masks[i], self.transform_parameters)
                masks[i] = np.expand_dims(masks[i], axis=2)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations, masks

    def preprocess_group_entry(self, image, annotations, masks):
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations, masks = self.random_transform_group_entry(image, annotations, masks)

        # resize image
        image, image_scale = self.resize_image(image)

        # resize masks
        for i in range(len(masks)):
            masks[i], _ = self.resize_image(masks[i])

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations, masks

    def preprocess_group(self, image_group, annotations_group, masks_group):
        for index, (image, annotations, masks) in enumerate(zip(image_group, annotations_group, masks_group)):
            # preprocess a single group entry
            image, annotations, masks = self.preprocess_group_entry(image, annotations, masks)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations
            masks_group[index]       = masks

        return image_group, annotations_group, masks_group

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group, masks_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], annotations, anchors = self.compute_anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)
            regression_group[index] = bbox_transform(anchors, annotations)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        # copy all annotations / masks to the batch
        max_annotations = max(a.shape[0] for a in annotations_group)
        masks_batch     = np.zeros((self.batch_size, max_annotations, 5 + 2 + max_shape[0] * max_shape[1]), dtype=keras.backend.floatx())
        for index, (annotations, masks) in enumerate(zip(annotations_group, masks_group)):
            masks_batch[index, :annotations.shape[0], :annotations.shape[1]] = annotations
            masks_batch[index, :, 5] = max_shape[1]  # width
            masks_batch[index, :, 6] = max_shape[0]  # height

            # add flattened mask
            for mask_index, mask in enumerate(masks):
                masks_batch[index, mask_index, 7:] = mask.flatten()

        return [regression_batch, labels_batch, masks_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # split annotations and masks
        masks_group       = [m for _, m in annotations_group]
        annotations_group = [a for a, _ in annotations_group]

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group, masks_group = self.preprocess_group(image_group, annotations_group, masks_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group, masks_group)

        return inputs, targets
