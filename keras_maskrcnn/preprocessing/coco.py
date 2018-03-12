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
            annotations = np.array([[0, 0, 1, 1, 0]], dtype=keras.backend.floatx())
            mask = [np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)]
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

    def compute_inputs(self, image_group, annotations_group, masks_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        # construct the masks batch
        max_masks = max(len(m) for m in masks_group)
        masks_batch = -1 * np.ones((self.batch_size, max_masks) + max_shape[:-1], dtype=keras.backend.floatx())
        for batch_index, masks in enumerate(masks_group):
            if len(masks) == 0:
                continue

            masks = np.stack(masks)
            masks_batch[batch_index, :masks.shape[0], :masks.shape[1], :masks.shape[2]] = masks

        # construct the annotations batch
        max_annotations = max(a.shape[0] for a in annotations_group)
        assert(max_masks == max_annotations)
        annotations_batch = -1 * np.ones((self.batch_size, max_annotations, 5), dtype=keras.backend.floatx())
        for batch_index, annotations in enumerate(annotations_group):
            annotations_batch[batch_index, :annotations.shape[0], :] = annotations

        return [image_batch, annotations_batch, masks_batch]

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group     = [[]] * self.batch_size
        regression_group = [[]] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], regression_targets, anchors = self.anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)
            num_positive = max(np.sum([np.sum(labels == 1) for labels in labels_group[index]]), 1)

            # append anchor states and normalization value
            for a, at, lg in zip(anchors, regression_targets, labels_group[index]):
                regression_group[index].append(bbox_transform(a, at))

                # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
                # append normalization value (1 / num_positive_anchors)
                regression_group[index][-1] = np.concatenate([
                    regression_group[index][-1],
                    np.max(lg, axis=1, keepdims=True),  # anchor states
                    np.ones((regression_group[index][-1].shape[0], 1)) * num_positive,  # normalization value
                ], axis=1)

            # append normalization value
            for i in range(len(labels_group[index])):
                labels_group[index][i] = np.concatenate([
                    labels_group[index][i],
                    np.ones((labels_group[index][i].shape[0], 1)) * num_positive,  # normalization value
                ], axis=1)

        # use labels and regression to construct batches for P3...P7
        labels_batches     = [np.zeros((self.batch_size,) + lg.shape, dtype=keras.backend.floatx()) for lg in labels_group[0]]
        regression_batches = [np.zeros((self.batch_size,) + r.shape, dtype=keras.backend.floatx()) for r in regression_group[0]]

        # loop over images
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            # loop over P3...P7 for one image
            for i in range(len(labels)):
                # copy data to corresponding batch
                labels_batches[i][index, ...]     = labels[i]
                regression_batches[i][index, ...] = regression[i]

        return regression_batches + labels_batches + [np.zeros((1,))]  # the last target is for mask loss

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
        inputs = self.compute_inputs(image_group, annotations_group, masks_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets
