#!/usr/bin/env python

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

import argparse
import os
import sys
import cv2
import numpy as np

from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.visualization import draw_annotations, draw_boxes, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.config import read_config_file
from keras_retinanet.utils.anchors import anchors_for_shape, compute_gt_annotations

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_maskrcnn.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..utils.visualization import draw_mask


def create_generator(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        # min_rotation=-0.1,
        # max_rotation=0.1,
        # min_translation=(-0.1, -0.1),
        # max_translation=(0.1, 0.1),
        # min_shear=-0.1,
        # max_shear=0.1,
        # min_scaling=(0.9, 0.9),
        # max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        # flip_y_chance=0.5,
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator,
            config=args.config
        )
    elif args.dataset_type == 'csv':
        from ..preprocessing.csv_generator import CSVGenerator

        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet-MaskRCNN network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to a CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--masks', help='Show annotated masks on the image.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)


def run(generator, args, anchor_params):
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)

        # apply random transformations
        if args.random_transform:
            image, annotations = generator.random_transform_group_entry(image, annotations)

        # resize the image and annotations
        if args.resize:
            image, image_scale  = generator.resize_image(image)
            annotations['bboxes'] *= image_scale
            for m in range(len(annotations['masks'])):
                annotations['masks'][m], _ = generator.resize_image(annotations['masks'][m])

        anchors = anchors_for_shape(image.shape, anchor_params=anchor_params)
        positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

        # draw anchors on the image
        if args.anchors:
            anchors = generator.generate_anchors(image.shape)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])
            draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

        # draw annotations on the image
        if args.annotations:
            # draw annotations in red
            draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

            # draw regressed anchors in green to override most red annotations
            # result is that annotations without anchors are red, with anchors are green
            draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

        # Draw masks over the image with random colours
        if args.masks:
            for m in range(len(annotations['masks'])):
                # crop the mask with the related bbox size, and then draw them
                box = annotations['bboxes'][m].astype(int)
                mask = annotations['masks'][m][box[1]:box[3], box[0]:box[2]]
                draw_mask(image, box, mask, annotations['labels'][m].astype(int))
                # add the label caption
                caption = '{}'.format(generator.label_to_name(annotations['labels'][m]))
                draw_caption(image, box, caption)

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    if args.loop:
        while run(generator, args, anchor_params):
            pass
    else:
        run(generator, args, anchor_params)

if __name__ == '__main__':
    main()
