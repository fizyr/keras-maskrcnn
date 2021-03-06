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

from keras_retinanet.utils.config import read_config_file
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_maskrcnn.bin
    __package__ = "keras_maskrcnn.bin"

# Change these to absolute imports if you copy this script outside the keras_maskrcnn package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.eval import evaluate


def create_generator(args):
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a MaskRCNN network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',                help='Path to MaskRCNN model.')
    parser.add_argument('--backbone',           help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',                help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',    help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',      help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',     help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--binarize-threshold', help='Threshold used for binarizing masks (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--save-path',          help='Path for saving images with detections.')
    parser.add_argument('--image-min-side',     help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',     help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',             help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average',   help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        average_precisions = evaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            binarize_threshold=args.binarize_threshold,
            save_path=args.save_path
        )

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if args.weighted_average:
            print('mAP: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        else:
            print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

if __name__ == '__main__':
    main()
