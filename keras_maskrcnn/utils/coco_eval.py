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

from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import numpy as np
import json
import os
import cv2


def evaluate_coco(generator, model, threshold=0.05):
    # start collecting results
    results = []
    image_ids = []
    for index in range(generator.size()):
        image = generator.load_image(index)
        image_shape = image.shape
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        # run network
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes   = outputs[-4]
        scores  = outputs[-3]
        labels  = outputs[-2]
        masks   = outputs[-1]

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[..., 2] -= boxes[..., 0]
        boxes[..., 3] -= boxes[..., 1]

        # compute predicted labels and scores
        for box, score, label, mask in zip(boxes[0], scores[0], labels[0], masks[0]):
            # scores are sorted by the network
            if score < threshold:
                break

            b = box.astype(int)  # box (x, y, w, h) as one int vector

            mask = mask.astype(np.float32)
            mask = cv2.resize(mask[:, :, label], (b[2], b[3]))
            mask = (mask > 0.5).astype(np.uint8)  # binarize for encoding as RLE

            segmentation = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
            segmentation[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] = mask
            segmentation = mask_utils.encode(np.asfortranarray(segmentation))

            # append boxes for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_coco_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'segmentation': segmentation
            }

            # convert byte to str to write in json (in Python 3)
            if not isinstance(image_result['segmentation']['counts'], str):
                image_result['segmentation']['counts'] = image_result['segmentation']['counts'].decode()

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])

        # print progress
        print('{}/{}'.format(index, generator.size()), end='\r')

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_segm_results.json'.format(generator.set_name), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('{}_segm_results.json'.format(generator.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'segm')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
