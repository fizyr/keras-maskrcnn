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

from keras_maskrcnn.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.transform import random_transform_generator
import numpy as np
import cv2
import random

r = lambda: random.randint(0,255)

transform_generator = random_transform_generator(flip_x_chance=0.5)

generator = CSVGenerator(
		'/srv/datasets/postnl/train.txt',
		'/srv/datasets/postnl/classes.txt',
		base_dir='/srv/datasets/postnl',
		transform_generator=transform_generator
		)

for index in range(generator.size()):

	image    = generator.load_image(index)
	image, _ = generator.resize_image(image)

	_, masks   = generator.load_annotations(index)
	added_mask = np.zeros(image.shape, dtype=np.uint8)
	for i in range(len(masks)):
		masks[i], _ = generator.resize_image(np.expand_dims(masks[i], axis=-1))
		added_mask[np.where(masks[i] > 0)] = (r(), r(), r())

	blended = cv2.addWeighted(image, 0.7, added_mask, 0.3, 0)
	cv2.imshow('img', blended)
	cv2.waitKey()

