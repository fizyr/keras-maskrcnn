from keras_maskrcnn.preprocessing.csv_generator import CSVGenerator
import numpy as np
import cv2
import random

r = lambda: random.randint(0,255)

generator = CSVGenerator(
		'./data/train.txt',
		'./data/classes.txt',
		base_dir='/srv/datasets/postnl'
		)

for group_index in range(0,2):
	group = generator.groups[group_index]
	image_group       = generator.load_image_group(group)
	annotations_group = generator.load_annotations_group(group)
	inputs, targets   = generator.compute_input_output(group)

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

