from keras_maskrcnn.preprocessing.csv_generator import CSVGenerator
import numpy as np
import cv2
import random

r = lambda: random.randint(0,255)

generator = CSVGenerator(
		'/srv/datasets/postnl/train.txt',
		'/srv/datasets/postnl/classes.txt',
		base_dir='/srv/datasets/postnl'
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

