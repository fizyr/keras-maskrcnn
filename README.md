# Keras MaskRCNN [![Build Status](https://travis-ci.org/fizyr/keras-maskrcnn.svg?branch=master)](https://travis-ci.org/fizyr/keras-maskrcnn) [![DOI](https://zenodo.org/badge/124870610.svg)](https://zenodo.org/badge/latestdoi/124870610)



Keras implementation of MaskRNN instance aware segmentation as described in [Mask R-CNN](https://arxiv.org/abs/1703.06870)
by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, using RetinaNet as base.

## Disclaimer

This repository doesn't strictly implement MaskRCNN as described in their paper. The difference is that their paper describes using a RPN to propose ROIs and to use those ROIs to perform bounding box regression, classification and mask estimation simultaneously. Instead, this repository uses RetinaNet to do the bounding box regression and classification and builds a mask estimation head on top of those predictions.

In theory RetinaNet can be configured to act as a RPN network, which would then be identical to MaskRCNN, but doing so would require more layers and complexity than is actually necessary. Less is more :)

## Installation

1) Clone this repository.
2) Install [keras-retinanet](https://github.com/fizyr/keras-retinanet) (`pip install keras-retinanet --user`). Make sure `tensorflow` is installed and is using the GPU.
3) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/fizyr/keras-maskrcnn/blob/master/examples/ResNet50MaskRCNN.ipynb).
In general, inference of the network works as follows:
```python
outputs = model.predict_on_batch(inputs)
boxes  = outputs[-4]
scores = outputs[-3]
labels = outputs[-2]
masks  = outputs[-1]
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score), labels is shaped `(None, None)` (label corresponding to the score) and masks is shaped `(None, None, 28, 28)`. In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_maskrcnn.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

Execution time on NVIDIA Pascal Titan X is roughly 175msec for an image of shape `1000x800x3`.

Example output images using `keras-maskrcnn` are shown below.

<p align="center">
  <img src="https://github.com/fizyr/keras-maskrcnn/blob/master/images/01.png" alt="Example result of MaskRCNN on MS COCO"/>
  <img src="https://github.com/fizyr/keras-maskrcnn/blob/master/images/02.png" alt="Example result of MaskRCNN on MS COCO"/>
  <img src="https://github.com/fizyr/keras-maskrcnn/blob/master/images/03.png" alt="Example result of MaskRCNN on MS COCO"/>
</p>

## Training

`keras-maskrcnn` can be trained using [this](https://github.com/fizyr/keras-maskrcnn/blob/master/keras_maskrcnn/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_maskrcnn` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

### Usage

For training on [MS COCO](http://cocodataset.org/#home), run:
```shell
# Running directly from the repository:
./keras_maskrcnn/bin/train.py coco /path/to/MS/COCO

# Using the installed script:
maskrcn-train coco /path/to/MS/COCO
```

The pretrained MS COCO model can be downloaded [here](https://github.com/fizyr/keras-maskrcnn/releases). Results using the `cocoapi` are shown below (note: the closest resembling architecture in the MaskRCNN paper achieves an mAP of 0.336).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.278
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.286
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
```

For training on a [custom dataset], a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```shell
# Running directly from the repository:
./keras_maskrcnn/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes

# Using the installed script:
maskrcnn-train csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes
```

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name,/path/to/mask.png
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2`, `class_name` and `mask` are all empty:
```
path/to/image.jpg,,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow,/data/masks/img_001_001.png
/data/imgs/img_002.jpg,215,312,279,391,cat,/data/masks/img_002_001.png
/data/imgs/img_002.jpg,22,5,89,84,bird,/data/masks/img_002_002.png
/data/imgs/img_003.jpg,,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```
