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

import warnings

import keras
import keras_resnet
import keras_resnet.models
import keras_retinanet.models.resnet
from ..models import retinanet, Backbone


class ResNetBackbone(Backbone, keras_retinanet.models.resnet.ResNetBackbone):
    def maskrcnn(self, *args, **kwargs):
        """ Returns a maskrcnn model using the correct backbone.
        """
        return resnet_maskrcnn(*args, backbone=self.backbone, **kwargs)


def resnet_maskrcnn(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3), name='image')

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    model = retinanet.retinanet_mask(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)

    return model


def resnet50_maskrcnn(num_classes, inputs=None, **kwargs):
    return resnet_maskrcnn(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


def resnet101_maskrcnn(num_classes, inputs=None, **kwargs):
    return resnet_maskrcnn(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)


def resnet152_maskrcnn(num_classes, inputs=None, **kwargs):
    return resnet_maskrcnn(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)
