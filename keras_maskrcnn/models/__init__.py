import keras_retinanet.models


class Backbone(keras_retinanet.models.Backbone):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone_name):
        super(Backbone, self).__init__(backbone_name)

        # a dictionary mapping custom layer names to the correct classes
        from ..layers.roi import RoiAlign
        from ..layers.upsample import Upsample
        from ..layers.misc import Shape, ConcatenateBoxes
        from .. import losses
        self.custom_objects.update({
            'RoiAlign'              : RoiAlign,
            'Upsample'              : Upsample,
            'Shape'                 : Shape,
            'ConcatenateBoxes'      : ConcatenateBoxes,
            'ConcatenateBoxesMasks' : ConcatenateBoxes,  # legacy
            '_mask_conditional'     : losses.mask(),
        })

    def maskrcnn(self, *args, **kwargs):
        """ Returns a maskrcnn model using the correct backbone.
        """
        raise NotImplementedError('maskrcnn method not implemented.')


def backbone(backbone_name):
    """ Returns a backbone object for the given backbone_name.
    """
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    """ Loads a retinanet model using the correct custom objects.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name: Backbone with which the model was trained.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models
    return keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
