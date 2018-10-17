import setuptools

setuptools.setup(
    name='keras-maskrcnn',
    version='0.2.1',
    description='Keras implementation of MaskRCNN instance aware segmentation.',
    url='https://github.com/fizyr/keras-maskrcnn',
    maintainer='Hans Gaiser',
    maintainer_email='h.gaiser@fizyr.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-retinanet'],
    entry_points = {
        'console_scripts': [
            'maskrcnn-train=keras_maskrcnn.bin.train:main',
        ],
    }
)
