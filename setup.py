import setuptools

setuptools.setup(
    name='keras-maskrcnn',
    version='0.0.1',
    description='Keras implementation of MaskRCNN instance aware segmentation.',
    url='https://github.com/fizyr/keras-maskrcnn',
    author='Hans Gaiser',
    author_email='h.gaiser@fizyr.com',
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
