########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
import download
import random
from dataset import one_hot_encoded
import cv2

#from PIL import ImageEnhance, Image, ImageOps

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32
crop_imageSize = 28

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
        Return the full path of a data-file for the data-set.
        
        If filename=="" then return the directory of the files.
        """
    
    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
        Unpickle the given file and return the data.
        
        Note that the appropriate dir-name is prepended the filename.
        """
    
    # Create full path for the file.
    file_path = _get_file_path(filename)
    
    print("Loading data: " + file_path)
    
    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')
    
    return data

def randomCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width, ]
    return img

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image

def augment_brightness_camera_images(image):
    #image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image[:,:,2] = image[:,:,2]*random_bright
    #image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    image = np.clip(image, -1.0, 1.0)
    return image

'''def RandomBrightness(image, min_factor, max_factor):
    """
    Random change the passed image brightness.
    :param images: The image to convert into monochrome.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
    PIL.Image.
    """
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_brightness = ImageEnhance.Brightness(image)
    return image_enhancer_brightness.enhance(factor)

def RandomContrast(image, min_factor, max_factor):
    """
    Random change the passed image contrast.
    :param images: The image to convert into monochrome.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
    PIL.Image.
    """
    factor = np.random.uniform(min_factor, max_factor)

    image_enhancer_contrast = ImageEnhance.Contrast(image)
    return image_enhancer_contrast.enhance(factor)
'''

def _convert_images(raw, train_test):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 127.5 - 1.0
    
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    
    # Random crop the images
    _images_ = []
    if train_test == True:
        for i in range(images.shape[0]):
            image = horizontal_flip(images[i], rate=0.5)
            image = augment_brightness_camera_images(image)
            #image = RandomContrast(image, 0.8, 1.8)
            _images_.append(randomCrop(image, crop_imageSize, crop_imageSize))
        _images_ = np.asarray(_images_)
    else:
        for i in range(images.shape[0]):
            _images_.append(randomCrop(images[i], crop_imageSize, crop_imageSize))
        _images_ = np.asarray(_images_)
    del images
    _images_ -= np.mean(_images_)
    _images_ /= np.std(_images_, axis = 0)

    return _images_


def _load_data(filename, train_test):
    """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """
    
    # Load the pickled data-file.
    data = _unpickle(filename)
    
    # Get the raw images.
    raw_images = data[b'data']
    
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    
    # Convert the images.
    images = _convert_images(raw_images, train_test)
    
    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
        Download and extract the CIFAR-10 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """
    
    download.maybe_download_and_extract(url=data_url, download_dir=data_path)


def load_class_names():
    """
        Load the names for the classes in the CIFAR-10 data-set.
        
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """
    
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']
    
    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    
    return names


def load_training_data():
    """
        Load all the training-data for the CIFAR-10 data-set.
        
        The data-set is split into 5 data-files which are merged here.
        
        Returns the images, class-numbers and one-hot encoded class-labels.
        """
    
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, crop_imageSize, crop_imageSize, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    
    # Begin-index for the current batch.
    begin = 0
    
    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1), train_test = True)
        
        # Number of images in this batch.
        num_images = len(images_batch)
        
        # End-index for the current batch.
        end = begin + num_images
        
        # Store the images into the array.
        images[begin:end, :] = images_batch
        
        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch
        
        # The begin-index for the next batch is the current end-index.
        begin = end
    
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
        Load all the test-data for the CIFAR-10 data-set.
        
        Returns the images, class-numbers and one-hot encoded class-labels.
        """
    
    images, cls = _load_data(filename="test_batch", train_test = False)
    
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            assert images.shape[3] == 3
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        """if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]"""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            """# Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]"""
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    maybe_download_and_extract()
    load_class_names()

    train_images, _, train_labels = load_training_data()
    test_images, _, test_labels = load_test_data()
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
########################################################################





























