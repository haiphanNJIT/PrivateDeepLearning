########################################################################
# Author: NhaiHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, https://github.com/rmccorm4/Tiny-Imagenet-200
########################################################################

'''
Read and process tinyimagenet
'''

import numpy as np
import pickle
import os, sys
import random
import scipy.misc
from imageio import imread
import time

np.set_printoptions(threshold=sys.maxsize)


def one_hot_encoded(class_numbers, num_classes=None):
    """
        Generate the One-Hot encoded class-labels from an array of integers.
        
        For example, if class_number=2 and num_classes=4 then
        the one-hot encoded label is the float array: [0. 0. 1. 0.]
        
        :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
        
        :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
        
        :return:
        2-dim array of shape: [len(class_numbers), num_classes]
        """
    
    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    
    return np.eye(num_classes, dtype=float)[class_numbers]

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

def convert_images(raw, train=True):
  """
  Just normalize the images with 
  return a 4-dim array with shape: [image_number, height, width, channel]
  where the pixels are floats between 0.0 and 1.0.
  """
  
  # Convert the raw images from the data-files to floating-points.
  img_float = raw / 127.5 - 1.0
  
  # Random crop the images
  _images_ = []
  if train:
    for i in range(img_float.shape[0]):
      image = horizontal_flip(img_float[i], rate=0.5)
      image = augment_brightness_camera_images(image)
      #image = RandomContrast(image, 0.8, 1.8)
      # _images_.append(randomCrop(image, crop_imageSize, crop_imageSize))
      _images_.append(image)
    _images_ = np.asarray(_images_)
  else:
    for i in range(img_float.shape[0]):
      _images_.append(img_float[i])
    _images_ = np.asarray(_images_)
  del img_float

  # mean at each channel
  _images_ -= np.mean(_images_, axis=(0,1,2))
  # _images_ /= np.std(_images_, axis=0)
  # std at each channel
  _images_ /= np.std(_images_, axis=(0,1,2))

  return _images_

# From https://github.com/rmccorm4/Tiny-Imagenet-200
def load_tinyimagenet(path, wnids_path, first_n_classes=200, augment_copies=1, convert=True, resize=False, num_classes=200, dtype=np.float32):
  """
  Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
  TinyImageNet-200 have the same directory structure, so this can be used
  to load any of them.
  Inputs:
  - path: String giving path to the directory to load.
  - dtype: numpy datatype used to load the data.
  Returns: A tuple of
  - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.
  - X_train: (N_train, 3, 64, 64) array of training images
  - y_train: (N_train,) array of training labels
  - X_val: (N_val, 3, 64, 64) array of validation images
  - y_val: (N_val,) array of validation labels
  - X_test: (N_test, 3, 64, 64) array of testing images.
  - y_test: (N_test,) array of test labels; if test labels are not available
    (such as in student code) then y_test will be None.
  """
  # First load wnids
  wnids_file = os.path.join(wnids_path, 'wnids.txt')
  with open(os.path.join(path, wnids_file), 'r') as f:
    wnids = [x.strip() for x in f]

  print(len(wnids))
  wnids = wnids[0:first_n_classes]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  words_file = os.path.join(wnids_path, 'words.txt')
  with open(os.path.join(path, words_file), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]
  class_names = np.asarray(class_names)

  # Next load training data.
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
    if resize:
      X_train_block = np.zeros((num_images, 32, 32, 3), dtype=dtype)
    else:
      X_train_block = np.zeros((num_images, 64, 64, 3), dtype=dtype)
    
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      
      if resize:
        img = scipy.misc.imresize(img, (32, 32, 3))
      if img.ndim == 2:
        ## grayscale file
        img = np.stack((img, img, img), axis=-1)
      # X_train_block[j] = img.transpose(2, 0, 1)
      X_train_block[j] = img
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # We need to concatenate all training data
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      # Select only validation images in chosen wnids set
      if line.split()[1] in wnids:
        img_file, wnid = line.split('\t')[:2]
        img_files.append(img_file)
        val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    
    if resize:
      X_val = np.zeros((num_val, 32, 32, 3), dtype=dtype)
    else:
      X_val = np.zeros((num_val, 64, 64, 3), dtype=dtype)
 
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if resize:
        img = scipy.misc.imresize(img, (32, 32, 3))
      if img.ndim == 2:
        img = np.stack((img, img, img), axis=-1)
        # print(img.shape)

      # X_val[i] = img.transpose(2, 0, 1)
      X_val[i] = img
    # print(X_val.shape)
    # print(num_val)

  # TODO:
  # put them together and covert
  X_train_all = []
  y_train_all = []
  if convert:
    for i in range(augment_copies):
      X_train_copy = convert_images(X_train, True)
      X_train_all.append(X_train_copy)
      y_train_all.append(y_train)
    # X_train = convert_images(X_train, True)
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val = convert_images(X_val, False)

  """
  # Next load test images
  # Students won't have test labels, so we need to iterate over files in the
  # images directory.
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)
  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)
  """
  
  # Omit x_test and y_test because they're unlabeled
  #return class_names, X_train, y_train, X_val, y_val, X_test, y_test
  # return class_names, X_train, one_hot_encoded(y_train, num_classes=200), X_val, one_hot_encoded(y_val, num_classes=200)
  return class_names, X_train, y_train, X_val, y_val
  # return class_names, X_val, y_val

def load_npz_tinyimagenet_onehot(path, filename):
  loaded = np.load(path + filename, allow_pickle=True)
  return loaded['class_names'], loaded['X_train'], loaded['y_train'], loaded['X_val'], loaded['y_val']

################################# normal version #################################
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

############################# parallel version ################################
# will split the dataset into a number of fixed batches
# n_ensemble will re-mix the batches, not really used later

class DataSetParallel(object):
  def __init__(self, images, labels, batch_size, n_ensemble, random_seed=1024):
    # init parameters
    assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    assert images.shape[3] == 3
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._current_index = 0
    self._batch_size = batch_size
    self._n_ensemble = n_ensemble

    # init each batch
    # a random init of index
    np.random.seed(random_seed)
    # shuffle the dataset
    idx = np.arange(self._num_examples)
    np.random.shuffle(idx)
    # pre-allocate n batchs
    # self._idx_list = np.array_split(idx, self._n_batchs)
    self._idx_list, self._n_batchs = self.get_chunks(idx, self._batch_size, exact=True)
    # also pre-allocate ensemble batchs for each batch
    self._sub_idx_list = [np.array_split(i, n_ensemble) for i in self._idx_list]
    # init a set to log the available batchs
    self._idx_in_epoch = list(range(self._n_batchs))

  
  def get_chunks(self, idx, batch_size, exact=True):
    if exact:
      n_batchs = len(idx) // batch_size
      n_data = int(n_batchs * batch_size)
      idx = idx[:n_data]
      new_idx = np.array_split(idx, n_batchs)
      print('Total data points: {}, n_batchs: {}, batch_size: {}'.format(n_data, n_batchs, batch_size))
    else:
      raise NotImplementedError("exact=False not implemented for get_chunks")
    return new_idx, n_batchs

  
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

  def next_super_batch(self, n_batchs, ensemble=False, random=False):
    index_list = []
    for x in range(n_batchs):
      if random:
        # check for epoch
        if len(self._idx_in_epoch) <= 0:
          self._epochs_completed += 1
          self._idx_in_epoch = list(range(self._n_batchs))
        # a random batch
        index = np.random.randint(0, len(self._idx_in_epoch), 1)[0]
        index_list.append(index)
        self._idx_in_epoch = self._idx_in_epoch[:index] + self._idx_in_epoch[index+1:]
      else:
        # next batch in sequence
        index = self._current_index
        self._current_index += 1
        # check if new epoch
        if index >= self._n_batchs:
          index = 0
          self._current_index = 0
          self._epochs_completed += 1
        index_list.append(index)
    # if need ensemble batch, get corresponding ensemble batch
    if ensemble:
      sub_idx_list = [self._sub_idx_list[i] for i in index_list]
      ensemble_data_list = []
      for i in range(self._n_ensemble):
        ensemble_idx = np.concatenate([sub_idx_list[j][i] for j in range(n_batchs)])
        ensemble_images = self._images[ensemble_idx]
        ensemble_labels = self._labels[ensemble_idx]
        ensemble_data_list.append((ensemble_images, ensemble_labels))
      return ensemble_data_list
    # ot just normal batch
    else:
      all_idx = np.concatenate([self._idx_list[index] for index in index_list])
      return self._images[all_idx], self._labels[all_idx]

  def next_batch(self, ensemble=False, random=False):
    ''' Sequentially return next pre-determined batch
    ensemble: Return the sub-batch for ensemble
    random: Random the order, each batch is still considered
    '''
    # print(self._idx_list)
    if random:
      # check for epoch
      if len(self._idx_in_epoch) <= 0:
        self._epochs_completed += 1
        self._idx_in_epoch = list(range(self._n_batchs))
      # return a random batch
      index = np.random.randint(0, len(self._idx_in_epoch), 1)[0]
      self._idx_in_epoch = self._idx_in_epoch[:index] + self._idx_in_epoch[index+1:]
    else:
      # return the next batch in sequence
      index = self._current_index
      self._current_index += 1
      # check if new epoch
      if index >= self._n_batchs:
        index = 0
        self._current_index = 0
        self._epochs_completed += 1
    # if need ensemble batch, get corresponding ensemble batch
    if ensemble:
      i_list = self._sub_idx_list[index]
      imgs = [self._images[i_list[i]] for i in range(self._n_ensemble)]
      lbls = [self._labels[i_list[i]] for i in range(self._n_ensemble)]
      return imgs, lbls
    # ot just normal batch
    else:
      return self._images[self._idx_list[index]], self._labels[self._idx_list[index]]


def read_data_sets_parallel(data_dir, filename, batch_size, n_ensemble):
    '''
      Read train dataset as parallel dataset, test dataset as normal dataset
    '''
    class DataSets(object):
      pass
    data_sets = DataSets()

    print('loading tinyimagenet from npz')
    st = time.time()
    class_names, X_train, y_train, X_val, y_val = load_npz_tinyimagenet_onehot(data_dir, filename)
    print('time take: {}s'.format(time.time() - st))

    data_sets.train = DataSetParallel(X_train, y_train, batch_size, n_ensemble)
    data_sets.test = DataSet(X_val, y_val)
    return data_sets

if __name__ == '__main__':
  # run this script to save pre-build dataset as npz
  n_classes = 30
  augment_copies = 3
  # change this before 
  path = '/data/public/TinyImageNet/tiny-imagenet-200'
  wnids_path = path
  print('now loading from raw file')
  st = time.time()
  class_names, X_train, y_train, X_val, y_val = load_tinyimagenet(path, wnids_path, first_n_classes=n_classes, augment_copies=augment_copies)
  for cla in class_names:
    print(cla[0])
  y_train = one_hot_encoded(y_train, num_classes=n_classes)
  y_val = one_hot_encoded(y_val, num_classes=n_classes)
  print(X_train.shape)
  print(y_train.shape)
  print(X_val.shape)
  print(y_val.shape)
  # print(new_y_val)
  print('time take: {}s'.format(time.time() - st))
  np.savez(path + '/tinyimagenet_{}_classes_augment_{}_onehot.npz'.format(n_classes, augment_copies), class_names=class_names, 
                      X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
