########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow
########################################################################

'''
Resnet18 functions
'''

import math
import tensorflow as tf
import numpy as np
from utils import print_var

_BATCH_NORM_DECAY = 0.5
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training, data_format, name):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.compat.v1.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=training, fused=True, name=name)

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, name):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.compat.v1.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
      data_format=data_format, name=name)

def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format, name='bn1')
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format, name='conv1')

  inputs = batch_norm(inputs, training, data_format, name='bn2')
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format, name='conv2')

  return inputs + shortcut

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  # 1*1 conv for downsize
  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format, name='conv_sc')

  # Only the first block per block_layer uses projection_shortcut and strides
  with tf.compat.v1.variable_scope('block1'):
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

  for i in range(1, blocks):
    with tf.compat.v1.variable_scope('block{}'.format(i+1)):
      inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)

# modded resnet18 builder
def resnet18_builder_mod(inputs, keep_prob, training, data_format, num_filters, 
  resnet_version, first_pool_size, first_pool_stride, block_sizes,
  bottleneck, block_fn, block_strides, pre_activation, num_classes, hk):
  """Add operations to classify a batch of input images.

  Args:
    inputs: A Tensor representing a batch of input images.
    training: A boolean. Set to True to add operations required only when
      training the classifier.

  Returns:
    A logits Tensor with shape [<batch_size>, num_classes].
  """

  with tf.compat.v1.variable_scope('resnet_model', reuse=tf.AUTO_REUSE):
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

    # # first conv was replaced with auto-enc layer
    # with tf.compat.v1.variable_scope('conv0'):
    #   inputs = conv2d_fixed_padding(
    #       inputs=inputs, filters=num_first_filters, kernel_size=kernel_size,
    #       strides=conv_stride, data_format=data_format, name='conv0')
    #   inputs = tf.identity(inputs, 'initial_conv')
    #   print_var('conv0', inputs)

      # # We do not include batch normalization or activation functions in V2
      # # for the initial conv1 because the first ResNet unit will perform these
      # # for both the shortcut and non-shortcut paths as part of the first
      # # block's projection. Cf. Appendix of [2].
      # if resnet_version == 1:
      #   inputs = batch_norm(inputs, training, data_format, name='bn0')
      #   inputs = tf.nn.relu(inputs)

      # to fit the weights, added this layer for v2
      # inputs = batch_norm(inputs, training, data_format, name='bn0')
      # inputs = tf.nn.relu(inputs)

    if first_pool_size:
      inputs = tf.compat.v1.layers.max_pooling2d(
          inputs=inputs, pool_size=first_pool_size,
          strides=first_pool_stride, padding='SAME',
          data_format=data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

    for i, num_blocks in enumerate(block_sizes):
      with tf.compat.v1.variable_scope('res{}'.format(i+1)):
        block_num_filters = num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=block_num_filters, bottleneck=bottleneck,
            block_fn=block_fn, blocks=num_blocks,
            strides=block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=data_format)
        print_var('res{}'.format(i+1), inputs)

    # Only apply the BN and ReLU for model that does pre_activation in each
    # building/bottleneck block, eg resnet V2.
    if pre_activation:
      inputs = batch_norm(inputs, training, data_format, name='bn1')
      inputs = tf.nn.relu(inputs)

    # The current top layer has shape
    # `batch_size x pool_size x pool_size x final_size`.
    # ResNet does an Average Pooling layer over pool_size,
    # but that is the same as doing a reduce_mean. We do a reduce_mean
    # here because it performs better than AveragePooling2D.
    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
    print_var('flat_mean', inputs)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    inputs = tf.squeeze(inputs, axes)
    print_var('flat_squeeze', inputs)

    inputs = tf.nn.dropout(inputs, keep_prob)

    # print(inputs)
    # inputs = tf.reshape(inputs, [-1, 2*2*num_filters*8])
    # print(inputs)
    # inputs = max_out(inputs, hk)
    # print(inputs)

    with tf.compat.v1.variable_scope('fc1'):
      inputs = tf.compat.v1.layers.dense(inputs=inputs, units=hk)
      inputs = tf.nn.dropout(inputs, keep_prob)
      print_var('fc1', inputs)
    inputs = tf.identity(inputs, 'last_hidden')
    
    # # final layer built else where
    # with tf.compat.v1.variable_scope('fc2'):
    #   inputs = tf.compat.v1.layers.dense(inputs=inputs, units=num_classes)
    #   # inputs = tf.nn.dropout(inputs, keep_prob)
    # inputs = tf.identity(inputs, 'final_dense')
    # print_var('fc2', inputs)
    # exit()
    return inputs
