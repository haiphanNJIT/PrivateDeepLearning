# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for building and training NN models.
"""
from __future__ import division

import math

import numpy
import tensorflow as tf


def GetTensorOpName(x):
  """Get the name of the op that created a tensor.

  Useful for naming related tensors, as ':' in name field of op is not permitted

  Args:
    x: the input tensor.
  Returns:
    the name of the op.
  """

  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]


def VaryRate(start, end, saturate_epochs, epoch):
  """Compute a linearly varying number.

  Decrease linearly from start to end until epoch saturate_epochs.

  Args:
    start: the initial number.
    end: the end number.
    saturate_epochs: after this we do not reduce the number; if less than
      or equal to zero, just return start.
    epoch: the current learning epoch.
  Returns:
    the caculated number.
  """
  if saturate_epochs <= 0:
    return start

  step = (start - end) / (saturate_epochs - 1)
  if epoch < saturate_epochs:
    return start - step * epoch
  else:
    return end


def BatchClipByL2norm(t, upper_bound, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t


def SoftThreshold(t, threshold_ratio, name=None):
  """Soft-threshold a tensor by the mean value.

  Softthreshold each dimension-0 vector (for matrix it is each column) by
  the mean of absolute value multiplied by the threshold_ratio factor. Here
  we soft threshold each column as it corresponds to each unit in a layer.

  Args:
    t: the input tensor.
    threshold_ratio: the threshold ratio.
    name: the optional name for the returned tensor.
  Returns:
    the thresholded tensor, where each entry is soft-thresholded by
    threshold_ratio times the mean of the aboslute value of each column.
  """

  assert threshold_ratio >= 0
  with tf.name_scope(values=[t, threshold_ratio], name=name,
                     default_name="soft_thresholding") as name:
    saved_shape = tf.shape(t)
    t2 = tf.reshape(t, tf.concat(axis=0, values=[tf.slice(saved_shape, [0], [1]), -1]))
    t_abs = tf.abs(t2)
    t_x = tf.sign(t2) * tf.nn.relu(t_abs -
                                   (tf.reduce_mean(t_abs, [0],
                                                   keep_dims=True) *
                                    threshold_ratio))
    return tf.reshape(t_x, saved_shape, name=name)


def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """

  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t


def GenerateBinomialTable(m):
  """Generate binomial table.

  Args:
    m: the size of the table.
  Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
  """

  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)
