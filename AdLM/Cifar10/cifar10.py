"""
An implementation of cifar10.py for the Adaptive Laplace Mechanism (AdLM)
Author: Hai Phan
This code used several functions from Tensorflow, including: 
_activation_summary, _variable_on_cpu, _variable_with_weight_decay, inputs, loss, _add_loss_summaries, and train.
We added redistributeNoise() and TaylorExp() for differentially private cross-entropy objective function based on Taylor Expansion.
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import math

from six.moves import urllib
import tensorflow as tf

import cifar10_input

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=7200,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default=os.getcwd(),
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.')

FLAGS = parser.parse_args()

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 3500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1.0       # Initial learning rate.
infl = 1; #inflation rate in the privacy budget redistribution
R_lowerbound = 1e-2; #lower bound of the LRP

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def redistributeNoise(LRPfile):
    IMAGE_SIZE = 32
    #Step 1: Load differentially private LRP (dpLRP)#
    R = [[0.0 for b in range(IMAGE_SIZE)] for c in range(IMAGE_SIZE)];
    with open(LRPfile, "r") as ins:
        c_index = 0;
        for line in ins:
            array = [];
            array = line.split();
            tempArray = [];
            for i in range(0, len(array)):
                tempArray.append(float(array[i]));
            for b in range(0, IMAGE_SIZE):
                #for c in range(0, 3):
                R[c_index][b] = tempArray[b]
            c_index += 1;
    #End Step 1#
    sum_R = 0;
    for k in range(0,IMAGE_SIZE):
        for j in range(0, IMAGE_SIZE):
            if R[k][j] < R_lowerbound:
                R[k][j] = R_lowerbound;
            sum_R += R[k][j]**infl;
    _beta = [[0.0 for b in range(IMAGE_SIZE)] for c in range(IMAGE_SIZE)];
    for k in range(0,IMAGE_SIZE):
        for j in range(0, IMAGE_SIZE):
            #Compute Privacy Budget Redistribution Vector#
            _beta[k][j] = ((IMAGE_SIZE)**2)*(R[k][j]**infl)/sum_R;
            if _beta[k][j] < R_lowerbound:
                _beta[k][j] = R_lowerbound;
    return _beta;

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(LRPfile, batch_size, Delta2, epsilon2):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  beta = redistributeNoise(LRPfile)
  images, labels = cifar10_input.distorted_inputs(beta = beta, data_dir=data_dir,
                            batch_size=batch_size, Delta2 = Delta2, epsilon2 = epsilon2)
    
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=2500)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def loss(logits, labels):
  """You can also add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def TaylorExp(logits, labels):
    """You can also add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs().
    
        Returns:
        Loss tensor of type float.
        """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.float32)
    # Differentially private sparse cross entropy error based on Taylor Expansion
    '''
        Computes differentially private sigmoid cross entropy given `logits`.
        
        Measures the probability error in discrete classification tasks in which each
        class is independent and not mutually exclusive.
        
        For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))
        
        For x < 0, to avoid overflow in exp(-x), we reformulate the above
        
        x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))
        
        Hence, to ensure stability and avoid overflow, the implementation uses this
        equivalent formulation
        
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
        
        `logits` and `labels` must have the same type and shape. Let denote neg_abs_logits = -abs(x) = -abs(h * W). By Applying Taylor Expansion, we have:
        
        Taylor = max(x, 0) - x * z + log(1 + exp(-abs(x)));
        = max(h * W, 0) - (z * h) * W + (math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        = max(h * W, 0) - (z * h) * W + (math.log(2.0) + 0.5*(-abs(h * W)) + 1.0/8.0*(-abs(h * W))**2)
        = F1 + F2
        where: F1 = max(h * W, 0) + (math.log(2.0) + 0.5*(-abs(h * W)) + 1.0/8.0*(-abs(h * W))**2) and F2 = - (z * h) * W
        
        To ensure that Taylor is differentially private, we need to perturb all the coefficients, including the terms h * W, z * h * W.
        Note that h is differentially private, since its computation on top of the DP Affine transformation does not access the original data.
        Therefore, F1 should be differentially private. We need to preserve DP in F2, which reads the groundtruth label z, as follows:
        
        Since '* z' is an element-wise multiplication and 'z' is one_hot encoding, h can be considered coefficients of the term z * h. By applying Funtional Mechanism, we perturb (z * h) * W as tf.matmul(h + perturbFM, W) * z:
        
        h += perturbFM; where
        
        perturbFM = np.random.laplace(0.0, scale3, hk); # hk = |h|
        perturbFM = np.reshape(perturbFM, [hk]);
        
        This has been done in the last hidden layer in AdLMCNN_CIFAR.inference():
        
        "perturbFM = np.random.laplace(0.0, scale3, hk)
        perturbFM = np.reshape(perturbFM, [hk]);
        local4 += perturbFM;"
        
        where scale3 = Delta3/(epsilon3*batch_size) = 10*(hk + 1/4 * hk**2)/(epsilon3*batch_size); (Lemma 5)
        
        To allow computing gradients at zero, we define custom versions of max and abs functions [Tensorflow].
        
        Source: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_impl.py @ TensorFlow
        '''
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    #Taylor = math_ops.add(relu_logits - y_conv * y_, math_ops.log1p(math_ops.exp(neg_abs_logits)))
    Taylor = math_ops.add(relu_logits - logits * labels, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
    cross_entropy_mean = tf.reduce_mean(Taylor, name='cross_entropy');
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step, lr):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  '''# Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)'''

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  '''if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')'''
  extracted_dir_path = os.path.join(os.getcwd(), 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
