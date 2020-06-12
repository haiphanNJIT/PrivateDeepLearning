########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow
########################################################################

"""
CIFAR10 realted loss functions and utils
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
parser.add_argument('--batch_size', type=int, default=128,
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
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
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


def distorted_inputs(batch_size, Delta2, epsilon2, W_conv1Noise):
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
  beta = redistributeNoise(os.getcwd() + '/LRP_0_25_v12.txt')
  images, labels = cifar10_input.distorted_inputs(beta = beta, data_dir=data_dir,
                            batch_size=batch_size, Delta2 = Delta2, epsilon2 = epsilon2, W_conv1Noise = W_conv1Noise)
    
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


def TaylorExp(logits, labels, adv_logits, b_labels, L, alpha, perturbW):
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
    labels = tf.cast(labels, tf.float32)
    b_labels = tf.cast(b_labels, tf.float32)
    
    # Differentially private sparse cross entropy error based on Taylor Expansion
    zeros = array_ops.zeros_like(adv_logits, dtype=adv_logits.dtype)
    cond = (adv_logits >= zeros)
    relu_logits = array_ops.where(cond, adv_logits, zeros)
    neg_abs_logits = array_ops.where(cond, -adv_logits, adv_logits)
    Taylor_adv = math_ops.add(relu_logits - adv_logits * b_labels, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
    
    ### Taylor for benign x
    zeros2 = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond2 = (logits >= zeros2)
    relu_logits_benign = array_ops.where(cond2, logits, zeros2)
    neg_abs_logits_benign = array_ops.where(cond2, -logits, logits)
    Taylor_benign = math_ops.add(relu_logits_benign - logits * labels, math.log(2.0) + 0.5*neg_abs_logits_benign + 1.0/8.0*neg_abs_logits_benign**2)
    
    zeros1 = array_ops.zeros_like(perturbW, dtype=perturbW.dtype)
    cond1 = (perturbW >= zeros1)
    perturbW = array_ops.where(cond1, perturbW, -perturbW)
    
    ### Adversarial training loss
    adv_loss = (1/(1 + alpha))*(Taylor_benign + alpha * Taylor_adv)

    cross_entropy_mean = tf.reduce_mean(adv_loss, name='cross_entropy') + tf.reduce_mean(perturbW, name = 'perturbW');
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def TaylorExpAdLM(logits, labels, perturbW):
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
    labels = tf.cast(labels, tf.float32)
    
    ### Taylor for benign x
    zeros2 = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond2 = (logits >= zeros2)
    relu_logits_benign = array_ops.where(cond2, logits, zeros2)
    neg_abs_logits_benign = array_ops.where(cond2, -logits, logits)
    Taylor_benign = math_ops.add(relu_logits_benign - logits * labels, math.log(2.0) + 0.5*neg_abs_logits_benign + 1.0/8.0*neg_abs_logits_benign**2)
    
    zeros1 = array_ops.zeros_like(perturbW, dtype=perturbW.dtype)
    cond1 = (perturbW >= zeros1)
    perturbW = array_ops.where(cond1, perturbW, -perturbW)
    
    cross_entropy_mean = tf.reduce_mean(Taylor_benign, name='cross_entropy') + tf.reduce_mean(perturbW, name = 'perturbW');
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def lossDPSGD(logits, labels):
    """Add L2Loss to all the trainable variables.
        
    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
    of shape [batch_size]
    
    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.argmax(tf.cast(labels, tf.int64), 1)
    print(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
        
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
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


def train(total_loss, global_step, lr, var_list):
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
    #opt = tf.train.AdamOptimizer(lr)
    grads_and_vars = opt.compute_gradients(total_loss, var_list=var_list)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  #variables_averages_op = variable_averages.apply(tf.trainable_variables())
  variables_averages_op = variable_averages.apply(var_list)

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def trainDPSGD(total_loss, global_step, clip_bound, sigma, sensitivity):
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
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
        
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
                                
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        #clip gradient
        grads = tf.clip_by_norm(grads,clip_bound)
        #perturb grads
        grads += tf.random_normal(shape=tf.shape(grads), mean=0.0, stddev = sigma * (sensitivity**2), dtype=tf.float32)
                                                    
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
                                                                
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
                                                                                
    return train_op

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
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
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(os.getcwd(), '/data/cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    os.makedirs(extracted_dir_path)
    tarfile.open(filepath, 'r:gz').extractall(extracted_dir_path)
