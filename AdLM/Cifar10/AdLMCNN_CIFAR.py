"""
An implementation of the Adaptive Laplace Mechanism (AdLM)
Author: Hai Phan, CCS, NJIT
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
from math import sqrt;
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10;

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
FLAGS = tf.app.flags.FLAGS;
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")

#############################
##Hyper-parameter Setting####
#############################
hk = 30; #number of hidden units at the last layer
D = 50000;
infl = 1; #inflation rate in the privacy budget redistribution
R_lowerbound = 1e-5; #lower bound of the LRP
c = [0, 40, 50, 75] #norm bounds
#epochs = 100; #number of epochs
image_size = 20;
padding = 4;
#############################

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

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
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = cifar10._variable_on_cpu(name, shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
    return current

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def block(input, layers, in_features, growth, is_training, keep_prob):
    current = input
    features = in_features
    for idx in xrange(layers):
        tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features

def inference(images, scale3):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  
  ###
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  #xavier = tf.contrib.layers.xavier_initializer_conv2d()
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    #conv = tf.nn.dropout(conv, 0.9)
    biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name = scope.name)
    cifar10._activation_summary(conv1)
  
  norm1 = tf.contrib.layers.batch_norm(conv1, scale=True, is_training=True, updates_collections=None)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name = scope.name)
    #conv2 = tf.nn.dropout(conv2, 0.9)
    cifar10._activation_summary(conv2)
  
  # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
  current = tf.concat((conv2, norm1), axis=3)
  # norm2
  norm2 = tf.contrib.layers.batch_norm(current, scale=True, is_training=True, updates_collections=None)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name = scope.name)
    #conv3 = tf.nn.dropout(conv3, 0.9)
    cifar10._activation_summary(conv3)

  # norm3
  norm3 = tf.contrib.layers.batch_norm(conv3, scale=True, is_training=True, updates_collections=None)
  #pool3, row_pooling_sequence, col_pooling_sequence = tf.nn.fractional_max_pool(norm3, pooling_ratio=[1.0, 2.0, 2.0, 1.0])
  pool3 = avg_pool(norm3, 2)
    
  # local4
  with tf.variable_scope('local4') as scope:
    weights = cifar10._variable_with_weight_decay('weights', shape=[5*5*256, hk],
                                          stddev=0.04, wd=0.004)
    biases = cifar10._variable_on_cpu('biases', [hk], tf.constant_initializer(0.1))
    h_pool2_flat = tf.reshape(pool3, [-1, 5*5*256]);
    z2 = tf.add(tf.matmul(h_pool2_flat, weights), biases, name=scope.name)
    #Applying normalization for the flat connected layer h_fc1#
    batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
    scale2 = tf.Variable(tf.ones([hk]))
    beta2 = tf.Variable(tf.zeros([hk]))
    BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
    ###
    local4 = max_out(BN_norm, hk)
    local4 = tf.clip_by_value(local4, -1, 1) # hidden neurons must be bounded in [-1, 1]
    perturbFM = np.random.laplace(0.0, scale3, hk)
    perturbFM = np.reshape(perturbFM, [hk]);
    local4 += perturbFM; # perturb hidden neurons, which are considered coefficients in the differentially private logistic regression layer
    cifar10._activation_summary(local4)
    
  """print(images.get_shape());
  print(norm1.get_shape());
  print(norm2.get_shape());
  print(pool3.get_shape());
  print(local4.get_shape());"""

  # linear layer(WX + b),
  # We don't apply softmax here because 
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits 
  # and performs the softmax internally for efficiency.
  weights = cifar10._variable_with_weight_decay('weights', [hk, 10],
                                          stddev=1/(hk*1.0), wd=0.0)
  biases = cifar10._variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
  softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  cifar10._activation_summary(softmax_linear)
  return softmax_linear

def test_inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, 3, 128],
                                                 stddev=5e-2,
                                                 wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
        #conv = tf.nn.dropout(conv, 1.0)
        biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        cifar10._activation_summary(conv1)

    # norm1
    norm1 = tf.contrib.layers.batch_norm(conv1, scale=True, is_training=True, updates_collections=None)
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                    shape=[5, 5, 128, 128],
                                                    stddev=5e-2,
                                                    wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name = scope.name)
        #conv2 = tf.nn.dropout(conv2, 1.0)
        cifar10._activation_summary(conv2)
    
    # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
    current = tf.concat((conv2, norm1), axis=3)
    # norm2
    norm2 = tf.contrib.layers.batch_norm(current, scale=True, is_training=True, updates_collections=None)

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')#noiseless model
        biases = cifar10._variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name = scope.name)
        #conv3 = tf.nn.dropout(conv3, 1.0)
        cifar10._activation_summary(conv3)

    # norm3
    norm3 = tf.contrib.layers.batch_norm(conv3, scale=True, is_training=True, updates_collections=None)
    #pool3, row_pooling_sequence, col_pooling_sequence = tf.nn.fractional_max_pool(norm3, pooling_ratio=[1.0, 2.0, 2.0, 1.0])
    pool3 = avg_pool(norm3, 2)

    # local4, note that we do not need to inject Laplace noise into the testing phase
    with tf.variable_scope('local4') as scope:
        weights = cifar10._variable_with_weight_decay('weights', shape=[5*5*256, hk],
                                              stddev=0.04, wd=0.004)
        biases = cifar10._variable_on_cpu('biases', [hk], tf.constant_initializer(0.1))
        h_pool2_flat = tf.reshape(pool3, [-1, 5*5*256]);
        z2 = tf.add(tf.matmul(h_pool2_flat, weights), biases, name=scope.name)
        #Applying normalization for the flat connected layer h_fc1#
        batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
        scale2 = tf.Variable(tf.ones([hk]))
        beta2 = tf.Variable(tf.zeros([hk]))
        BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
        ###
        local4 = max_out(BN_norm, hk)
        cifar10._activation_summary(local4)
    weights = cifar10._variable_with_weight_decay('weights', [hk, 10],
                                                  stddev=1/(hk*1.0), wd=0.0)
    biases = cifar10._variable_on_cpu('biases', [10],
                                        tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    cifar10._activation_summary(softmax_linear)
    return softmax_linear

def train(epochs, L, learning_rate, scale3, Delta2, epsilon2, LRPfile):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs(LRPfile, L, Delta2, epsilon2)
    labels = tf.one_hot(labels, 10)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(images, scale3)

    # Calculate loss. Apply Taylor Expansion for the output layer
    loss = cifar10.TaylorExp(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, learning_rate)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
                                                          
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(os.getcwd() + '/tmp/cifar10_train', sess.graph)
    
    # load the most recent models
    _global_step = 0
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path);
        saver.restore(sess, ckpt.model_checkpoint_path)
        _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found')
    
    T = int(int(math.ceil(D/L))*epochs + 1) # number of steps
    step_for_epoch = int(math.ceil(D/L)); #number of steps for one epoch
    for step in xrange(_global_step, _global_step + T):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      
      # report the result periodically
      if step % (5*step_for_epoch) == 0:
        num_examples_per_step = L
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % (5*step_for_epoch) == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % (5*step_for_epoch) == 0 and (step > _global_step):
        checkpoint_path = os.path.join(os.getcwd() + '/tmp/cifar10_train', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step);

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract();
  if tf.gfile.Exists('/tmp/cifar10_train'):
    tf.gfile.DeleteRecursively('/tmp/cifar10_train');
  tf.gfile.MakeDirs('/tmp/cifar10_train');
  stair_L = [7200, 8000, 8800, 10000] #batch size
  stair_epochs = [100, 100, 50, 50] #number of epoch for one iteration
  stair_iters = [9, 2, 2, 6] #number of staired iterations
  stair_learning_rate = [1.0, 1.0, 1.0, 1.5] #learning rate
  epsilon1 = 0.25; #epsilon for dpLRP
  epsilon2 = 1.5; #epsilon for the first hidden layer
  epsilon3 = 0.75; #epsilon for the last hidden layer
  Delta2 = 3*2*10*10*9; #global sensitivity for the first hidden layer
  Delta3 = 10*(hk + 1/4*hk**2); #global sensitivity for the output layer
  LRPfile = os.getcwd() + '/LRP_0_25_v12.txt';
  for i in range(0, len(stair_L)):
    scale3 = Delta3/(epsilon3*stair_L[i]); #Laplace noise scale
    for j in range(0, stair_iters[i]):
      train(stair_epochs[i], stair_L[i], stair_learning_rate[i], scale3, Delta2, epsilon2, LRPfile)
      time.sleep(5)

if __name__ == '__main__':
  tf.app.run()
