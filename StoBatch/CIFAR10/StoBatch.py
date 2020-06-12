"""
An implementation of StoBatch
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
import random
import cifar10;
from more_attack import *
from cleverhans.attacks_tf import fgm, fgsm
import copy
import cifar10_read
from mlp import EncLayer
import robustness
from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans import utils_tf
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper
from cleverhans.utils import set_log_level
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    
FLAGS = tf.app.flags.FLAGS;
dirCheckpoint = '/tmp/cifar10_train_AdvT';
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + dirCheckpoint,
                           """Directory where to read model checkpoints.""")

AECODER_VARIABLES = 'AECODER_VARIABLES'
CONV_VARIABLES = 'CONV_VARIABLES'

#############################
##Hyper-parameter Setting####
#############################
hk = 256; #number of hidden units at the last layer
D = 50000;
infl = 1; #inflation rate in the privacy budget redistribution
R_lowerbound = 1e-5; #lower bound of the LRP
c = [0, 40, 50, 75] #norm bounds
#epochs = 100; #number of epochs
image_size = 28;
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

def _variable_with_weight_decay(name, shape, stddev, wd, collect):
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
    tf.add_to_collections(collect, var)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _bias_on_cpu(name, shape, initializer, collect):
    """Helper to create a Variable stored on CPU memory.
        
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        
    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    tf.add_to_collections(collect, var)
    
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

def redistributeNoise(LRPfile):
    IMAGE_SIZE = 32
    infl = 1
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

def generateIdLMNoise(image_size, Delta2, epsilon2, L):
    #Initiate the noise for the first hidden layer#
    W_conv1Noise = np.random.laplace(0.0, 0.1, image_size**2 * 3).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [image_size, image_size, 3]);
    for i in range(0, image_size):
        for j in range(0, image_size):
            noise = np.random.laplace(0.0, Delta2/(L*epsilon2), 1);
            for k in range(0, 3):
                W_conv1Noise[i][j][k] = noise;
    #W_conv1Noise = np.random.laplace(0.0, Delta2/(L*epsilon2), image_size**2*3).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 3]);
    return W_conv1Noise;

def generateNoise(image_size, Delta2, epsilon2, L, beta):
    W_conv1Noise = np.random.laplace(0.0, 0.1, image_size**2 * 3).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [image_size, image_size, 3]);
    for i in range(0, image_size):
        for j in range(0, image_size):
            noise = np.random.laplace(0.0, Delta2/(L*(beta[i+6][j+6])*epsilon2), 1);
            for k in range(0, 3):
                W_conv1Noise[i][j][k] = noise;
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 3]);
    return W_conv1Noise;

def inference(images, perturbH, params):
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
    #kernel = _variable_with_weight_decay('weights',
    #                                     shape=[3, 3, 3, 128],
    #                                     stddev=5e-2,
    #                                     wd=0.0)
    conv = tf.nn.conv2d(images, params[0], [1, 2, 2, 1], padding='SAME') + perturbH
    #conv = tf.nn.dropout(conv, 0.9)
    #biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, params[1])
    conv1 = tf.nn.relu(pre_activation, name = scope.name)
    cifar10._activation_summary(conv1)
  
  norm1 = tf.contrib.layers.batch_norm(conv1, scale=True, is_training=True, updates_collections=[CONV_VARIABLES])

  # conv2
  with tf.variable_scope('conv2') as scope:
    #kernel = _variable_with_weight_decay('weights',
    #                                     shape=[5, 5, 128, 128],
    #                                     stddev=5e-2,
    #                                     wd=0.0)
    conv = tf.nn.conv2d(norm1, params[2], [1, 1, 1, 1], padding='SAME')
    #biases = cifar10._variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, params[3])
    conv2 = tf.nn.relu(pre_activation, name = scope.name)
    #conv2 = tf.nn.dropout(conv2, 0.9)
    cifar10._activation_summary(conv2)
  
  # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
  current = tf.concat((conv2, norm1), axis=3)
  # norm2
  norm2 = tf.contrib.layers.batch_norm(current, scale=True, is_training=True, updates_collections=[CONV_VARIABLES])

  # conv3
  with tf.variable_scope('conv3') as scope:
    #kernel = _variable_with_weight_decay('weights',
    #                                     shape=[5, 5, 256, 256],
    #                                     stddev=5e-2,
    #                                     wd=0.0)
    conv = tf.nn.conv2d(norm2, params[4], [1, 1, 1, 1], padding='SAME')
    #biases = cifar10._variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, params[5])
    conv3 = tf.nn.relu(pre_activation, name = scope.name)
    #conv3 = tf.nn.dropout(conv3, 0.9)
    cifar10._activation_summary(conv3)

  # norm3
  norm3 = tf.contrib.layers.batch_norm(conv3, scale=True, is_training=True, updates_collections=[CONV_VARIABLES])
  #pool3, row_pooling_sequence, col_pooling_sequence = tf.nn.fractional_max_pool(norm3, pooling_ratio=[1.0, 2.0, 2.0, 1.0])
  pool3 = avg_pool(norm3, 2)
    
  # local4
  with tf.variable_scope('local4') as scope:
    #weights = cifar10._variable_with_weight_decay('weights', shape=[5*5*256, hk],
    #                                      stddev=0.04, wd=0.004)
    #biases = cifar10._variable_on_cpu('biases', [hk], tf.constant_initializer(0.1))
    h_pool2_flat = tf.reshape(pool3, [-1, int(image_size/4)**2*256]);
    z2 = tf.add(tf.matmul(h_pool2_flat, params[6]), params[7], name=scope.name)
    #Applying normalization for the flat connected layer h_fc1#
    #batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
    #scale2 = tf.Variable(tf.ones([hk]))
    #beta2 = tf.Variable(tf.zeros([hk]))
  BN_norm = tf.contrib.layers.batch_norm(z2,scale=True, is_training=True, updates_collections=[CONV_VARIABLES])
    ###
  local4 = max_out(BN_norm, hk)
  local4 = tf.clip_by_value(local4, -1, 1) # hidden neurons must be bounded in [-1, 1]
    #local4 += perturbFM; # perturb hidden neurons, which are considered coefficients in the differentially private logistic regression layer
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
  #with tf.variable_scope('local5') as scope:
  #weights = cifar10._variable_with_weight_decay('weights', [hk, 10],
  #                                        stddev=1/(hk*1.0), wd=0.0)
  #biases = cifar10._variable_on_cpu('biases', [10],
    #                            tf.constant_initializer(0.0))
  softmax_linear = tf.add(tf.matmul(local4, params[8]), params[9], name=scope.name)
  cifar10._activation_summary(softmax_linear)
  return softmax_linear

def test_inference(images, params, image_size, adv_noise):
    kernel1 = params[0]
    biases1 = params[1]
    with tf.variable_scope('conv1') as scope:
        conv = tf.nn.conv2d(images + adv_noise, kernel1, [1, 2, 2, 1], padding='SAME')
        #conv = tf.nn.dropout(conv, 1.0)
        pre_activation = tf.nn.bias_add(conv, biases1)
        conv1 = tf.nn.relu(pre_activation, name = scope.name)
        cifar10._activation_summary(conv1)

    # norm1
    norm1 = tf.contrib.layers.batch_norm(conv1, scale=True, is_training=True, updates_collections=None)
    
    # conv2
    kernel2 = params[2]
    biases2 = params[3]
    with tf.variable_scope('conv2') as scope:
        conv = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases2)
        conv2 = tf.nn.relu(pre_activation, name = scope.name)
        #conv2 = tf.nn.dropout(conv2, 1.0)
        cifar10._activation_summary(conv2)
    
    # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
    current = tf.concat((conv2, norm1), axis=3)
    # norm2
    norm2 = tf.contrib.layers.batch_norm(current, scale=True, is_training=False, updates_collections=None)

    # conv3
    kernel3 = params[4]
    biases3 = params[5]
    with tf.variable_scope('conv3') as scope:
        conv = tf.nn.conv2d(norm2, kernel3, [1, 1, 1, 1], padding='SAME')#noiseless model
        pre_activation = tf.nn.bias_add(conv, biases3)
        conv3 = tf.nn.relu(pre_activation, name = scope.name)
        #conv3 = tf.nn.dropout(conv3, 1.0)
        cifar10._activation_summary(conv3)

    # norm3
    norm3 = tf.contrib.layers.batch_norm(conv3, scale=True, is_training=False, updates_collections=None)
    #pool3, row_pooling_sequence, col_pooling_sequence = tf.nn.fractional_max_pool(norm3, pooling_ratio=[1.0, 2.0, 2.0, 1.0])
    pool3 = avg_pool(norm3, 2)

    # local4, note that we do not need to inject Laplace noise into the testing phase
    kernel4 = params[6]
    biases4 = params[7]
    with tf.variable_scope('local4') as scope:
        h_pool2_flat = tf.reshape(pool3, [-1, int(image_size/4)**2*256]);
        z2 = tf.add(tf.matmul(h_pool2_flat, kernel4), biases4, name=scope.name)
        #Applying normalization for the flat connected layer h_fc1#
        #batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
        #scale2 = params[10] #tf.Variable(tf.ones([hk]))
        #beta2 = params[11] #tf.Variable(tf.zeros([hk]))
    BN_norm = tf.contrib.layers.batch_norm(z2,scale=True, is_training=False, updates_collections=None)
        #BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
        ###
    local4 = max_out(BN_norm, hk)
    local4 = tf.clip_by_value(local4, -1, 1)
    cifar10._activation_summary(local4)

    #with tf.variable_scope('local5') as scope:
    kernel5 = params[8]
    biases5 = params[9]
    softmax_linear = tf.add(tf.matmul(local4, kernel5), biases5, name=scope.name)
    cifar10._activation_summary(softmax_linear)
    return softmax_linear

def inference_test_input_probs(x, params, image_size, adv_noise):
    logits = test_inference(x, params, image_size, adv_noise)
    return tf.nn.softmax(logits)

"""def adv_test_inference(images):
    # Parameters Declarification
    kernel1 = _variable_with_weight_decay('kernel1',
                                          shape=[3, 3, 3, 128],
                                          stddev=5e-2,
                                          wd=0.0)
    biases1 = cifar10._variable_on_cpu('biases1', [128], tf.constant_initializer(0.0))
    kernel2 = _variable_with_weight_decay('kernel2',
                                        shape=[5, 5, 128, 128],
                                        stddev=5e-2,
                                        wd=0.0)
    biases2 = cifar10._variable_on_cpu('biases2', [128], tf.constant_initializer(0.1))
    kernel3 = _variable_with_weight_decay('kernel3',
                                        shape=[5, 5, 256, 256],
                                        stddev=5e-2,
                                        wd=0.0)
    biases3 = cifar10._variable_on_cpu('biases3', [256], tf.constant_initializer(0.1))
    kernel4 = cifar10._variable_with_weight_decay('kernel4', shape=[5*5*256, hk],
                                        stddev=0.04, wd=0.004)
    biases4 = cifar10._variable_on_cpu('biases4', [hk], tf.constant_initializer(0.1))
    kernel5 = cifar10._variable_with_weight_decay('kernel5', [hk, 10],
                                        stddev=np.sqrt(2.0/(5*5*32))/math.ceil(5 / 2), wd=0.0)
    biases5 = cifar10._variable_on_cpu('biases5', [10],
                                        tf.constant_initializer(0.1))
                                          
    params = [kernel1, biases1, kernel2, biases2, kernel3, biases3, kernel4, biases4, kernel5, biases5]
    ########
    perturbFMAdv = np.random.laplace(0.0, 0, hk)
    perturbFMAdv = np.reshape(perturbFMAdv, [hk]);
    
    logits_adv = inference(images, perturbFMAdv, params)
    softmax_y = tf.nn.softmax(logits_adv);
    c_x_adv = fgsm(images, softmax_y, eps=fgsm_eps, clip_min=-1.0, clip_max=1.0)
    adv_logits = inference(c_x_adv, perturbFMAdv, params)
    return adv_logits"""

def train(cifar10_data, epochs, L, learning_rate, scale3, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, logfile):
  logfile.write("fgsm_eps \t %g, LR \t %g, alpha \t %d , epsilon \t %d \n"%(fgsm_eps, learning_rate, alpha, total_eps))
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    eps_benign = 1/(1+eps2_ratio)*(epsilon2)
    eps_adv = eps2_ratio/(1+eps2_ratio)*(epsilon2)
    
    # Parameters Declarification
    #with tf.variable_scope('conv1') as scope:
    kernel1 = _variable_with_weight_decay('kernel1',
                                         shape=[4, 4, 3, 128],
                                         stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                         wd=0.0, collect=[AECODER_VARIABLES])
    biases1 = _bias_on_cpu('biases1', [128], tf.constant_initializer(0.0), collect=[AECODER_VARIABLES])
    
    shape     = kernel1.get_shape().as_list()
    w_t       = tf.reshape(kernel1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivity = tf.reduce_max(sing_vals)
    gamma = 2*Delta2/(L*sensitivity) #2*3*(14*14 + 2)*16/(L*sensitivity)
    
    #with tf.variable_scope('conv2') as scope:
    kernel2 = _variable_with_weight_decay('kernel2',
                                         shape=[5, 5, 128, 128],
                                         stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                         wd=0.0, collect=[CONV_VARIABLES])
    biases2 = _bias_on_cpu('biases2', [128], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    #with tf.variable_scope('conv3') as scope:
    kernel3 = _variable_with_weight_decay('kernel3',
                                         shape=[5, 5, 256, 256],
                                         stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                         wd=0.0, collect=[CONV_VARIABLES])
    biases3 = _bias_on_cpu('biases3', [256], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    #with tf.variable_scope('local4') as scope:
    kernel4 = _variable_with_weight_decay('kernel4', shape=[int(image_size/4)**2*256, hk], stddev=0.04, wd=0.004, collect=[CONV_VARIABLES])
    biases4 = _bias_on_cpu('biases4', [hk], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
        #with tf.variable_scope('local5') as scope:
    kernel5 = _variable_with_weight_decay('kernel5', [hk, 10],
                                                  stddev=np.sqrt(2.0/(int(image_size/4)**2*256))/math.ceil(5 / 2), wd=0.0, collect=[CONV_VARIABLES])
    biases5 = _bias_on_cpu('biases5', [10], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])

    #scale2 = tf.Variable(tf.ones([hk]))
    #beta2 = tf.Variable(tf.zeros([hk]))
    
    params = [kernel1, biases1, kernel2, biases2, kernel3, biases3, kernel4, biases4, kernel5, biases5]
    ########
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    FM_h = tf.placeholder(tf.float32, [None, 14, 14, 128]);
    noise = tf.placeholder(tf.float32, [None, image_size, image_size, 3]);
    adv_noise = tf.placeholder(tf.float32, [None, image_size, image_size, 3]);
    
    x = tf.placeholder(tf.float32, [None,image_size,image_size,3]);
    adv_x = tf.placeholder(tf.float32, [None,image_size,image_size,3]);
    
    # Auto-Encoder #
    Enc_Layer2 = EncLayer(inpt=adv_x, n_filter_in = 3, n_filter_out = 128, filter_size = 3, W=kernel1, b=biases1, activation=tf.nn.relu)
    pretrain_adv = Enc_Layer2.get_train_ops2(xShape = tf.shape(adv_x)[0], Delta = Delta2, epsilon = epsilon2, batch_size = L, learning_rate= learning_rate, W = kernel1, b = biases1, perturbFMx = adv_noise, perturbFM_h = FM_h)
    Enc_Layer3 = EncLayer(inpt=x, n_filter_in = 3, n_filter_out = 128, filter_size = 3, W=kernel1, b=biases1, activation=tf.nn.relu)
    pretrain_benign = Enc_Layer3.get_train_ops2(xShape = tf.shape(x)[0], Delta = Delta2, epsilon = epsilon2, batch_size = L, learning_rate= learning_rate, W = kernel1, b = biases1, perturbFMx = noise, perturbFM_h = FM_h)
    cost = tf.reduce_sum((Enc_Layer2.cost + Enc_Layer3.cost)/2.0);
    ###
    
    x_image = x + noise;
    y_conv = inference(x_image, FM_h, params);
    softmax_y_conv = tf.nn.softmax(y_conv)
    y_ = tf.placeholder(tf.float32, [None, 10]);
    
    adv_x += adv_noise
    y_adv_conv = inference(adv_x, FM_h, params)
    adv_y_ = tf.placeholder(tf.float32, [None, 10]);
    
    # Calculate loss. Apply Taylor Expansion for the output layer
    perturbW = perturbFM*params[8]
    loss = cifar10.TaylorExp(y_conv, y_, y_adv_conv, adv_y_, L, alpha, perturbW)
    
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #pretrain_step = tf.train.AdamOptimizer(1e-4).minimize(pretrain_adv, global_step=global_step, var_list=[kernel1, biases1]);
    pretrain_var_list = tf.get_collection(AECODER_VARIABLES)
    train_var_list = tf.get_collection(CONV_VARIABLES)
    #print(pretrain_var_list)
    #print(train_var_list)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        pretrain_step = tf.train.AdamOptimizer(learning_rate).minimize(pretrain_adv+pretrain_benign, global_step=global_step, var_list=pretrain_var_list);
        train_op = cifar10.train(loss, global_step, learning_rate, _var_list= train_var_list)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    
    sess.run(kernel1.initializer)
    dp_epsilon=1.0
    _gamma = sess.run(gamma)
    _gamma_x = Delta2/L
    epsilon2_update = epsilon2/(1.0 + 1.0/_gamma + 1/_gamma_x)
    print(epsilon2_update/_gamma + epsilon2_update/_gamma_x)
    print(epsilon2_update)
    delta_r = fgsm_eps*(image_size**2);
    _sensitivityW = sess.run(sensitivity)
    delta_h = _sensitivityW*(14**2)
    #delta_h = 1.0 * delta_r; #sensitivity*(14**2) = sensitivity*(\beta**2) can also be used
    #dp_mult = (Delta2/(L*epsilon2))/(delta_r / dp_epsilon) + (2*Delta2/(L*epsilon2))/(delta_h / dp_epsilon)
    dp_mult = (Delta2/(L*epsilon2_update))/(delta_r / dp_epsilon) + (2*Delta2/(L*epsilon2_update))/(delta_h / dp_epsilon)
    
    dynamic_eps = tf.placeholder(tf.float32);
    """y_test = inference(x, FM_h, params)
    softmax_y = tf.nn.softmax(y_test);
    c_x_adv = fgsm(x, softmax_y, eps=dynamic_eps/3, clip_min=-1.0, clip_max=1.0)
    x_adv = tf.reshape(c_x_adv, [L, image_size, image_size, 3])"""
    
    attack_switch = {'fgsm':True, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}
    
    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_input_probs, output_layer='probs', params=params, image_size=image_size, adv_noise = adv_noise)
    
    # define each attack method's tensor
    mu_alpha = tf.placeholder(tf.float32, [1]);
    attack_tensor_dict = {}
    # FastGradientMethod
    if attack_switch['fgsm']:
        print('creating attack tensor of FastGradientMethod')
        fgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_fgsm = fgsm_obj.generate(x=x, eps=fgsm_eps, clip_min=-1.0, clip_max=1.0, ord=2) # testing now
        x_adv_test_fgsm = fgsm_obj.generate(x=x, eps=mu_alpha, clip_min=-1.0, clip_max=1.0) # testing now
        attack_tensor_dict['fgsm'] = x_adv_test_fgsm
    
    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    # default: eps_iter=0.05, nb_iter=10
    if attack_switch['ifgsm']:
        print('creating attack tensor of BasicIterativeMethod')
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=mu_alpha, eps_iter=fgsm_eps/3, nb_iter=3, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['ifgsm'] = x_adv_test_ifgsm
    
    # MomentumIterativeMethod
    # default: eps_iter=0.06, nb_iter=10
    if attack_switch['mim']:
        print('creating attack tensor of MomentumIterativeMethod')
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_mim = mim_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, decay_factor=1.0, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_mim = mim_obj.generate(x=x, eps=mu_alpha, eps_iter=fgsm_eps/3, nb_iter=3, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['mim'] = x_adv_test_mim
    
    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # default: eps_iter=0.01, nb_iter=40
    if attack_switch['madry']:
        print('creating attack tensor of MadryEtAl')
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        #x_adv_test_madry = madry_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_madry = madry_obj.generate(x=x, eps=mu_alpha, eps_iter=fgsm_eps/3, nb_iter=3, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['madry'] = x_adv_test_madry
    
    #====================== attack =========================
    
    #adv_logits, _ = inference(c_x_adv + W_conv1Noise, perturbFM, params)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(os.getcwd() + dirCheckpoint, sess.graph)
    
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
    
    perturbH_test = np.random.laplace(0.0, 0, 14*14*128)
    perturbH_test = np.reshape(perturbH_test, [-1, 14, 14, 128]);
    
    #W_conv1Noise = np.random.laplace(0.0, Delta2/(L*epsilon2), 32 * 32 * 3).astype(np.float32)
    #W_conv1Noise = np.reshape(_W_conv1Noise, [32, 32, 3])
    
    perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*128)
    perturbFM_h = np.reshape(perturbFM_h, [-1, 14, 14, 128]);
    
    #_W_adv = np.random.laplace(0.0, 0, 32 * 32 * 3).astype(np.float32)
    #_W_adv = np.reshape(_W_adv, [32, 32, 3])
    #_perturbFM_h_adv = np.random.laplace(0.0, 0, 10*10*128)
    #_perturbFM_h_adv = np.reshape(_perturbFM_h_adv, [10, 10, 128]);
    
    test_size = len(cifar10_data.test.images)
    #beta = redistributeNoise(os.getcwd() + '/LRP_0_25_v12.txt')
    #BenignLNoise = generateIdLMNoise(image_size, Delta2, eps_benign, L) #generateNoise(image_size, Delta2, eps_benign, L, beta);
    #AdvLnoise = generateIdLMNoise(image_size, Delta2, eps_adv, L)
    Noise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L)
    #generateNoise(image_size, Delta2, eps_adv, L, beta);
    Noise_test = generateIdLMNoise(image_size, 0, epsilon2_update, L) #generateNoise(image_size, 0, 2*epsilon2, test_size, beta);
    
    emsemble_L = int(L/3)
    preT_epochs = 100
    pre_T = int(int(math.ceil(D/L))*preT_epochs + 1);
    """logfile.write("pretrain: \n")
    for step in range(_global_step, _global_step + pre_T):
        d_eps = random.random()*0.5;
        batch = cifar10_data.train.next_batch(L); #Get a random batch.
        adv_images = sess.run(x_adv, feed_dict = {x: batch[0], dynamic_eps: d_eps, FM_h: perturbH_test})
        for iter in range(0, 2):
            adv_images = sess.run(x_adv, feed_dict = {x: adv_images, dynamic_eps: d_eps, FM_h: perturbH_test})
        #sess.run(pretrain_step, feed_dict = {x: batch[0], noise: AdvLnoise, FM_h: perturbFM_h});
        batch = cifar10_data.train.next_batch(L);
        sess.run(pretrain_step, feed_dict = {x: np.append(batch[0], adv_images, axis = 0), noise: Noise, FM_h: perturbFM_h});
        if step % int(25*step_for_epoch) == 0:
            cost_value = sess.run(cost, feed_dict={x: cifar10_data.test.images, noise: Noise_test, FM_h: perturbH_test})/(test_size*128)
            logfile.write("step \t %d \t %g \n"%(step, cost_value))
            print(cost_value)
    print('pre_train finished')"""
    
    _global_step = 0
    for step in xrange(_global_step, _global_step + T):
      start_time = time.time()
      d_eps = random.random()*0.5;
      batch = cifar10_data.train.next_batch(emsemble_L); #Get a random batch.
      y_adv_batch = batch[1]
      """adv_images = sess.run(x_adv, feed_dict = {x: batch[0], dynamic_eps: d_eps, FM_h: perturbH_test})
      for iter in range(0, 2):
          adv_images = sess.run(x_adv, feed_dict = {x: adv_images, dynamic_eps: d_eps, FM_h: perturbH_test})"""
      adv_images_ifgsm = sess.run(attack_tensor_dict['ifgsm'], feed_dict ={x:batch[0], adv_noise: Noise, mu_alpha:[d_eps]})
      batch = cifar10_data.train.next_batch(emsemble_L);
      y_adv_batch = np.append(y_adv_batch, batch[1], axis = 0)
      adv_images_mim = sess.run(attack_tensor_dict['mim'], feed_dict ={x:batch[0], adv_noise: Noise, mu_alpha:[d_eps]})
      batch = cifar10_data.train.next_batch(emsemble_L);
      y_adv_batch = np.append(y_adv_batch, batch[1], axis = 0)
      adv_images_madry = sess.run(attack_tensor_dict['madry'], feed_dict ={x:batch[0], adv_noise: Noise, mu_alpha:[d_eps]})
      adv_images = np.append(np.append(adv_images_ifgsm, adv_images_mim, axis = 0),adv_images_madry, axis = 0)
      
      batch = cifar10_data.train.next_batch(L); #Get a random batch.

      sess.run(pretrain_step, feed_dict = {x: batch[0], adv_x: adv_images, adv_noise: Noise_test, noise: Noise, FM_h: perturbFM_h});
      _, loss_value = sess.run([train_op, loss], feed_dict = {x: batch[0], y_: batch[1], adv_x: adv_images, adv_y_: y_adv_batch, noise: Noise, adv_noise: Noise_test, FM_h: perturbFM_h})
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      
      # report the result periodically
      if step % (50*step_for_epoch) == 0 and step >= (300*step_for_epoch):
          '''predictions_form_argmax = np.zeros([test_size, 10])
          softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: cifar10_data.test.images, noise: Noise_test, FM_h: perturbH_test})
          argmax_predictions = np.argmax(softmax_predictions, axis=1)
          """for n_draws in range(0, 2000):
            _BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2, L)
            _perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2*L), 14*14*128)
            _perturbFM_h = np.reshape(_perturbFM_h, [-1, 14, 14, 128]);"""
          for j in range(test_size):
            pred = argmax_predictions[j]
            predictions_form_argmax[j, pred] += 2000;
          """softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: cifar10_data.test.images, noise: _BenignLNoise, FM_h: _perturbFM_h})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)"""
          final_predictions = predictions_form_argmax;
          is_correct = []
          is_robust = []
          for j in range(test_size):
              is_correct.append(np.argmax(cifar10_data.test.labels[j]) == np.argmax(final_predictions[j]))
              robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=1.0, dp_delta=0.05, dp_mechanism='laplace') / dp_mult
              is_robust.append(robustness_from_argmax >= fgsm_eps)
          acc = np.sum(is_correct)*1.0/test_size
          robust_acc = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
          robust_utility = np.sum(is_robust)*1.0/test_size
          log_str = "step: {:.1f}\t epsilon: {:.1f}\t benign: {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t".format(step, total_eps, acc, robust_acc, robust_utility, robust_acc*robust_utility)'''
          
          #===================adv samples=====================
          log_str = "step: {:.1f}\t epsilon: {:.1f}\t".format(step, total_eps)
          """adv_images_dict = {}
          for atk in attack_switch.keys():
              if attack_switch[atk]:
                  adv_images_dict[atk] = sess.run(attack_tensor_dict[atk], feed_dict ={x:cifar10_data.test.images})
          print("Done with the generating of Adversarial samples")"""
          #===================adv samples=====================
          adv_acc_dict = {}
          robust_adv_acc_dict = {}
          robust_adv_utility_dict = {}
          test_bach_size = 5000
          for atk in attack_switch.keys():
              print(atk)
              if atk not in adv_acc_dict:
                  adv_acc_dict[atk] = -1
                  robust_adv_acc_dict[atk] = -1
                  robust_adv_utility_dict[atk] = -1
              if attack_switch[atk]:
                  test_bach = cifar10_data.test.next_batch(test_bach_size)
                  adv_images_dict = sess.run(attack_tensor_dict[atk], feed_dict ={x:test_bach[0], adv_noise: Noise_test, mu_alpha:[fgsm_eps]})
                  print("Done adversarial examples")
                  ### PixelDP Robustness ###
                  predictions_form_argmax = np.zeros([test_bach_size, 10])
                  softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: Noise, FM_h: perturbFM_h})
                  argmax_predictions = np.argmax(softmax_predictions, axis=1)
                  for n_draws in range(0, 1000):
                      _BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L);
                      _perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*128)
                      _perturbFM_h = np.reshape(_perturbFM_h, [-1, 14, 14, 128]);
                      if n_draws == 500:
                          print("n_draws = 500")
                      for j in range(test_bach_size):
                          pred = argmax_predictions[j]
                          predictions_form_argmax[j, pred] += 1;
                      softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: (_BenignLNoise/10 + Noise), FM_h: perturbFM_h}) * sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: Noise, FM_h: (_perturbFM_h/10 + perturbFM_h)})
                      #softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: (_BenignLNoise), FM_h: perturbFM_h}) * sess.run(softmax_y_conv, feed_dict={x: adv_images_dict, noise: Noise, FM_h: (_perturbFM_h)})
                      argmax_predictions = np.argmax(softmax_predictions, axis=1)
                  final_predictions = predictions_form_argmax;
                  is_correct = []
                  is_robust = []
                  for j in range(test_bach_size):
                      is_correct.append(np.argmax(test_bach[1][j]) == np.argmax(final_predictions[j]))
                      robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=dp_epsilon, dp_delta=0.05, dp_mechanism='laplace') / dp_mult
                      is_robust.append(robustness_from_argmax >= fgsm_eps)
                  adv_acc_dict[atk] = np.sum(is_correct)*1.0/test_bach_size
                  robust_adv_acc_dict[atk] = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
                  robust_adv_utility_dict[atk] = np.sum(is_robust)*1.0/test_bach_size
                  ##############################
          for atk in attack_switch.keys():
              if attack_switch[atk]:
                  # added robust prediction
                  log_str += " {}: {:.4f} {:.4f} {:.4f} {:.4f}".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk] * robust_adv_utility_dict[atk])
          print(log_str)
          logfile.write(log_str + '\n')

      # Save the model checkpoint periodically.
      if step % (10*step_for_epoch) == 0 and (step > _global_step):
        num_examples_per_step = L
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch))
      """if step % (50*step_for_epoch) == 0 and (step >= 900*step_for_epoch):
        checkpoint_path = os.path.join(os.getcwd() + dirCheckpoint, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step);"""


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract();
  if tf.gfile.Exists('/tmp/cifar10_train'):
    tf.gfile.DeleteRecursively('/tmp/cifar10_train');
  tf.gfile.MakeDirs('/tmp/cifar10_train');
  stair_L = [1851] #batch size # stair_L = [1851] #batch size
  stair_epochs = [600] #number of epoch for one iteration
  stair_iters = [1] #number of staired iterations
  stair_learning_rate = [5e-2] #learning rate
  gen_ratio = 0; # [0, 2, 4, 6, 8]
  """epsilon3 = 1.0*(1 + gen_ratio/3.0); #epsilon for the last hidden layer, for total epsilon = 10, we used [epsilon3, epsilon2] = [4, 6]
  epsilon2 = 1.0*(1 + 2*gen_ratio/3.0); #epsilon for the first hidden layer"""
  epsilon3 = 6;
  epsilon2 = 4;
  total_eps = epsilon2 + epsilon3
  print(total_eps)
  
  cifar10_data = cifar10_read.read_data_sets("cifar-10-batches-bin/", one_hot = True);
  print('Done getting images')

  Delta2 = 3*(14*14+2)*16; #global sensitivity for the first hidden layer
  alpha = 1.5
  eps2_ratio = 2.0

  Delta3_adv = 2*(hk); #10*(hk + 1/4 * hk**2); #global sensitivity for the output layer
  Delta3_benign = 2*(hk); #10*(hk); #global sensitivity for the output layer

  eps3_ratio = Delta3_adv/Delta3_benign;
  eps3_benign = epsilon3; # = 1/(1+eps3_ratio)*(epsilon3)
  eps3_adv = epsilon3; # = eps3_ratio/(1+eps3_ratio)*(epsilon3)
  fgsm_eps = 0.2
  logfile = open('./tmp/results/StoBatch_' + str(total_eps) + '_' + str(fgsm_eps) + '_' + str(stair_learning_rate[0]) + '_' + str(alpha) + '_run2.txt','w')
  for i in range(0, len(stair_L)):
    scale3_benign, scale3_adv = Delta3_benign/(eps3_benign*stair_L[i]), Delta3_adv/(eps3_adv*stair_L[i]); #Laplace noise scale
    perturbFM = np.random.laplace(0.0, scale3_benign, hk*10)
    perturbFM = np.reshape(perturbFM, [hk, 10]);
    for j in range(0, stair_iters[i]):
      train(cifar10_data, stair_epochs[i], stair_L[i], stair_learning_rate[i], scale3_benign, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, logfile)
      logfile.flush()
      time.sleep(5)
  logfile.close();
if __name__ == '__main__':
  tf.app.run()
