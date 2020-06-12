########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, Cleverhans
########################################################################

"""
Train Scaleable DP on CIFAR10
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

# GPU settings
_GPUS = "1"
_AUX_GPUS = '2,3,4'

N_GPUS = len(_GPUS.split(','))
N_AUX_GPUS = len(_AUX_GPUS.split(','))
N_ALL_GPUS = N_GPUS + N_AUX_GPUS
print(N_GPUS)
print(N_AUX_GPUS)
print(N_ALL_GPUS)
# ALL_GPU_IDX = [str(n) for n in list(range(N_ALL_GPUS))]
GPU_IDX = [str(n) for n in list(range(0, N_GPUS))]
AUX_GPU_IDX = [str(n) for n in list(range(N_GPUS, N_ALL_GPUS))]

print(GPU_IDX)
print(AUX_GPU_IDX)


# AUX_GPU = str(N_GPUS)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join((_GPUS, _AUX_GPUS))

from datetime import datetime
import time
import math
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools

import numpy as np
np.set_printoptions(threshold=sys.maxsize, precision=16)
import tensorflow as tf

from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans.model import Model
import cifar10
import cifar10_read
from mlp import EncLayer
import robustness

DEBUG = True
DEBUG_STEP = 100
DEBUG_ONETIME = True
    
FLAGS = tf.app.flags.FLAGS
dirCheckpoint = '/ckpts/cifar10_train'
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + dirCheckpoint,
                           """Directory where to read model checkpoints.""")

AECODER_VARIABLES = 'AECODER_VARIABLES'
AECODER_UPDATES = 'AECODER_UPDATES'
CONV_VARIABLES = 'CONV_VARIABLES'
CONV_UPDATES = 'CONV_UPDATES'

#############################
##Hyper-parameter Setting####
#############################
hk = 256; #number of hidden units at the last layer
# D = 50000;
D = 49977
infl = 1; #inflation rate in the privacy budget redistribution
R_lowerbound = 1e-5; #lower bound of the LRP
c = [0, 40, 50, 75] #norm bounds
#epochs = 100; #number of epochs
image_size = 28
padding = 4
#############################

class CustomCallableModelWrapper(Model):

    def __init__(self, callable_fn, output_layer, params, scopes, image_size, adv_noise):
        """
        Wrap a callable function that takes a tensor as input and returns
        a tensor as output with the given layer name.
        :param callable_fn: The callable function taking a tensor and
                            returning a given layer as output.
        :param output_layer: A string of the output layer returned by the
                             function. (Usually either "probs" or "logits".)
        """

        self.output_layer = output_layer
        self.callable_fn = callable_fn
        #self.hk = hk
        self.params = params
        self.scopes = scopes
        self.image_size = image_size
        self.adv_noise = adv_noise

    def fprop(self, x, **kwargs):
        return {self.output_layer: self.callable_fn(x, self.params, self.scopes, self.image_size, self.adv_noise)}

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

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

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

def inference(images, perturbH, params, scopes, training, bn_index):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  #  just rename all scopes
  scope_conv1 = scopes[0]
  scope_conv2 = scopes[1]
  scope_conv3 = scopes[2]
  scope_local4 = scopes[3]
  scope_local5 = scopes[4]
  
  # TODO: check the batch_norm layers to create new variable for each tower
  # conv1
  with tf.variable_scope(scope_conv1) as scope:
    conv = tf.nn.conv2d(images, params[0], [1, 2, 2, 1], padding='SAME') + perturbH
    pre_activation = tf.nn.bias_add(conv, params[1])
    conv1 = tf.nn.relu(pre_activation)
    norm1 = tf.layers.batch_normalization(conv1, scale=True, 
                                          training=training,
                                          reuse=tf.AUTO_REUSE, name='conv1bn{}'.format(bn_index))

  # conv2
  with tf.variable_scope(scope_conv2) as scope:
    conv = tf.nn.conv2d(norm1, params[2], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[3])
    conv2 = tf.nn.relu(pre_activation)
    # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
    current = tf.concat((conv2, norm1), axis=3)
    norm2 = tf.layers.batch_normalization(current, scale=True, 
                                          training=training,
                                          reuse=tf.AUTO_REUSE, name='conv2bn{}'.format(bn_index))

  # conv3
  with tf.variable_scope(scope_conv3) as scope:
    conv = tf.nn.conv2d(norm2, params[4], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[5])
    conv3 = tf.nn.relu(pre_activation)
    norm3 = tf.layers.batch_normalization(conv3, scale=True, 
                                          training=training,
                                          reuse=tf.AUTO_REUSE, name='conv3bn{}'.format(bn_index))
    pool3 = avg_pool(norm3, 2)
    
  # local4
  with tf.variable_scope(scope_local4) as scope:
    h_pool2_flat = tf.reshape(pool3, [-1, int(image_size/4)**2*256])
    z2 = tf.add(tf.matmul(h_pool2_flat, params[6]), params[7])
    BN_norm = tf.layers.batch_normalization(z2, scale=True, 
                                            training=training,
                                            reuse=tf.AUTO_REUSE, name='local4bn{}'.format(bn_index))
  local4 = max_out(BN_norm, hk)
  local4 = tf.clip_by_value(local4, -1, 1) # hidden neurons must be bounded in [-1, 1]
  
  with tf.variable_scope(scope_local5) as scope:
    softmax_linear = tf.add(tf.matmul(local4, params[8]), params[9])
  
  return softmax_linear

def test_inference(images, params, scopes, image_size, adv_noise, bn_index):
  # rename the scopes
  scope_conv1 = scopes[0]
  scope_conv2 = scopes[1]
  scope_conv3 = scopes[2]
  scope_local4 = scopes[3]
  scope_local5 = scopes[4]
  
  # TODO: check the batch_norm layers to use averaged variable
  with tf.variable_scope(scope_conv1) as scope:
    conv = tf.nn.conv2d(images + adv_noise, params[0], [1, 2, 2, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[1])
    conv1 = tf.nn.relu(pre_activation)
    norm1 = tf.layers.batch_normalization(conv1, scale=True,
                                          training=False, reuse=tf.AUTO_REUSE, 
                                          name='conv1bn{}'.format(bn_index))
      
  # conv2
  with tf.variable_scope(scope_conv2) as scope:
    conv = tf.nn.conv2d(norm1, params[2], [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, params[3])
    conv2 = tf.nn.relu(pre_activation)
    # concat conv2 with norm1 to increase the number of features, this step does not affect the privacy preserving guarantee
    current = tf.concat((conv2, norm1), axis=3)
    norm2 = tf.layers.batch_normalization(current, scale=True,
                                          training=False, reuse=tf.AUTO_REUSE, 
                                          name='conv2bn{}'.format(bn_index))

  # conv3
  with tf.variable_scope(scope_conv3) as scope:
    conv = tf.nn.conv2d(norm2, params[4], [1, 1, 1, 1], padding='SAME')#noiseless model
    pre_activation = tf.nn.bias_add(conv, params[5])
    conv3 = tf.nn.relu(pre_activation)
    # norm3
    norm3 = tf.layers.batch_normalization(conv3, scale=True,
                                          training=False, reuse=tf.AUTO_REUSE, 
                                          name='conv3bn{}'.format(bn_index))
    pool3 = avg_pool(norm3, 2)

  # local4, note that we do not need to inject Laplace noise into the testing phase
  with tf.variable_scope(scope_local4) as scope:
    h_pool2_flat = tf.reshape(pool3, [-1, int(image_size/4)**2*256])
    z2 = tf.add(tf.matmul(h_pool2_flat, params[6]), params[7])
    BN_norm = tf.layers.batch_normalization(z2, scale=True,
                                            training=False, reuse=tf.AUTO_REUSE, 
                                            name='local4bn{}'.format(bn_index))
  local4 = max_out(BN_norm, hk)
  local4 = tf.clip_by_value(local4, -1, 1)

  with tf.variable_scope(scope_local5) as scope:
    softmax_linear = tf.add(tf.matmul(local4, params[8]), params[9])
  
  return softmax_linear

def inference_test_input_probs(x, params, scopes, image_size, adv_noise):
  logits = test_inference(x, params, scopes, image_size, adv_noise, 0)
  return tf.nn.softmax(logits)

def train(cifar10_data, epochs, L, learning_rate, scale3, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, logfile, parameter_dict):
  logfile.write("fgsm_eps \t %g, LR \t %g, alpha \t %d , epsilon \t %d \n"%(fgsm_eps, learning_rate, alpha, total_eps))
    
  """Train CIFAR-10 for a number of steps."""
  # make sure variables are placed on cpu
  # TODO: for AWS version, check if put variables on GPU will be better
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
    attacks = ['ifgsm', 'mim', 'madry']

    # manually create all scopes
    with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
      scope_conv1 = scope
    with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
      scope_conv2 = scope
    with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE) as scope:
      scope_conv3 = scope
    with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
      scope_local4 = scope
    with tf.variable_scope('local5', reuse=tf.AUTO_REUSE) as scope:
      scope_local5 = scope
    
    # Parameters Declarification
    #with tf.variable_scope('conv1') as scope:
    # with tf.device('/gpu:{}'.format(AUX_GPU_IDX[0])):
    with tf.variable_scope(scope_conv1) as scope:
      kernel1 = _variable_with_weight_decay('kernel1',
                                          shape=[4, 4, 3, 128],
                                          stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                          wd=0.0, collect=[AECODER_VARIABLES])
      biases1 = _bias_on_cpu('biases1', [128], tf.constant_initializer(0.0), collect=[AECODER_VARIABLES])
    
    # 
    shape     = kernel1.get_shape().as_list()
    w_t       = tf.reshape(kernel1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivity = tf.reduce_max(sing_vals)
    gamma = 2*Delta2/(L*sensitivity)
    
    with tf.variable_scope(scope_conv2) as scope:
      kernel2 = _variable_with_weight_decay('kernel2',
                                          shape=[5, 5, 128, 128],
                                          stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                          wd=0.0, collect=[CONV_VARIABLES])
      biases2 = _bias_on_cpu('biases2', [128], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    
    with tf.variable_scope(scope_conv3) as scope:
      kernel3 = _variable_with_weight_decay('kernel3',
                                          shape=[5, 5, 256, 256],
                                          stddev=np.sqrt(2.0/(5*5*256))/math.ceil(5 / 2),
                                          wd=0.0, collect=[CONV_VARIABLES])
      biases3 = _bias_on_cpu('biases3', [256], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    
    with tf.variable_scope(scope_local4) as scope:
      kernel4 = _variable_with_weight_decay('kernel4', shape=[int(image_size/4)**2*256, hk], 
                                          stddev=0.04, wd=0.004, collect=[CONV_VARIABLES])
      biases4 = _bias_on_cpu('biases4', [hk], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    
    with tf.variable_scope(scope_local5) as scope:
      kernel5 = _variable_with_weight_decay('kernel5', [hk, 10],
                                          stddev=np.sqrt(2.0/(int(image_size/4)**2*256))/math.ceil(5 / 2), 
                                          wd=0.0, collect=[CONV_VARIABLES])
      biases5 = _bias_on_cpu('biases5', [10], tf.constant_initializer(0.1), collect=[CONV_VARIABLES])
    
    # group these for use as parameters
    params = [kernel1, biases1, kernel2, biases2, kernel3, biases3, kernel4, biases4, kernel5, biases5]
    scopes = [scope_conv1, scope_conv2, scope_conv3, scope_local4, scope_local5]
    
    # placeholders for input values
    FM_h = tf.placeholder(tf.float32, [None, 14, 14, 128]) # one time
    noise = tf.placeholder(tf.float32, [None, image_size, image_size, 3]) # one time
    adv_noise = tf.placeholder(tf.float32, [None, image_size, image_size, 3]) # one time

    x_sb = tf.placeholder(tf.float32, [None,image_size,image_size,3]) # input is the bunch of n_batchs
    x_list = tf.split(x_sb, N_GPUS, axis=0) # split it into each batch
    adv_x_sb = tf.placeholder(tf.float32, [None,image_size,image_size,3])
    adv_x_list = tf.split(adv_x_sb, N_GPUS, axis=0)

    x_test = tf.placeholder(tf.float32, [None,image_size,image_size,3])

    y_sb = tf.placeholder(tf.float32, [None, 10]) # input is the bunch of n_batchs
    y_list = tf.split(y_sb, N_GPUS, axis=0) # split it into each batch
    adv_y_sb = tf.placeholder(tf.float32, [None, 10]) # input is the bunch of n_batchs
    # adv_y_list = tf.split(adv_y_sb, N_GPUS, axis=0) # split it into each batch

    y_test = tf.placeholder(tf.float32, [None, 10])

    # re-arrange the input samples
    _split_adv_y_sb = tf.split(adv_y_sb, N_AUX_GPUS, axis=0)
    reorder_adv_y_sb = []
    for i in range(N_GPUS):
      reorder_adv_y_sb.append(tf.concat([_split_adv_y_sb[i + N_GPUS*atk_index] for atk_index in range(len(attacks))], axis=0))

    tower_pretrain_grads = []
    tower_train_grads = []
    all_train_loss = []

    pretrain_opt = tf.train.AdamOptimizer(learning_rate)
    train_opt = tf.train.GradientDescentOptimizer(learning_rate)

    # batch index
    bi = 0
    for gpu in GPU_IDX:
      # putting ops on each tower (GPU)
      with tf.device('/gpu:{}'.format(gpu)):
        print('Train inference GPU placement')
        print('/gpu:{}'.format(gpu))
        # Auto-Encoder #
        # pretrain_adv and pretrain_benign are cost tensor of the encoding layer
        with tf.variable_scope(scope_conv1) as scope:
            Enc_Layer2 = EncLayer(inpt=adv_x_list[bi], n_filter_in = 3, n_filter_out = 128, filter_size = 3, 
                                  W=kernel1, b=biases1, activation=tf.nn.relu)
            pretrain_adv = Enc_Layer2.get_train_ops2(xShape = tf.shape(adv_x_list[bi])[0], Delta = Delta2, 
                                                    epsilon = epsilon2, batch_size = L, learning_rate= learning_rate, 
                                                    W = kernel1, b = biases1, perturbFMx = adv_noise, perturbFM_h = FM_h,bn_index=bi)
            Enc_Layer3 = EncLayer(inpt=x_list[bi], n_filter_in = 3, n_filter_out = 128, filter_size = 3, 
                                                    W=kernel1, b=biases1, activation=tf.nn.relu)
            pretrain_benign = Enc_Layer3.get_train_ops2(xShape = tf.shape(x_list[bi])[0], Delta = Delta2, epsilon = epsilon2, 
                                                      batch_size = L, learning_rate= learning_rate, W = kernel1, b = biases1, 
                                                      perturbFMx = noise, perturbFM_h = FM_h, bn_index=bi)
            pretrain_cost = pretrain_adv + pretrain_benign
        # this cost is not used
        # cost = tf.reduce_sum((Enc_Layer2.cost + Enc_Layer3.cost)/2.0);
    
        # benign conv output
        x_image = x_list[bi] + noise
        y_conv = inference(x_image, FM_h, params, scopes, training=True, bn_index=bi)
        # softmax_y_conv = tf.nn.softmax(y_conv)

        # adv conv output
        adv_x_image = adv_x_list[bi] + adv_noise
        y_adv_conv = inference(adv_x_image, FM_h, params, scopes, training=True, bn_index=bi)
    
        # Calculate loss. Apply Taylor Expansion for the output layer
        perturbW = perturbFM*params[8]
        train_loss = cifar10.TaylorExp(y_conv, y_list[bi], y_adv_conv, reorder_adv_y_sb[bi], L, alpha, perturbW)
        all_train_loss.append(train_loss)

        # list of variables to train
        pretrain_var_list = tf.get_collection(AECODER_VARIABLES)
        train_var_list = tf.get_collection(CONV_VARIABLES)

        # compute tower gradients
        pretrain_grads = pretrain_opt.compute_gradients(pretrain_cost, var_list=pretrain_var_list)
        train_grads = train_opt.compute_gradients(train_loss, var_list=train_var_list)
        # get_pretrain_grads(pretrain_cost, global_step, learning_rate, pretrain_var_list)
        # train_grads = get_train_grads(train_loss, global_step, learning_rate, train_var_list)

        # note this list contains grads and variables
        tower_pretrain_grads.append(pretrain_grads)
        tower_train_grads.append(train_grads)

        # batch index
        bi += 1
    
    # average the gradient from each tower
    pretrain_var_dict = {}
    all_pretrain_grads = {}
    avg_pretrain_grads = []
    for var in tf.get_collection(AECODER_VARIABLES):
      if var.name not in all_pretrain_grads:
        all_pretrain_grads[var.name] = []
        pretrain_var_dict[var.name] = var
    for tower in tower_pretrain_grads:
      for var_grad in tower:
        all_pretrain_grads[var_grad[1].name].append(var_grad[0])
    for var_name in all_pretrain_grads:
      # expand dim 0, then concat on dim 0, then reduce mean on dim 0
      expand_pretrain_grads = [tf.expand_dims(g, 0) for g in all_pretrain_grads[var_name]]
      concat_pretrain_grads = tf.concat(expand_pretrain_grads, axis=0)
      reduce_pretrain_grads = tf.reduce_mean(concat_pretrain_grads, 0)
      # rebuild (grad, var) list
      avg_pretrain_grads.append((reduce_pretrain_grads, pretrain_var_dict[var_name]))
    print('*****************************')
    print("avg_pretrain_grads:")
    for avg_pretrain_grad in avg_pretrain_grads:
      print('grads')
      print((avg_pretrain_grad[0].name, avg_pretrain_grad[0].shape))
      print('var')
      print((avg_pretrain_grad[1].name, avg_pretrain_grad[1].shape))
      print('------')

    train_var_dict = {}
    all_train_grads = {}
    avg_train_grads = []
    for var in tf.get_collection(CONV_VARIABLES):
      if var.name not in all_train_grads:
        all_train_grads[var.name] = []
        train_var_dict[var.name] = var
    for tower in tower_train_grads:
      for var_grad in tower:
        all_train_grads[var_grad[1].name].append(var_grad[0])
    for var_name in all_train_grads:
      # expand dim 0, then concat on dim 0, then reduce mean on dim 0
      expand_train_grads = [tf.expand_dims(g, 0) for g in all_train_grads[var_name]]
      concat_train_grads = tf.concat(expand_train_grads, axis=0)
      reduce_train_grads = tf.reduce_mean(concat_train_grads, 0)
      # rebuild (grad, var) list
      avg_train_grads.append((reduce_train_grads, train_var_dict[var_name]))
    print('*****************************')
    print("avg_train_grads:")
    for avg_train_grad in avg_train_grads:
      print('grads')
      print((avg_train_grad[0].name, avg_train_grad[0].shape))
      print('var')
      print((avg_train_grad[1].name, avg_train_grad[1].shape))
      print('------')
    print('*****************************')

    # get averaged loss tensor
    avg_loss = tf.reduce_mean(tf.stack(all_train_loss), axis=0)

    # TODO: take the average of the bn variables from each tower/training GPU
    # currently, testing is using the bn variables on bn_index 0 (tower/training GPU 0)

    # build train op (apply average gradient to variables)
    # according to 1.13 doc, updates need to be manually applied
    _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print('update ops:')
    print(_update_ops)
    
    with tf.control_dependencies(_update_ops):
      pretrain_op = pretrain_opt.apply_gradients(avg_pretrain_grads, global_step=global_step)
      train_op = train_opt.apply_gradients(avg_train_grads, global_step=global_step)

    # start a session with memory growth
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("session created")
    
    # init kernel 1 and get some values from it
    sess.run(kernel1.initializer)
    dp_epsilon=1.0
    parameter_dict['dp_epsilon'] = dp_epsilon
    _gamma = sess.run(gamma)
    _gamma_x = Delta2/L
    epsilon2_update = epsilon2/(1.0 + 1.0/_gamma + 1/_gamma_x)
    parameter_dict['epsilon2_update'] = epsilon2_update
    print(epsilon2_update/_gamma + epsilon2_update/_gamma_x)
    print(epsilon2_update)
    # NOTE: these values needs to be calculated in testing
    delta_r = fgsm_eps*(image_size**2)
    parameter_dict['delta_r'] = delta_r
    _sensitivityW = sess.run(sensitivity)
    parameter_dict['_sensitivityW'] = _sensitivityW
    delta_h = _sensitivityW*(14**2)
    parameter_dict['delta_h'] = delta_h
    dp_mult = (Delta2/(L*epsilon2_update))/(delta_r / dp_epsilon) + (2*Delta2/(L*epsilon2_update))/(delta_h / dp_epsilon)
    parameter_dict['dp_mult'] = dp_mult

    # place test-time inference into CPU
    with tf.device('/cpu:0'):
      # testing pipeline
      test_x_image = x_test + noise
      test_y_conv = inference(test_x_image, FM_h, params, scopes, training=True, bn_index=0)
      test_softmax_y_conv = tf.nn.softmax(test_y_conv)

    # ============== attacks ================
    iter_step_training = 3
    parameter_dict['iter_step_training'] = iter_step_training
    # iter_step_testing = 1000
    aux_dup_count = N_GPUS
    # split input x_super_batch into N_AUX_GPUS parts
    x_attacks = tf.split(x_sb, N_AUX_GPUS, axis=0)
    # split input x_test into aux_dup_count parts
    x_test_split = tf.split(x_test, aux_dup_count, axis=0)
    
    # setup all attacks
    # attack_switch = {'fgsm':False, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}

    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_input_probs, output_layer='probs', params=params, scopes=scopes, image_size=image_size, adv_noise = adv_noise)
    attack_tensor_training_dict = {}
    attack_tensor_testing_dict = {}

    # define each attack method's tensor
    mu_alpha = tf.placeholder(tf.float32, [1])

    # build each attack
    for atk_idx in range(len(attacks)):
      atk = attacks[atk_idx]
      print('building attack {} tensors'.format(atk))
      # for each gpu assign to each attack
      attack_tensor_training_dict[atk] = []
      attack_tensor_testing_dict[atk] = []
      for i in range(aux_dup_count):
        if atk == 'ifgsm':
          with tf.device('/gpu:{}'.format(AUX_GPU_IDX[i])):
            print('ifgsm GPU placement: /gpu:{}'.format(AUX_GPU_IDX[i]))
            # ifgsm tensors for training
            ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
            attack_tensor_training_dict[atk].append(ifgsm_obj.generate(x=x_attacks[i], 
                                                                       eps=mu_alpha, 
                                                                       eps_iter=mu_alpha/iter_step_training, 
                                                                       nb_iter=iter_step_training, 
                                                                       clip_min=-1.0, clip_max=1.0))

        elif atk == 'mim':
          with tf.device('/gpu:{}'.format(AUX_GPU_IDX[i+1*aux_dup_count])):
            print('mim GPU placement: /gpu:{}'.format(AUX_GPU_IDX[i+1*aux_dup_count]))
            # mim tensors for training
            mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
            attack_tensor_training_dict[atk].append(mim_obj.generate(x=x_attacks[i+1*aux_dup_count], 
                                                                     eps=mu_alpha, 
                                                                     eps_iter=mu_alpha/iter_step_training, 
                                                                     nb_iter=iter_step_training, 
                                                                     decay_factor=1.0, 
                                                                     clip_min=-1.0, clip_max=1.0))

        elif atk == 'madry':
          with tf.device('/gpu:{}'.format(AUX_GPU_IDX[i+2*aux_dup_count])):
            print('madry GPU placement: /gpu:{}'.format(AUX_GPU_IDX[i+2*aux_dup_count]))
            # madry tensors for training
            madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
            attack_tensor_training_dict[atk].append(madry_obj.generate(x=x_attacks[i+2*aux_dup_count],
                                                                       eps=mu_alpha, 
                                                                       eps_iter=mu_alpha/iter_step_training, 
                                                                       nb_iter=iter_step_training, 
                                                                       clip_min=-1.0, clip_max=1.0))


    # combine all attack tensors
    adv_concat_list = []
    for i in range(aux_dup_count):
      adv_concat_list.append(tf.concat([attack_tensor_training_dict[atk][i] for atk in attacks], axis=0))
    # the tensor that contains each batch of adv samples for training
    # has same sample order as the labels
    adv_super_batch_tensor = tf.concat(adv_concat_list, axis=0)

    #====================== attack =========================
    
    #adv_logits, _ = inference(c_x_adv + W_conv1Noise, perturbFM, params)

    print('******************** debug info **********************')
    # list of variables to train
    pretrain_var_list = tf.get_collection(AECODER_VARIABLES)
    print('pretrain var list')
    for v in pretrain_var_list:
      print((v.name, v.shape))
    print('**********************************')
    train_var_list = tf.get_collection(CONV_VARIABLES)
    print('train var list')
    for v in train_var_list:
      print((v.name, v.shape))
    print('**********************************')

    # all variables
    print('all variables')
    vl = tf.global_variables()
    for v in vl:
      print((v.name, v.shape))
    print('**********************************')

    # all ops
    ops = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print('total number of ops')
    print(len(ops))
    # for op in ops:
    #   print(op)
    print('******************** debug info **********************')
    # exit()

    # Create a saver.
    saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # load the most recent models
    _global_step = 0
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found')
    
    T = int(int(math.ceil(D/L))*epochs + 1) # number of steps
    print('total number of steps: {}'.format(T))
    step_for_epoch = int(math.ceil(D/L)); #number of steps for one epoch
    parameter_dict['step_for_epoch'] = step_for_epoch
    print('step_for_epoch: {}'.format(step_for_epoch))
    
    # generate some fixed noise
    perturbH_test = np.random.laplace(0.0, 0, 14*14*128) # one time
    perturbH_test = np.reshape(perturbH_test, [-1, 14, 14, 128]) # one time
    parameter_dict['perturbH_test'] = perturbH_test
    print('perturbH_test')
    print(perturbH_test.shape)
    
    perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*128) # one time
    perturbFM_h = np.reshape(perturbFM_h, [-1, 14, 14, 128]) # one time
    parameter_dict['perturbFM_h'] = perturbFM_h
    print('perturbFM_h')
    print(perturbFM_h.shape)
    
    Noise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L) # one time
    parameter_dict['Noise'] = Noise
    Noise_test = generateIdLMNoise(image_size, 0, epsilon2_update, L) # one time
    parameter_dict['Noise_test'] = Noise_test
    print('Noise and Noise_test')
    print(Noise.shape)
    print(Noise_test.shape)
    # exit()

    # some timing variables
    adv_duration_total = 0.0
    adv_duration_count = 0
    train_duration_total = 0.0
    train_duration_count = 0

    # some debug flag
    adv_batch_flag = True
    batch_flag = True
    L_flag = True
    parameter_flag = True

    _global_step = 0
    for step in xrange(_global_step, _global_step + T):
      start_time = time.time()
      # TODO: fix this
      d_eps = random.random()*0.5
      # d_eps = 0.25
      print('d_eps: {}'.format(d_eps))

      # version with 3 AUX GPU
      # get two super batchs, one for benign training, one for adv training
      super_batch_images, super_batch_labels = cifar10_data.train.next_super_batch(N_GPUS, random=True)
      super_batch_images_for_adv, super_batch_adv_labels = cifar10_data.train.next_super_batch_premix_ensemble(N_GPUS, random=True)

      # TODO: re-arrange the adv labels to match the adv samples

      # run adv_tensors_batch_concat to generate adv samples
      super_batch_adv_images = sess.run(adv_super_batch_tensor, 
                                        feed_dict={x_sb:super_batch_images_for_adv, 
                                                    adv_noise: Noise, mu_alpha:[d_eps]})

      adv_finish_time = time.time()
      adv_duration = adv_finish_time - start_time
      adv_duration_total += adv_duration
      adv_duration_count += 1

      if adv_batch_flag:
        print(super_batch_images.shape)
        print(super_batch_labels.shape)
        print(super_batch_adv_images.shape)
        print(super_batch_adv_labels.shape)
        adv_batch_flag = False

      if batch_flag:
        print(super_batch_images.shape)
        print(super_batch_labels.shape)
        batch_flag = False
      
      if L_flag:
        print("L: {}".format(L))
        L_flag = False

      if parameter_flag:
        print('*=*=*=*=*')
        print(parameter_dict)
        print('*=*=*=*=*', flush=True)
        logfile.write('*=*=*=*=*\n')
        logfile.write(str(parameter_dict))
        logfile.write('*=*=*=*=*\n')
        parameter_flag = False
      
      _, _, avg_loss_value = sess.run([pretrain_op, train_op, avg_loss], 
                                    feed_dict = {x_sb: super_batch_images, y_sb: super_batch_labels, 
                                                 adv_x_sb: super_batch_adv_images, adv_y_sb: super_batch_adv_labels, 
                                                 noise: Noise, adv_noise: Noise_test, FM_h: perturbFM_h})

      assert not np.isnan(avg_loss_value), 'Model diverged with loss = NaN'

      train_finish_time = time.time()
      train_duration = train_finish_time - adv_finish_time
      train_duration_total += train_duration
      train_duration_count += 1

      # save model every 50 epochs
      if step % (50*step_for_epoch) == 0 and (step >= 50*step_for_epoch):
        print('saving model')
        checkpoint_path = os.path.join(os.getcwd() + dirCheckpoint, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      
      # Save the model checkpoint periodically.
      # if step % (10*step_for_epoch) == 0 and (step > _global_step):
      if step % 10 == 0 and (step > _global_step):
        # print n steps and time
        print("current epoch: {:.2f}".format(step / step_for_epoch))
        num_examples_per_step = L * N_GPUS * 2
        avg_adv_duration = adv_duration_total / adv_duration_count
        avg_train_duration = train_duration_total / train_duration_count
        avg_total_duration = avg_adv_duration + avg_train_duration
        examples_per_sec = num_examples_per_step / avg_total_duration
        sec_per_step = avg_total_duration
        # sec_per_batch = sec_per_step / (N_GPUS * 2)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.2f '
                        'sec/step; %.2f sec/adv_gen_op; %.2f sec/train_op)')
        actual_str = format_str % (datetime.now(), step, avg_loss_value,
                            examples_per_sec, sec_per_step, 
                            avg_adv_duration, avg_train_duration)
        print(actual_str, flush=True)
        logfile.write(actual_str + '\n')


def main(argv=None):
  parameter_dict = {}

  # new parameters
  train_size = 50000
  parameter_dict['train_size'] = 50000
  target_batch_size = 1851
  parameter_dict['target_batch_size'] = 1851
  n_batchs = int(np.round(train_size / target_batch_size))
  parameter_dict['n_batchs'] = n_batchs
  n_ensemble = 3
  parameter_dict['n_ensemble'] = n_ensemble

  stair_L = [1851] #batch size # stair_L = [1851] #batch size
  parameter_dict['L'] = 1851
  stair_epochs = [600] #number of epoch for one iteration
  parameter_dict['epochs'] = stair_epochs[0]
  stair_iters = [1] #number of staired iterations
  stair_learning_rate = [5e-2] #learning rate
  parameter_dict['learning_rate'] = 5e-2
  gen_ratio = 0; # [0, 2, 4, 6, 8]
  epsilon3 = 1.0*(1 + gen_ratio/3.0); #epsilon for the last hidden layer, for total epsilon = 10, we used [epsilon3, epsilon2] = [4, 6]
  parameter_dict['epsilon3'] = epsilon3
  epsilon2 = 1.0*(1 + 2*gen_ratio/3.0); #epsilon for the first hidden layer
  parameter_dict['epsilon2'] = epsilon2
  total_eps = epsilon2 + epsilon3 # stay fixed
  parameter_dict['total_eps'] = total_eps
  print(total_eps)
  
  # cifar10_data = cifar10_read.read_data_sets("cifar-10-batches-bin/", one_hot = True)
  cifar10_data = cifar10_read.read_data_sets_parallel("cifar-10-batches-bin/", target_batch_size, n_ensemble)
  print('Done getting images')

  Delta2 = 3*(14*14+2)*16; #global sensitivity for the first hidden layer
  parameter_dict['Delta2'] = Delta2
  alpha = 1.5
  parameter_dict['alpha'] = alpha
  eps2_ratio = 2.0
  parameter_dict['eps2_ratio'] = eps2_ratio

  Delta3_adv = 2*(hk); #10*(hk + 1/4 * hk**2); #global sensitivity for the output layer
  parameter_dict['Delta3_adv'] = Delta3_adv
  Delta3_benign = 2*(hk); #10*(hk); #global sensitivity for the output layer
  parameter_dict['Delta3_benign'] = Delta3_benign

  # eps3_ratio = Delta3_adv/Delta3_benign
  # eps3_benign = 1/(1+eps3_ratio)*(epsilon3)
  # eps3_adv = eps3_ratio/(1+eps3_ratio)*(epsilon3)

  # just changed this from 1/2 to 1
  # later change output layer hidden size to 512 instead of 256
  eps3_benign = epsilon3
  parameter_dict['eps3_benign'] = eps3_benign
  eps3_adv = epsilon3
  parameter_dict['eps3_adv'] = eps3_adv

  fgsm_eps = 0.05
  parameter_dict['fgsm_eps'] = fgsm_eps
  
  dt = datetime.now().strftime("%y%m%d%H%M%S")
  logfile = open('./results/DPAL_PixelDP_train_' + str(total_eps) + '_' + 'random_eps' + '_' + '3_iter' + '_run_3' + '.txt','w')
  for i in range(0, len(stair_L)):
    scale3_benign, scale3_adv = Delta3_benign/(eps3_benign*stair_L[i]), Delta3_adv/(eps3_adv*stair_L[i]); #Laplace noise scale
    parameter_dict['scale3_benign'] = scale3_benign
    parameter_dict['scale3_adv'] = scale3_adv
    perturbFM = np.random.laplace(0.0, scale3_benign, hk*10)
    perturbFM = np.reshape(perturbFM, [hk, 10])
    parameter_dict['perturbFM'] = perturbFM
    for j in range(0, stair_iters[i]):
      train(cifar10_data, stair_epochs[i], stair_L[i], stair_learning_rate[i], scale3_benign, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, logfile, parameter_dict)
      logfile.flush()
      time.sleep(5)
  logfile.close()

if __name__ == '__main__':
  tf.app.run()
