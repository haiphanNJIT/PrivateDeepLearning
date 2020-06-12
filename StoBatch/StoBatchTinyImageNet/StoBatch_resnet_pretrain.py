########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, Cleverhans
########################################################################

'''
Train Scaleable DP with RESNET18, TinyImageNet, large batch setting
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

# GPU IDs on our machine, change accordingly
_GPUS = "1"
_AUX_GPUS = '2,3,4'
N_GPUS = len(_GPUS.split(','))
N_AUX_GPUS = len(_AUX_GPUS.split(','))
N_ALL_GPUS = N_GPUS + N_AUX_GPUS

# logicial GPU IDs
GPU_IDX = ['0']
AUX_GPU_IDX = ['1','2','3']

# Set visible GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join((_GPUS, _AUX_GPUS))

# reduce the number of tf system logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# print('GPU:', os.environ["CUDA_VISIBLE_DEVICES"])

from datetime import datetime
import time
import math
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import h5py
import pickle
import math

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans.model import Model

from mlp import EncLayer
import robustness
from tinyimagenet_read import read_data_sets_parallel
from taylor_loss import TaylorExp, TaylorExp_no_noise
from resnet_utils import batch_norm, fixed_padding, _building_block_v2, resnet18_builder_mod
from utils import _fc, print_var, print_var_list, print_op_list

# some global variables for resnet
_DATA_FORMAT = 'channels_last'

def generateIdLMNoise(image_size, Delta2, epsilon2, L):
    #Initiate the noise for the first hidden layer#
    W_conv1Noise = np.random.laplace(0.0, 0.1, image_size**2 * 3).astype(np.float32)
    W_conv1Noise = np.reshape(W_conv1Noise, [image_size, image_size, 3])
    for i in range(0, image_size):
        for j in range(0, image_size):
            noise = np.random.laplace(0.0, Delta2/(L*epsilon2), 1)
            for k in range(0, 3):
                W_conv1Noise[i][j][k] = noise
    #W_conv1Noise = np.random.laplace(0.0, Delta2/(L*epsilon2), image_size**2*3).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 3])
    return W_conv1Noise

def generateNoise(image_size, Delta2, epsilon2, L, beta):
    W_conv1Noise = np.random.laplace(0.0, 0.1, image_size**2 * 3).astype(np.float32)
    W_conv1Noise = np.reshape(W_conv1Noise, [image_size, image_size, 3])
    for i in range(0, image_size):
        for j in range(0, image_size):
            noise = np.random.laplace(0.0, Delta2/(L*(beta[i+6][j+6])*epsilon2), 1)
            for k in range(0, 3):
                W_conv1Noise[i][j][k] = noise
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 3])
    return W_conv1Noise

# inference in training with perturbH
def inference(inputs, perturbH, keep_prob, pre_define_vars, resnet_params, train_params):
  # through first conv/auto-encoding layer
  # no relu and batch norm here as they are included in resnet
  with tf.variable_scope('enc_layer') as scope:
    inputs = fixed_padding(inputs, pre_define_vars['kernel1'].shape[0], _DATA_FORMAT)
    print_var('enc padding', inputs)
    inputs = tf.nn.conv2d(inputs, pre_define_vars['kernel1'], 
                          [1, train_params.enc_stride, train_params.enc_stride, 1], padding='VALID')
    inputs = inputs + perturbH + pre_define_vars['biases1']
    print_var('enc output', inputs)
  # resnet18 without first conv, return last hidden layer
  inputs = resnet18_builder_mod(inputs, keep_prob, True, _DATA_FORMAT, resnet_params.num_filters, 
    resnet_params.resnet_version, resnet_params.first_pool_size, resnet_params.first_pool_stride, 
    resnet_params.block_sizes, resnet_params.bottleneck, resnet_params.block_fn, resnet_params.block_strides,
    resnet_params.pre_activation, train_params.num_classes, train_params.hk)
  # the last fc layer
  inputs = tf.clip_by_value(inputs, -1, 1)
  inputs = _fc(inputs, train_params.num_classes, None, reuse=tf.AUTO_REUSE, name='fc2')
  # inputs = _fc(inputs, train_params.num_classes, None, name='fc2')
  return inputs

# inference in testing with perturbH
def test_inference(inputs, perturbH, keep_prob, pre_define_vars, resnet_params, train_params):
  # through first conv/auto-encoding layer
  # no relu and batch norm here as they are included in resnet
  with tf.variable_scope('enc_layer') as scope:
    inputs = fixed_padding(inputs, pre_define_vars['kernel1'].shape[0], _DATA_FORMAT)
    print_var('enc padding', inputs)
    inputs = tf.nn.conv2d(inputs, pre_define_vars['kernel1'], 
                          [1, train_params.enc_stride, train_params.enc_stride, 1], padding='VALID')
    inputs = inputs + perturbH + pre_define_vars['biases1']
    print_var('enc output', inputs)
  # resnet18 without first conv, return last hidden layer
  inputs = resnet18_builder_mod(inputs, keep_prob, False, _DATA_FORMAT, resnet_params.num_filters, 
    resnet_params.resnet_version, resnet_params.first_pool_size, resnet_params.first_pool_stride, 
    resnet_params.block_sizes, resnet_params.bottleneck, resnet_params.block_fn, resnet_params.block_strides,
    resnet_params.pre_activation, train_params.num_classes, train_params.hk)
  # the last fc layer
  inputs = tf.clip_by_value(inputs, -1, 1)
  inputs = _fc(inputs, train_params.num_classes, None, reuse=tf.AUTO_REUSE, name='fc2')
  # inputs = _fc(inputs, train_params.num_classes, None, name='fc2')
  return inputs

# inference in testing (adv generating) with adv_noise
def adv_inference(inputs, adv_noise, keep_prob, pre_define_vars, resnet_params, train_params):
  # through first conv/auto-encoding layer
  # no relu and batch norm here as they are included in resnet
  with tf.variable_scope('enc_layer') as scope:
    inputs = fixed_padding(inputs + adv_noise, pre_define_vars['kernel1'].shape[0], _DATA_FORMAT)
    print_var('enc padding', inputs)
    inputs = tf.nn.conv2d(inputs, pre_define_vars['kernel1'], 
                          [1, train_params.enc_stride, train_params.enc_stride, 1], padding='VALID')
    inputs = inputs + pre_define_vars['biases1']
    print_var('enc output', inputs)
  # resnet18 without first conv
  inputs = resnet18_builder_mod(inputs, keep_prob, False, _DATA_FORMAT, resnet_params.num_filters, 
    resnet_params.resnet_version, resnet_params.first_pool_size, resnet_params.first_pool_stride, 
    resnet_params.block_sizes, resnet_params.bottleneck, resnet_params.block_fn, resnet_params.block_strides,
    resnet_params.pre_activation, train_params.num_classes, train_params.hk)
  # the last fc layer
  inputs = tf.clip_by_value(inputs, -1, 1)
  inputs = _fc(inputs, train_params.num_classes, None, reuse=tf.AUTO_REUSE, name='fc2')
  # inputs = _fc(inputs, train_params.num_classes, None, name='fc2')
  return inputs

# wrapper for inference when generating adv samples
def inference_test_output_probs(inputs, adv_noise, keep_prob, pre_define_vars, resnet_params, train_params):
  logits = adv_inference(inputs, adv_noise, keep_prob, pre_define_vars, resnet_params, train_params)
  return tf.nn.softmax(logits)

# model wrapper for cleverhans compatibility       
class CustomCallableModelWrapper(Model):
  def __init__(self, callable_fn, output_layer, adv_noise, keep_prob, pre_define_vars, 
              resnet_params, train_params):
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
    self.adv_noise = adv_noise
    self.keep_prob = keep_prob
    self.pre_define_vars = pre_define_vars
    self.resnet_params = resnet_params
    self.train_params = train_params

  def fprop(self, x, **kwargs):
    return {self.output_layer: self.callable_fn(x, self.adv_noise, self.keep_prob, self.pre_define_vars, 
                                                self.resnet_params, self.train_params)}

# train the scaleable DP with pre-trained RESNET-18
def PDP_resnet_with_pretrain_adv(TIN_data, resnet_params, train_params, params_to_save):
  # dict for encoding layer variables and output layer variables
  pre_define_vars = {}

  # list of variables to train
  train_vars = []
  pretrain_vars = []

  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
    
    # Parameters Declarification
    ######################################
    
    # encoding (pretrain) layer variables
    with tf.variable_scope('enc_layer', reuse=tf.AUTO_REUSE) as scope:
      kernel1 = tf.get_variable('kernel1', shape=[train_params.enc_kernel_size, train_params.enc_kernel_size, 
                                3, train_params.enc_filters], dtype=tf.float32, 
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
      biases1 = tf.get_variable('biases1', shape=[train_params.enc_filters], dtype=tf.float32, 
                                initializer=tf.constant_initializer(0.0))
    pre_define_vars['kernel1'] = kernel1
    pre_define_vars['biases1'] = biases1 
    train_vars.append(kernel1)
    train_vars.append(biases1)
    pretrain_vars.append(kernel1)
    pretrain_vars.append(biases1)

    shape     = kernel1.get_shape().as_list()
    w_t       = tf.reshape(kernel1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivity = tf.reduce_max(sing_vals)
    gamma = 2*train_params.Delta2/(train_params.effective_batch_size * sensitivity)
    print('gamma: {}'.format(gamma))
    
    # output layer variables
    with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE) as scope:
      stdv = 1.0 / math.sqrt(train_params.hk)
      final_w = tf.get_variable('kernel', shape=[train_params.hk, train_params.num_classes], dtype=tf.float32, 
                                initializer=tf.random_uniform_initializer(-stdv, stdv))
      final_b = tf.get_variable('bias', shape=[train_params.num_classes], dtype=tf.float32, 
                                initializer=tf.constant_initializer(0.0))
    pre_define_vars['final_w'] = final_w
    pre_define_vars['final_b'] = final_b 
    train_vars.append(final_w)
    train_vars.append(final_b)
    ######################################
    
    # Build a Graph that computes the logits predictions from the inputs
    ######################################
    # input placeholders
    x_sb = tf.placeholder(tf.float32, [None,train_params.image_size,train_params.image_size,3], name='x_sb') # input is the bunch of n_batchs
    x_sb_adv = tf.placeholder(tf.float32, [None,train_params.image_size,train_params.image_size,3], name='x_sb_adv')
    x_test = tf.placeholder(tf.float32, [None,train_params.image_size,train_params.image_size,3], name='x_test')

    y_sb = tf.placeholder(tf.float32, [None, train_params.num_classes], name='y_sb') # input is the bunch of n_batchs (super batch)
    y_sb_adv = tf.placeholder(tf.float32, [None, train_params.num_classes], name='y_sb_adv')
    y_test = tf.placeholder(tf.float32, [None, train_params.num_classes], name='y_test')

    FM_h = tf.placeholder(tf.float32, [None, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters], name='FM_h') # one time
    noise = tf.placeholder(tf.float32, [None, train_params.image_size, train_params.image_size, 3], name='noise') # one time
    adv_noise = tf.placeholder(tf.float32, [None, train_params.image_size, train_params.image_size, 3], name='adv_noise') # one time

    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

    # list of grads for each GPU
    tower_pretrain_grads = []
    tower_train_grads = []
    all_train_loss = []

    # optimizers
    pretrain_opt = tf.train.AdamOptimizer(learning_rate)
    train_opt = tf.train.AdamOptimizer(learning_rate)

    # model and loss on one GPU
    with tf.device('/gpu:{}'.format(GPU_IDX[0])):
      # setup encoding layer training
      with tf.variable_scope('enc_layer', reuse=tf.AUTO_REUSE) as scope:
        Enc_Layer2 = EncLayer(inpt=x_sb, n_filter_in=None, n_filter_out=None, filter_size=None, 
                              W=kernel1, b=biases1, activation=tf.nn.relu)
        pretrain_adv = Enc_Layer2.get_train_ops2(xShape=tf.shape(x_sb_adv)[0], Delta=train_params.Delta2, 
                                                epsilon=train_params.epsilon2, batch_size=None, learning_rate=None,
                                                W=kernel1, b=biases1, perturbFMx=adv_noise, perturbFM_h=FM_h)
        Enc_Layer3 = EncLayer(inpt=x_sb, n_filter_in=None, n_filter_out=None, filter_size=None, 
                              W=kernel1, b=biases1, activation=tf.nn.relu)
        pretrain_benign = Enc_Layer3.get_train_ops2(xShape=tf.shape(x_sb)[0], Delta=train_params.Delta2, 
                                                    epsilon=train_params.epsilon2, batch_size=None, learning_rate=None,
                                                    W=kernel1, b=biases1, perturbFMx=noise, perturbFM_h=FM_h)
        pretrain_cost = tf.reduce_mean(pretrain_adv + pretrain_benign)
      print_var('pretrain_cost', pretrain_cost)
      
      # use standard loss first
      y_logits = inference(x_sb + noise, FM_h, keep_prob, pre_define_vars, resnet_params, train_params)
      y_softmax = tf.nn.softmax(y_logits)

      y_logits_adv = inference(x_sb_adv + adv_noise, FM_h, keep_prob, pre_define_vars, resnet_params, train_params)
      y_softmax_adv = tf.nn.softmax(y_logits_adv)

      # taylor exp
      # TODO: use noise here
      perturbW = train_params.perturbFM * final_w
      # train_loss = TaylorExp_no_noise(y_softmax, y_sb, y_softmax_adv, y_sb_adv, 
      #                        train_params.effective_batch_size, train_params.alpha)
      train_loss = TaylorExp(y_softmax, y_sb, y_softmax_adv, y_sb_adv, 
                             train_params.effective_batch_size, train_params.alpha, perturbW)
      print_var('train_loss', train_loss)
      all_train_loss.append(train_loss)
    
    # split testing in each gpu
    x_sb_tests = tf.split(x_sb, N_ALL_GPUS, axis=0)
    y_softmax_test_list = []
    for gpu in range(N_ALL_GPUS):
      with tf.device('/gpu:{}'.format(gpu)):
        # testing graph now in each gpu
        y_logits_test = test_inference(x_sb_tests[gpu] + noise, FM_h, keep_prob, pre_define_vars, resnet_params, train_params)
        y_softmax_test_list.append(tf.nn.softmax(y_logits_test))
    y_softmax_test_concat = tf.concat(y_softmax_test_list, axis=0)

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    all_vars = tf.global_variables()
    print_var_list('all vars', all_vars)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # add selected vars into trainable variable list
    # ('res4' in var.name and ('gamma' in var.name or 'beta' in var.name)) or
    for var in tf.global_variables():
      if 'resnet_model' in var.name and \
        ('conv0' in var.name or 
        'fc' in var.name or 
        'res3' in var.name or 
        'res4' in var.name or 
        'res1' in var.name or 
        'res2' in var.name) and \
          ('gamma' in var.name or 
            'beta' in var.name or 
            'kernel' in var.name or
            'bias' in var.name):
        if var not in train_vars:
          train_vars.append(var)
      elif 'enc_layer' in var.name and \
        ('kernel' in var.name or
          'bias' in var.name):
        if var not in pretrain_vars:
          pretrain_vars.append(var)
        if var not in train_vars:
          train_vars.append(var)
      elif 'enc_layer' in var.name and \
        ('gamma' in var.name or 
          'beta' in var.name):
        if var not in pretrain_vars:
          pretrain_vars.append(var)
    
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print_var_list('train_vars', train_vars)
    print_var_list('pretrain_vars', pretrain_vars)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # op for compute grads on one gpu
    with tf.device('/gpu:{}'.format(GPU_IDX[0])):
      # get all update_ops (updates of moving averageand std) for batch normalizations
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      print_op_list('update ops', update_ops)
      enc_update_ops = [op for op in update_ops if 'enc_layer' in op.name]
      print_op_list('enc layer update ops', enc_update_ops)

      # when the gradients are computed, update the batch_norm
      with tf.control_dependencies(enc_update_ops):
        pretrain_grads = pretrain_opt.compute_gradients(pretrain_cost, var_list=pretrain_vars)
        print('*********** pretrain_grads ***********')
        for x in pretrain_grads:
          print(x)
        print('**********************')
      with tf.control_dependencies(update_ops):
        train_grads = train_opt.compute_gradients(train_loss, var_list=train_vars)
        print('*********** train_grads ***********')
        for x in train_grads:
          print(x)
        print('**********************')
      avg_pretrain_grads = pretrain_grads
      avg_train_grads = train_grads
      
      # get averaged loss tensor for pretrain and train ops
      total_loss = tf.reduce_sum(tf.stack(all_train_loss))
      total_pretrain_loss = tf.reduce_mean(pretrain_cost)

    # prepare to save gradients for large batch
    pretrain_grads_save = [g for g,v in pretrain_grads]
    # print('*********** pretrain_grads_save ***********' + str(pretrain_grads_save) + '**********************')
    train_grads_save = [g for g,v in train_grads]
    # print('*********** train_grads_save ***********' + str(train_grads_save) + '**********************')
    pretrain_grads_shapes = [g.shape.as_list() for g in pretrain_grads_save]
    train_grads_shapes = [g.shape.as_list() for g in train_grads_save]

    # placeholders for importing saved gradients
    pretrain_grads_placeholders = []
    for g,v in pretrain_grads:
      pretrain_grads_placeholders.append(tf.placeholder(tf.float32, v.shape))

    train_grads_placeholders = []
    for g,v in train_grads:
      train_grads_placeholders.append(tf.placeholder(tf.float32, v.shape))

    # construct the (grad, var) list
    assemble_pretrain_grads = []
    for i in range(len(pretrain_vars)):
      assemble_pretrain_grads.append((pretrain_grads_placeholders[i], pretrain_vars[i]))
    
    assemble_train_grads = []
    for i in range(len(train_grads)):
      assemble_train_grads.append((train_grads_placeholders[i], train_vars[i]))
    
    # apply the saved gradients
    pretrain_op = pretrain_opt.apply_gradients(assemble_pretrain_grads, global_step=global_step)
    train_op = train_opt.apply_gradients(assemble_train_grads, global_step=global_step)
    ######################################

    # Create a saver.
    saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)
    
    # start a session with memory growth
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("session created")

    # get some initial values
    sess.run(kernel1.initializer)
    _gamma = sess.run(gamma)
    _gamma_x = train_params.Delta2 / train_params.effective_batch_size
    epsilon2_update = train_params.epsilon2/(1.0 + 1.0/_gamma + 1/_gamma_x)
    delta_r = train_params.fgsm_eps * (train_params.image_size ** 2)
    _sensitivityW = sess.run(sensitivity)
    delta_h = _sensitivityW*(train_params.enc_h_size ** 2)
    dp_mult = (train_params.Delta2 / (train_params.effective_batch_size * epsilon2_update)) / (delta_r / train_params.dp_epsilon) + \
      (2 * train_params.Delta2 / (train_params.effective_batch_size * epsilon2_update))/(delta_h / train_params.dp_epsilon)
    
    # save some valus for testing
    params_to_save['epsilon2_update'] = epsilon2_update
    params_to_save['dp_mult'] = dp_mult

    #######################################
    # ADV attacks
    #######################################

    # split input for attacks
    x_attacks = tf.split(x_sb, 3, axis=0) # split it into each batch
    
    # currently only ifgsm, mim, and madry attacks are available
    attack_switch = {'fgsm':False, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}
    
    # wrap the inference
    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_output_probs, output_layer='probs', 
                                                adv_noise=adv_noise, keep_prob=keep_prob, pre_define_vars=pre_define_vars, 
                                                resnet_params=resnet_params, train_params=train_params)
    
    # to save the reference to the attack tensors
    attack_tensor_training_dict = {}
    attack_tensor_testing_dict = {}

    # placeholder for eps parameter
    mu_alpha = tf.placeholder(tf.float32, [1])
      
    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    # place on specific GPU
    with tf.device('/gpu:{}'.format(AUX_GPU_IDX[0])):
      print('ifgsm GPU placement')
      print('/gpu:{}'.format(AUX_GPU_IDX[0]))
      if attack_switch['ifgsm']:
          print('creating attack tensor of BasicIterativeMethod')
          ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
          attack_tensor_training_dict['ifgsm'] = ifgsm_obj.generate(x=x_attacks[0], eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_training, nb_iter=train_params.iter_step_training, clip_min=-1.0, clip_max=1.0)
          attack_tensor_testing_dict['ifgsm'] = ifgsm_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, clip_min=-1.0, clip_max=1.0)

    # MomentumIterativeMethod
    # place on specific GPU
    with tf.device('/gpu:{}'.format(AUX_GPU_IDX[1])):
      print('mim GPU placement')
      print('/gpu:{}'.format(AUX_GPU_IDX[1]))
      if attack_switch['mim']:
          print('creating attack tensor of MomentumIterativeMethod')
          mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
          attack_tensor_training_dict['mim'] = mim_obj.generate(x=x_attacks[1], eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_training, nb_iter=train_params.iter_step_training, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
          attack_tensor_testing_dict['mim'] = mim_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
      
    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # place on specific GPU
    with tf.device('/gpu:{}'.format(AUX_GPU_IDX[2])):
      print('madry GPU placement')
      print('/gpu:{}'.format(AUX_GPU_IDX[2]))
      if attack_switch['madry']:
          print('creating attack tensor of MadryEtAl')
          madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
          attack_tensor_training_dict['madry'] = madry_obj.generate(x=x_attacks[2], eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_training, nb_iter=train_params.iter_step_training, clip_min=-1.0, clip_max=1.0)
          attack_tensor_testing_dict['madry'] = madry_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, clip_min=-1.0, clip_max=1.0)

    # combine the tensors
    adv_tensors_concat = tf.concat([attack_tensor_training_dict[x] for x in train_params.attacks], axis=0)
    #######################################

    # init op
    print('initialize_all_variables')
    init = tf.initialize_all_variables()
    sess.run(init)

    # load pretrained variables of RESNET
    if train_params.load_weights:
      # first we need to load variable name convert table
      tgt_var_name_dict = {}
      with open(train_params.weight_table_path, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
        for line in lines:
          var_names = line.strip().split(' ')
          if var_names[1] == 'NONE':
            continue
          else:
            tgt_var_name_dict[var_names[0]] = var_names[1]

      # load variables dict from checkpoint
      pretrained_var_dict = load_pretrained_vars()

      # load pre-trained vars using name convert table
      for var in tf.global_variables():
        if var.name in tgt_var_name_dict:
          # print('var \"{}\" found'.format(var.name))
          try:
            var.load(pretrained_var_dict[tgt_var_name_dict[var.name]], session=sess)
            print('{} loaded'.format(var.name))
          except:
            print('var {} not loaded since shape changed'.format(var.name))
        else:
          if 'Adam' not in var.name:
            print('var \"{}\" NOT FOUND'.format(var.name))
    else:
      print('Training model from scratch')


    #####################################
    # init noise and save for testing
    perturbH_test = np.random.laplace(0.0, 0, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters)
    perturbH_test = np.reshape(perturbH_test, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
    params_to_save['perturbH_test'] = perturbH_test
    
    perturbFM_h = np.random.laplace(0.0, 2*train_params.Delta2/(epsilon2_update*train_params.effective_batch_size), 
                                        train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters)
    perturbFM_h = np.reshape(perturbFM_h, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
    params_to_save['perturbFM_h'] = perturbFM_h

    Noise = generateIdLMNoise(train_params.image_size, train_params.Delta2, epsilon2_update, train_params.effective_batch_size)
    params_to_save['Noise'] = Noise

    Noise_test = generateIdLMNoise(train_params.image_size, 0, epsilon2_update, train_params.effective_batch_size)
    params_to_save['Noise_test'] = Noise_test

    # save params for testing
    with open(os.getcwd() + train_params.params_save_path, 'wb') as outf:
      pickle.dump(params_to_save, outf)
      print('params saved')

    ####################################
    print('start pretrain')
    start_time = time.time()
    lr_schedule_list = sorted(train_params.lr_schedule_pretrain.keys())
    attacks_and_benign = train_params.attacks + ['benign']
    # build zeros numpy arrays for accumulate grads
    accumu_pretrain_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in pretrain_grads_shapes]
    total_pretrain_loss_value = 0.0
    step = 0
    # pretrain loop
    while True:
      # if enough steps, break
      if step > train_params.pretrain_steps:
        break
      # add steps here so not forgot
      else:
        step += 1

      # manual schedule learning rate
      current_epoch = step // (train_params.epoch_steps)
      current_lr = train_params.lr_schedule_pretrain[get_lr(current_epoch, lr_schedule_list)]

      # benign and adv batch
      super_batch = TIN_data.train.next_super_batch(N_GPUS, ensemble=False, random=True)
      adv_super_batch = TIN_data.train.next_super_batch(N_GPUS, ensemble=False, random=True)

      # get pretrain grads
      pretrain_grads_save_np, _pretain_loss_value = sess.run([pretrain_grads_save, total_pretrain_loss], feed_dict={x_sb: super_batch[0], 
                                                                                                                    x_sb_adv: adv_super_batch[0], 
                                                                                                                    learning_rate: current_lr,
                                                                                                                    adv_noise: Noise_test, 
                                                                                                                    noise: Noise, 
                                                                                                                    FM_h: perturbFM_h})
      # accumulate grads
      for i in range(len(accumu_pretrain_grads)):
        accumu_pretrain_grads[i] = accumu_pretrain_grads[i] + pretrain_grads_save_np[i]
      
      # accumulate loss values
      total_pretrain_loss_value = total_pretrain_loss_value + _pretain_loss_value

      # use accumulated gradients to update variables
      if step % train_params.batch_multi == 0 and step > 0:
        # print('effective batch reached at step: {}, epoch: {}'.format(step, step / train_params.epoch_steps))
        # compute the average grads and build the feed dict
        pretrain_feed_dict = {}
        for i in range(len(accumu_pretrain_grads)):
          pretrain_feed_dict[pretrain_grads_placeholders[i]] = accumu_pretrain_grads[i] / train_params.batch_multi
        pretrain_feed_dict[learning_rate] = current_lr

        # run train ops by feeding the gradients
        sess.run(pretrain_op, feed_dict=pretrain_feed_dict)

        # get loss value
        avg_pretrain_loss_value = total_pretrain_loss_value / train_params.batch_multi

        # reset the average grads
        accumu_pretrain_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in pretrain_grads_shapes]
        total_pretrain_loss_value = 0.0

      # print loss
      if step % (1*train_params.epoch_steps) == 0 and step >= (1*train_params.epoch_steps):
        print('pretrain report at step: {}, epoch: {}'.format(step, step / train_params.epoch_steps))
        dt = time.time() - start_time
        avg_epoch_time = dt / (step / train_params.epoch_steps)
        print('epoch: {:.4f}, avg epoch time: {:.4f}, current_lr: {}'.format(step/train_params.epoch_steps, avg_epoch_time, current_lr), flush=True)
        print('pretrain_loss: {:.6f}'.format(avg_pretrain_loss_value))

    ####################################
    print('start train')
    start_time = time.time()
    lr_schedule_list = sorted(train_params.lr_schedule.keys())
    # train whole model
    # build zeros numpy arrays for accumulate grads
    accumu_pretrain_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in pretrain_grads_shapes]
    accumu_train_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in train_grads_shapes]
    total_pretrain_loss_value = 0.0
    total_train_loss_value = 0.0
    step = 0
    # train loop
    while True:
      # if enough steps, break
      if step > train_params.train_steps:
        break
      # add steps here so not forgot
      else:
        step += 1

      # compute the grads every step
      # random eps value for trianing
      d_eps = random.random()*train_params.random_eps_range

      # manual schedule learning rate
      current_epoch = step // (train_params.epoch_steps)
      current_lr = train_params.lr_schedule[get_lr(current_epoch, lr_schedule_list)]
      
      # benign and adv batch
      super_batch = TIN_data.train.next_super_batch(N_GPUS, ensemble=False, random=True)
      adv_super_batch = TIN_data.train.next_super_batch(N_GPUS, ensemble=False, random=True)

      # create adv samples
      super_batch_adv_images = sess.run(adv_tensors_concat, 
                                        feed_dict={x_sb:adv_super_batch[0], keep_prob:1.0,
                                                    adv_noise: Noise, mu_alpha:[d_eps]})   

      # get pretrain and train grads
      pretrain_grads_save_np, _pretain_loss_value = sess.run([pretrain_grads_save, total_pretrain_loss], feed_dict={x_sb: super_batch[0], 
                                                                                                                    x_sb_adv: super_batch_adv_images, 
                                                                                                                    learning_rate: current_lr,
                                                                                                                    adv_noise: Noise_test, 
                                                                                                                    noise: Noise, 
                                                                                                                    FM_h: perturbFM_h})
      train_grads_save_np, _train_loss_value = sess.run([train_grads_save, total_loss], feed_dict = {x_sb: super_batch[0], y_sb: super_batch[1],
                                                                  x_sb_adv: super_batch_adv_images, y_sb_adv: adv_super_batch[1],
                                                                  keep_prob: train_params.keep_prob, learning_rate: current_lr,
                                                                  noise: Noise, adv_noise: Noise_test, FM_h: perturbFM_h})

      # accumulate grads
      for i in range(len(accumu_pretrain_grads)):
        accumu_pretrain_grads[i] = accumu_pretrain_grads[i] + pretrain_grads_save_np[i]

      for i in range(len(accumu_train_grads)):
        accumu_train_grads[i] = accumu_train_grads[i] + train_grads_save_np[i]

      # accumulate loss values
      total_pretrain_loss_value = total_pretrain_loss_value + _pretain_loss_value
      total_train_loss_value = total_train_loss_value + _train_loss_value
      
      # use accumulated gradients to update variables
      if step % train_params.batch_multi == 0 and step > 0:
        # compute the average grads and build the feed dict
        pretrain_feed_dict = {}
        for i in range(len(accumu_pretrain_grads)):
          pretrain_feed_dict[pretrain_grads_placeholders[i]] = accumu_pretrain_grads[i] / train_params.batch_multi
        pretrain_feed_dict[learning_rate] = current_lr
        # pretrain_feed_dict[keep_prob] = 0.5

        train_feed_dict = {}
        for i in range(len(accumu_train_grads)):
          train_feed_dict[train_grads_placeholders[i]] = accumu_train_grads[i] / train_params.batch_multi
        train_feed_dict[learning_rate] = current_lr
        # train_feed_dict[keep_prob] = 0.5

        # run train ops
        sess.run(pretrain_op, feed_dict=pretrain_feed_dict)
        sess.run(train_op, feed_dict=train_feed_dict)

        # get loss value
        avg_pretrain_loss_value = total_pretrain_loss_value / train_params.batch_multi
        avg_train_loss_value = total_train_loss_value / train_params.batch_multi

        # reset the average grads
        accumu_pretrain_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in pretrain_grads_shapes]
        accumu_train_grads = [np.zeros(g_shape, dtype=np.float32) for g_shape in train_grads_shapes]
        total_pretrain_loss_value = 0.0
        total_train_loss_value = 0.0

      # print status every epoch
      if step % int(train_params.epoch_steps) == 0:
        dt = time.time() - start_time
        avg_epoch_time = dt / (step / train_params.epoch_steps)
        print('epoch: {:.4f}, avg epoch time: {:.4f}s, current_lr: {}'.format(step/train_params.epoch_steps, avg_epoch_time, current_lr), flush=True)

      # save model
      if step % int(train_params.epoch_steps) == 0 and int(step / train_params.epoch_steps) in train_params.epochs_to_save:
        print('saving model at epoch {}'.format(step / train_params.epoch_steps))
        checkpoint_path = os.path.join(os.getcwd() + train_params.check_point_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
      # testing during training
      if step % int(train_params.epoch_steps) == 0 and int(step / train_params.epoch_steps) in train_params.epochs_to_test:
        test_start = time.time()
        print('train test reported at step: {}, epoch: {}'.format(step, step / train_params.epoch_steps))
        dt = time.time() - start_time
        avg_epoch_time = dt / (step / train_params.epoch_steps)
        print('epoch: {:.4f}, avg epoch time: {:.4f}s, current_lr: {}'.format(step/train_params.epoch_steps, avg_epoch_time, current_lr), flush=True)
        print('pretrain_loss: {:.6f}, train_loss: {:.6f}'.format(avg_pretrain_loss_value, avg_train_loss_value))
        # print('output layer: \n\t{}'.format(output_layer_value))

        #===================adv samples=====================
        adv_acc_dict = {}
        robust_adv_acc_dict = {}
        robust_adv_utility_dict = {}
        log_str = ''
        # cover all test data
        for i in range(train_params.test_epochs):
          test_batch = TIN_data.test.next_batch(train_params.test_batch_size)
          # if more GPUs available, generate testing adv samples at once
          if N_AUX_GPUS > 1:
            adv_images_dict = sess.run(attack_tensor_testing_dict, feed_dict ={x_sb: test_batch[0], 
                                                                               adv_noise: Noise_test, 
                                                                               mu_alpha: [train_params.fgsm_eps],
                                                                               keep_prob: 1.0})
          else:
            adv_images_dict = {}
          # test for each attack
          for atk in attacks_and_benign:
            if atk not in adv_acc_dict:
              adv_acc_dict[atk] = 0.0
              robust_adv_acc_dict[atk] = 0.0
              robust_adv_utility_dict[atk] = 0.0
            if atk == 'benign':
              testing_img = test_batch[0]
            elif attack_switch[atk]:
              # if only one gpu available, generate adv samples in-place
              if atk not in adv_images_dict:
                adv_images_dict[atk] = sess.run(attack_tensor_testing_dict[atk], feed_dict ={x_sb:test_batch[0], 
                                                                                             adv_noise: Noise_test, 
                                                                                             mu_alpha:[train_params.fgsm_eps],
                                                                                             keep_prob: 1.0})
              testing_img = adv_images_dict[atk]
            else:
              continue
            ### PixelDP Robustness ###
            predictions_form_argmax = np.zeros([train_params.test_batch_size, train_params.num_classes])
            softmax_predictions = sess.run(y_softmax_test_concat, feed_dict={x_sb: testing_img, noise: Noise, FM_h: perturbFM_h, keep_prob: 1.0})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
            for n_draws in range(0, train_params.num_samples):
              _BenignLNoise = generateIdLMNoise(train_params.image_size, train_params.Delta2, epsilon2_update, train_params.effective_batch_size)
              _perturbFM_h = np.random.laplace(0.0, 2*train_params.Delta2/(epsilon2_update*train_params.effective_batch_size), 
                                              train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters)
              _perturbFM_h = np.reshape(_perturbFM_h, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
              for j in range(train_params.test_batch_size):
                pred = argmax_predictions[j]
                predictions_form_argmax[j, pred] += 1
              softmax_predictions = sess.run(y_softmax_test_concat, feed_dict={x_sb: testing_img, noise: (_BenignLNoise/10 + Noise), FM_h: perturbFM_h, keep_prob: 1.0}) * \
                sess.run(y_softmax_test_concat, feed_dict={x_sb: testing_img, noise: Noise, FM_h: (_perturbFM_h/10 + perturbFM_h), keep_prob: 1.0})
              argmax_predictions = np.argmax(softmax_predictions, axis=1)
            final_predictions = predictions_form_argmax
            is_correct = []
            is_robust = []
            for j in range(train_params.test_batch_size):
              is_correct.append(np.argmax(test_batch[1][j]) == np.argmax(final_predictions[j]))
              robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],
                                                                        eta=0.05, dp_attack_size=train_params.fgsm_eps, 
                                                                        dp_epsilon=train_params.dp_epsilon, dp_delta=0.05, 
                                                                        dp_mechanism='laplace') / dp_mult
              is_robust.append(robustness_from_argmax >= train_params.fgsm_eps)
            adv_acc_dict[atk] += np.sum(is_correct)*1.0/train_params.test_batch_size
            robust_adv_acc_dict[atk] += np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
            robust_adv_utility_dict[atk] += np.sum(is_robust)*1.0/train_params.test_batch_size
        ##############################
        # average all acc for whole test data
        for atk in attacks_and_benign:
          adv_acc_dict[atk] = adv_acc_dict[atk] / train_params.test_epochs
          robust_adv_acc_dict[atk] = robust_adv_acc_dict[atk] / train_params.test_epochs
          robust_adv_utility_dict[atk] = robust_adv_utility_dict[atk] / train_params.test_epochs
          # added robust prediction
          log_str += " {}: {:.6f} {:.6f} {:.6f} {:.6f}\n".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk] * robust_adv_utility_dict[atk])
        dt = time.time() - test_start
        print('testing time: {}'.format(dt))
        print(log_str, flush=True)
        print('*******************')

def get_lr(epoch, lr_list):
  for epoch_thres in lr_list:
    if epoch <= epoch_thres:
      return epoch_thres
  return lr_list[-1]


def load_pretrained_vars():
  pretrained_var_dict = {}
  with h5py.File('./resnet18_imagenet_1000_no_top.h5', 'r') as inf:
    for key in inf.keys():
      var_name_list = list(inf[key].keys())
      if len(var_name_list) > 0:
        var_name_list2 = list(inf[key][var_name_list[0]])
        if len(var_name_list2) > 0:
          for var_name in var_name_list2:
            pretrained_var_dict['/'.join([key, var_name_list[0], var_name])] = np.array(inf[key][var_name_list[0]][var_name])
  return pretrained_var_dict

class trainParams():
  def __init__(self):
    self.load_weights = True
    # only use a subset of the tinyimagenet
    self.num_classes = 30
    self.train_size = self.num_classes * 500 * 3
    self.train_epochs = 350
    # effective batch size multiplier
    self.batch_multi = 20
    if self.load_weights:
      self.pretrain_epochs = 20
    else:
      self.pretrain_epochs = 50
    self.batch_size = 225
    self.effective_batch_size = self.batch_size * self.batch_multi
    self.epoch_steps = int(self.train_size / self.batch_size)
    self.pretrain_steps = int(self.pretrain_epochs * self.epoch_steps)
    self.train_steps = int(self.train_epochs * self.epoch_steps)
    
    self.epochs_to_test = set([2,10,25,50,75,100,120,130,140,150,160,180,200,250,300])
    self.test_size = self.num_classes * 50
    self.test_batch_size = 300
    self.test_epochs = int(self.test_size / self.test_batch_size)
    self.n_ensemble = 3
    self.lr_schedule = {500:1e-2}
    self.lr_schedule_pretrain = {500:1e-2}
    self.image_size = 64
    self.image_channels = 3
    self.keep_prob = 0.5
    self.weight_table_path = './resnet18_weight_table_enc_layer.txt'
    self.epochs_to_save = set([25,50,75,100,120,130,140,150,160,180,200,250,300])
    self.check_point_dir = '/ckpts'
    self.params_save_path = '/SDP_params.pik'

    self.hk = 256
    self.enc_filters = 64
    self.enc_kernel_size = 7
    self.enc_stride = 2
    self.enc_h_size = math.ceil(self.image_size / self.enc_stride)

    self.dp_epsilon = 1.0
    self.Delta2 = 3 * (self.enc_h_size * self.enc_h_size+2) * (self.enc_kernel_size**2)
    self.gen_ratio = 3.0
    self.epsilon3 = 1.0*(1)
    self.epsilon2 = 1.0*(1 + 5*self.gen_ratio/5.0)
    self.total_eps = self.epsilon2 + self.epsilon3
    self.Delta3_benign = 2*self.hk
    self.Delta3_adv = 2*self.hk
    self.eps3_benign = self.epsilon3
    self.eps3_adv = self.epsilon3
    self.scale3_benign = self.Delta3_benign/(self.eps3_benign*self.effective_batch_size)
    self.scale3_adv = self.Delta3_adv/(self.eps3_adv*self.effective_batch_size)
    self.perturbFM = np.reshape(np.random.laplace(0.0, self.scale3_benign, self.hk*self.num_classes), [self.hk, self.num_classes])
    self.alpha = 1.5
    self.fgsm_eps = 0.05
    self.random_eps_range = 0.1
    self.iter_step_training = 10
    self.iter_step_testing = 100
    self.attacks = ['ifgsm','mim','madry']
    self.num_samples = 500


class resnetParams():
  def __init__(self):
    # parameters
    self.num_filters = 64
    self.resnet_size = 18
    self.block_sizes = [2, 2, 2, 2]
    self.block_strides=[1, 1, 2, 2]

    self.first_pool_size = None
    self.first_pool_stride = 2
    self.bottleneck = False
    self.resnet_version = 2
    self.data_format = _DATA_FORMAT
    self.dtype = tf.float32
    self.block_fn = _building_block_v2
    self.pre_activation = True # because of v2

def main(argv=None):  # pylint: disable=unused-argument
  train_params = trainParams()
  resnet_params = resnetParams()

  # load TinyImageNet data
  # change this accordingly
  data_dir = '/data/public/TinyImageNet/tiny-imagenet-200'
  # data_dir = './data'
  # pre-packed dataset with data augmentation
  filename = '/tinyimagenet_{}_classes_augment_{}_onehot.npz'.format(train_params.num_classes, 3)
  TIN_data = read_data_sets_parallel(data_dir, filename, train_params.batch_size, train_params.n_ensemble)
  # TIN_data = None

  # prepare the parameters to save
  params_to_save = {}
  params_to_save['train_params'] = train_params
  params_to_save['resnet_params'] = resnet_params

  PDP_resnet_with_pretrain_adv(TIN_data, resnet_params, train_params, params_to_save)

if __name__ == '__main__':
  tf.app.run()