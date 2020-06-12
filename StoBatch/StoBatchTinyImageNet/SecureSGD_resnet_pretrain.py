########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, Cleverhans
########################################################################

'''
Train SecureSGD with RESNET18, TinyImageNet
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from datetime import datetime
import time
import math
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import h5py
import pickle
import math
from math import sqrt

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans.model import Model

from SSGD_loss import lossDPSGD
import accountant
import robustnessGGaussian
from tinyimagenet_read import read_data_sets_parallel
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
def inference(inputs, dp_mult, keep_prob, pre_define_vars, resnet_params, train_params):
  # through first conv/auto-encoding layer
  # no relu and batch norm here as they are included in resnet
  with tf.variable_scope('enc_layer') as scope:
    inputs = fixed_padding(inputs, pre_define_vars['kernel1'].shape[0], _DATA_FORMAT)
    print_var('enc padding', inputs)
    inputs = tf.nn.conv2d(inputs, pre_define_vars['kernel1'], 
                          [1, train_params.enc_stride, train_params.enc_stride, 1], padding='VALID')
    inputs = inputs + dp_mult + pre_define_vars['biases1']
    # h_conv1 = tf.nn.relu(inputs)
    h_conv1 = inputs
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
  return inputs, h_conv1

# inference in training with perturbH
def test_inference(inputs, dp_mult, keep_prob, pre_define_vars, resnet_params, train_params):
  # through first conv/auto-encoding layer
  # no relu and batch norm here as they are included in resnet
  with tf.variable_scope('enc_layer') as scope:
    inputs = fixed_padding(inputs, pre_define_vars['kernel1'].shape[0], _DATA_FORMAT)
    print_var('enc padding', inputs)
    inputs = tf.nn.conv2d(inputs, pre_define_vars['kernel1'], 
                          [1, train_params.enc_stride, train_params.enc_stride, 1], padding='VALID')
    inputs = inputs + dp_mult + pre_define_vars['biases1']
    h_conv1 = inputs
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
  return inputs, h_conv1

def inference_test_output_probs(inputs, keep_prob, pre_define_vars, resnet_params, train_params):
  logits, _ = test_inference(inputs, 0, keep_prob, pre_define_vars, resnet_params, train_params)
  return tf.nn.softmax(logits)
          
class CustomCallableModelWrapper(Model):
  def __init__(self, callable_fn, output_layer, keep_prob, pre_define_vars, 
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
    self.keep_prob = keep_prob
    self.pre_define_vars = pre_define_vars
    self.resnet_params = resnet_params
    self.train_params = train_params

  def fprop(self, x, **kwargs):
    return {self.output_layer: self.callable_fn(x, self.keep_prob, self.pre_define_vars, 
                                                self.resnet_params, self.train_params)}

def PDP_SSGD_resnet_with_pretrain(TIN_data, resnet_params, train_params, params_to_save):
  # dict for encoding layer variables and output layer variables
  pre_define_vars = {}

  # list of variables to train
  train_vars = []

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

    shape     = kernel1.get_shape().as_list()
    w_t       = tf.reshape(kernel1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivityW = tf.reduce_max(sing_vals)
    dp_mult = train_params.attack_norm_bound * math.sqrt(2 * math.log(1.25 / train_params.dp_delta)) / train_params.dp_epsilon
    params_to_save['dp_mult'] = dp_mult
    
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
    x_test = tf.placeholder(tf.float32, [None,train_params.image_size,train_params.image_size,3], name='x_test')

    y_sb = tf.placeholder(tf.float32, [None, train_params.num_classes], name='y_sb') # input is the bunch of n_batchs (super batch)
    y_test = tf.placeholder(tf.float32, [None, train_params.num_classes], name='y_test')

    noise = tf.placeholder(tf.float32, [None, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters], name='noise') # one time

    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

    # list of grads for each GPU
    tower_pretrain_grads = []
    tower_train_grads = []
    all_train_loss = []

    # optimizers
    train_opt = tf.train.GradientDescentOptimizer(learning_rate)

    with tf.device('/gpu:0'):     
      # the model
      y_logits, h_conv1 = inference(x_sb, train_params.attack_norm_bound * noise, keep_prob, pre_define_vars, resnet_params, train_params)
      y_softmax = tf.nn.softmax(y_logits)

      # loss
      with tf.device('/cpu:0'):
        loss = lossDPSGD(y_logits, y_sb)
      # get total loss tensor for train ops
      total_loss = tf.reduce_sum(loss)

      # noise redistribution parameters
      grad, = tf.gradients(loss, h_conv1)
      print(loss)
      print(h_conv1)
      print(grad)
      normalized_grad = tf.sign(grad)
      normalized_grad = tf.stop_gradient(normalized_grad)
      normalized_grad_r = tf.abs(tf.reduce_mean(normalized_grad, axis = (0)))**2
      sum_r = tf.reduce_sum(normalized_grad_r, axis = (0,1,2), keepdims=False)
      normalized_grad_r = train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters*normalized_grad_r/sum_r
      shape_grad = normalized_grad_r.get_shape().as_list()
      grad_t = tf.reshape(normalized_grad_r, [-1, shape_grad[-1]])
      g = tf.transpose(grad_t)
      sing_g_vals = tf.svd(g, compute_uv=False)
      sensitivity_2 = tf.reduce_max(sing_g_vals)

      y_logits_test, _ = test_inference(x_sb, train_params.attack_norm_bound * noise, keep_prob, pre_define_vars, resnet_params, train_params)
      y_softmax_test = tf.nn.softmax(y_logits_test)
    correct_prediction = tf.equal(tf.argmax(y_logits_test, 1), tf.argmax(y_sb, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    priv_accountant = accountant.GaussianMomentsAccountant(train_params.train_size)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], train_params.sigma, train_params.batch_size)
    
    # print all variables
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    all_vars = tf.global_variables()
    print_var_list('all vars', all_vars)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # add selected vars into list
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
          'bias' in var.name or
          'gamma' in var.name or
          'beta' in var.name):
        if var not in train_vars:
          train_vars.append(var)

    print_var_list('train_vars', train_vars)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    # op for compute grads
    with tf.device('/gpu:0'):
      # get all update_ops (updates of moving averageand std) for batch normalizations
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      print_op_list('update ops', update_ops)

      # compute each gradient and apply clip and noise
      train_grads = []
      for var in train_vars:
        g = tf.clip_by_norm(tf.gradients(loss, var)[0], train_params.clip_bound)
        g = g + tf.random_normal(shape=tf.shape(g), mean=0.0, stddev = train_params.sigma * (train_params.sensitivity**2), dtype=tf.float32)
        train_grads.append((g, var))
    
    # now the train and pretrain ops do not come with update_ops
    with tf.control_dependencies(update_ops):
      train_op = train_opt.apply_gradients(train_grads, global_step=global_step)

    ######################################

    # Create a saver.
    saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)
    
    # start a session with memory growth
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("session created")


    # init op
    print('initialize_all_variables')
    init = tf.initialize_all_variables()
    sess.run(init)

    if train_params.load_weights:
      # load pretrained variables
      # load variable convert table
      tgt_var_name_dict = {}
      with open(train_params.weight_table_path, 'r', encoding='utf-8') as inf:
        lines = inf.readlines()
        for line in lines:
          var_names = line.strip().split(' ')
          if var_names[1] == 'NONE':
            continue
          else:
            tgt_var_name_dict[var_names[0]] = var_names[1]

      # load variables
      pretrained_var_dict = load_pretrained_vars()

      # load pre-trained vars
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
    # init noise

    s = math.log(sqrt(2.0/math.pi)*1e+5)
    sigmaEGM = sqrt(2.0)*1.0*(sqrt(s) + sqrt(s+train_params.dp_epsilon))/(2.0*train_params.dp_epsilon)
    params_to_save['sigmaEGM'] = sigmaEGM
    
    __noiseE = np.random.normal(0.0, sigmaEGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
    __noiseE = np.reshape(__noiseE, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
    params_to_save['__noiseE'] = __noiseE
    
    _batch = TIN_data.train.next_super_batch(5, ensemble=False, random=True)   
    grad_redis = sess.run([normalized_grad_r], feed_dict = {x_sb: _batch[0], y_sb: _batch[1], noise: __noiseE*0, keep_prob:1.0})
    params_to_save['grad_redis'] = grad_redis
    
    _sensitivity_2 = sess.run([sensitivity_2], feed_dict={x_sb: _batch[0], y_sb: _batch[1], noise: __noiseE*0, keep_prob:1.0})
    params_to_save['_sensitivity_2'] = _sensitivity_2
    
    _sensitivityW = sess.run(sensitivityW)
    params_to_save['_sensitivityW'] = _sensitivityW

    Delta_redis = _sensitivityW/sqrt(_sensitivity_2[0])
    params_to_save['Delta_redis'] = Delta_redis

    sigmaHGM = sqrt(2.0)*Delta_redis*(sqrt(s) + sqrt(s+train_params.dp_epsilon))/(2.0*train_params.dp_epsilon)
    params_to_save['sigmaHGM'] = sigmaHGM

    __noiseH = np.random.normal(0.0, sigmaHGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
    __noiseH = np.reshape(__noiseH, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])*grad_redis
    params_to_save['__noiseH'] = __noiseH

    __noise_zero = np.random.normal(0.0, 0, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
    __noise_zero = np.reshape(__noise_zero, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
    params_to_save['__noise_zero'] = __noise_zero

    # save params
    with open(os.getcwd() + train_params.params_save_path, 'wb') as outf:
      pickle.dump(params_to_save, outf)
      print('params saved')

    ####################################
    
    ####################################
    print('start train')
    start_time = time.time()
    lr_schedule_list = sorted(train_params.lr_schedule.keys())
    # train whole model
    step = 0
    while True:
      # if enough steps, break
      if step > train_params.train_steps:
        break
      # add steps here so not forgot
      else:
        step += 1

      # manual schedule learning rate
      current_epoch = step // (train_params.epoch_steps)
      current_lr = train_params.lr_schedule[get_lr(current_epoch, lr_schedule_list)]
      
      # benign and adv batch
      super_batch = TIN_data.train.next_super_batch(1, ensemble=False, random=True)   

      _, loss_value = sess.run([train_op, loss], feed_dict = {x_sb: super_batch[0], y_sb: super_batch[1], noise: (__noiseE + __noiseH)/2, keep_prob:train_params.keep_prob, learning_rate:current_lr})

      # check with the privacy budget
      sess.run([privacy_accum_op])
      spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=train_params.target_eps)
      if step % int(train_params.epoch_steps) == 0:
        print(spent_eps_deltas)
      _break = False
      for _eps, _delta in spent_eps_deltas:
        if _delta >= train_params.delta:
          _break = True
          break
      if _break == True:
        print('budget all spent, stop training')
        break

      # print status every epoch
      if step % int(train_params.epoch_steps) == 0:
        dt = time.time() - start_time
        avg_epoch_time = dt / (step / train_params.epoch_steps)
        print('epoch: {:.4f}, avg epoch time: {:.4f}s, current_lr: {}'.format(step/train_params.epoch_steps, avg_epoch_time, current_lr), flush=True)
        print('train_loss: {:.6f}'.format(loss_value))

      # save model
      if step % int(train_params.epoch_steps) == 0 and int(step / train_params.epoch_steps) in train_params.epochs_to_save:
        print('saving model at epoch {}'.format(step / train_params.epoch_steps))
        checkpoint_path = os.path.join(os.getcwd() + train_params.check_point_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
      # for testing, I use basic inference procedure and accuracy measure on only benign data
      # if step % int(train_params.epoch_steps) == 0 and int(step / train_params.epoch_steps) in train_params.epochs_to_test:
      if step % int(10*train_params.epoch_steps) == 0 and step > 0:
        test_start = time.time()
        print('train test reported at step: {}, epoch: {}'.format(step, step / train_params.epoch_steps))
        dt = time.time() - start_time
        avg_epoch_time = dt / (step / train_params.epoch_steps)
        print('epoch: {:.4f}, avg epoch time: {:.4f}s, current_lr: {}'.format(step/train_params.epoch_steps, avg_epoch_time, current_lr), flush=True)
        print('train_loss: {:.6f}'.format(loss_value))
        # print('output layer: \n\t{}'.format(output_layer_value))

        #===================adv samples=====================
        adv_acc = 0.0
        robust_adv_acc = 0.0
        robust_adv_utility = 0.0
        log_str = ''
        # cover all test data
        for i in range(train_params.test_epochs):
          # start testing
          test_batch = TIN_data.test.next_batch(train_params.test_batch_size)
          ### PixelDP Robustness ###
          predictions_form_argmax = np.zeros([train_params.test_batch_size, train_params.num_classes])
          softmax_predictions = sess.run(y_softmax_test, feed_dict={x_sb: test_batch[0], noise: (__noiseE + __noiseH)/2, keep_prob:1.0})
          argmax_predictions = np.argmax(softmax_predictions, axis=1)
          for n_draws in range(0, train_params.num_samples):
            _noiseE = np.random.normal(0.0, sigmaEGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
            _noiseE = np.reshape(_noiseE, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
            _noise = np.random.normal(0.0, sigmaHGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
            _noise = np.reshape(_noise, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])*grad_redis
            for j in range(train_params.test_batch_size):
              pred = argmax_predictions[j]
              predictions_form_argmax[j, pred] += 1
            softmax_predictions = sess.run(y_softmax_test, feed_dict={x_sb: test_batch[0], noise: (__noiseE + __noiseH)/2 + (_noiseE + _noise)/4, keep_prob:1.0})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
          final_predictions = predictions_form_argmax
          is_correct = []
          is_robust = []
          for j in range(train_params.test_batch_size):
            is_correct.append(np.argmax(test_batch[1][j]) == np.argmax(final_predictions[j]))
            robustness_from_argmax = robustnessGGaussian.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=train_params.fgsm_eps, dp_epsilon=train_params.dp_epsilon, dp_delta=0.05, dp_mechanism='gaussian') / dp_mult
            is_robust.append(robustness_from_argmax >= train_params.fgsm_eps)
          adv_acc += np.sum(is_correct)*1.0/train_params.test_batch_size
          robust_adv_acc += np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
          robust_adv_utility += np.sum(is_robust)*1.0/train_params.test_batch_size
        ##############################
        # average all acc for whole test data
        adv_acc = adv_acc / train_params.test_epochs
        robust_adv_acc = robust_adv_acc / train_params.test_epochs
        robust_adv_utility = robust_adv_utility / train_params.test_epochs
        log_str += " {}: {:.6f} {:.6f} {:.6f} {:.6f}\n".format('benign', adv_acc, robust_adv_acc, robust_adv_utility, robust_adv_acc * robust_adv_utility)
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
    self.num_classes = 30
    self.train_size = self.num_classes * 500 * 3
    self.train_epochs = 200
    # effective batch size multiplier
    self.batch_size = 150
    self.epoch_steps = int(self.train_size / self.batch_size)
    self.train_steps = int(self.train_epochs * self.epoch_steps)
    
    self.epochs_to_test = set([2,5,10,15,20,25,40,50,60,65,70,75,80,82,84,85,86,90,100,110,120,130,140,150])
    self.test_size = self.num_classes * 50
    self.test_batch_size = 300
    self.test_epochs = int(self.test_size / self.test_batch_size)
    self.n_ensemble = 3
    self.lr_schedule = {500:1e-2}

    self.image_size = 64
    self.image_channels = 3
    self.keep_prob = 0.5
    self.weight_table_path = './resnet18_weight_table_enc_layer.txt'
    self.epochs_to_save = set([10,20,30,40,50,60,65,70,75,80,82,84,85,86,90,100,110,120,130,140,150])
    self.check_point_dir = '/ckpts'
    self.params_save_path = self.check_point_dir + '/SSGD_params.pik'

    self.hk = 256
    self.enc_filters = 64
    self.enc_kernel_size = 7
    self.enc_stride = 2
    self.enc_h_size = math.ceil(self.image_size / self.enc_stride)

    self.iter_step_testing = 100
    self.attacks = ['ifgsm','mim','madry']
    self.num_samples = 100

    # SGD params
    self.dp_delta = 0.05
    self.dp_epsilon = 4.0
    self.attack_norm_bound = 0.2
    self.clip_bound = 0.01 # 'the clip bound of the gradients'
    self.sigma = 1.5 # 'sigma'
    self.delta = 1e-5 # 'delta'
    self.sensitivity = self.clip_bound #adjacency matrix with one more tuple
    self.target_eps = [2.0]
    self.fgsm_eps = 0.001 # placeholder here

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
  data_dir = '/data/public/TinyImageNet/tiny-imagenet-200'
  # pre-packed dataset with data augmentation
  filename = '/tinyimagenet_{}_classes_augment_{}_onehot.npz'.format(train_params.num_classes, 3)
  TIN_data = read_data_sets_parallel(data_dir, filename, train_params.batch_size, train_params.n_ensemble)

  # prepare the parameters to save
  params_to_save = {}
  params_to_save['train_params'] = train_params
  params_to_save['resnet_params'] = resnet_params

  PDP_SSGD_resnet_with_pretrain(TIN_data, resnet_params, train_params, params_to_save)

if __name__ == '__main__':
  tf.app.run()