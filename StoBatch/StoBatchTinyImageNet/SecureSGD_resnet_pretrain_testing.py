########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, Cleverhans
########################################################################

"""
Testing SecureSGD with RESNET18, TinyImageNet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

from pretrain_resnet_SSGD import trainParams, resnetParams

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

from mlp import EncLayer
import robustnessGGaussian
from tinyimagenet_read import read_data_sets_parallel
from resnet_utils import batch_norm, fixed_padding, _building_block_v2, resnet18_builder_mod
from utils import _fc, print_var, print_var_list, print_op_list


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

def SSGD_resnet_testing(TIN_data, resnet_params, train_params, test_params, all_params):
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

    dp_mult = all_params['dp_mult']
    
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

    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')


    with tf.device('/gpu:0'):
      # the model for testing
      y_logits_test, _ = test_inference(x_sb, train_params.attack_norm_bound * noise, keep_prob, pre_define_vars, resnet_params, train_params)
      y_softmax_test = tf.nn.softmax(y_logits_test)
    correct_prediction = tf.equal(tf.argmax(y_logits_test, 1), tf.argmax(y_sb, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
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

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print_var_list('train_vars', train_vars)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    ######################################

    # Create a saver.
    saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)
    
    # start a session with memory growth
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("session created")

    # list all checkpoints in ckpt_path
    checkpoint_path_read = os.path.join(os.getcwd() + test_params.check_point_dir)
    ckpts = tf.train.get_checkpoint_state(checkpoint_path_read)
    print(ckpts)
    # find the ckpt we need to load and load it
    for ckpt in ckpts.all_model_checkpoint_paths:
      # print(ckpt)
      ckpt_step = int(ckpt.split('-')[-1])
      if ckpt_step == test_params.step_to_load:
        saver.restore(sess, ckpt)
        print('model loaded from {}'.format(ckpt))

    # #######################################
   
    # # setup all attacks
    attack_switch = {'fgsm':False, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}
    
    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_output_probs, output_layer='probs', 
                                                keep_prob=keep_prob, pre_define_vars=pre_define_vars, 
                                                resnet_params=resnet_params, train_params=train_params)
    attack_tensor_testing_dict = {}

    # define each attack method's tensor
    mu_alpha = tf.placeholder(tf.float32, [1])
      
    # Iterative FGSM (BasicIterativeMethod/ProjectedGradientMethod with no random init)
    with tf.device('/gpu:0'):
      if attack_switch['ifgsm']:
        print('creating attack tensor of BasicIterativeMethod')
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        attack_tensor_testing_dict['ifgsm'] = ifgsm_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, clip_min=-1.0, clip_max=1.0)

    # MomentumIterativeMethod
    with tf.device('/gpu:0'):
      if attack_switch['mim']:
        print('creating attack tensor of MomentumIterativeMethod')
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        attack_tensor_testing_dict['mim'] = mim_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
      
    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    with tf.device('/gpu:0'):
      if attack_switch['madry']:
        print('creating attack tensor of MadryEtAl')
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        attack_tensor_testing_dict['madry'] = madry_obj.generate(x=x_sb, eps=mu_alpha, eps_iter=mu_alpha/train_params.iter_step_testing, nb_iter=train_params.iter_step_testing, clip_min=-1.0, clip_max=1.0)

    # #######################################

    sigmaEGM = all_params['sigmaEGM']
    
    __noiseE = all_params['__noiseE']
    
    grad_redis = all_params['grad_redis']
    
    _sensitivity_2 = all_params['_sensitivity_2']
    
    _sensitivityW = all_params['_sensitivityW']

    Delta_redis = all_params['Delta_redis']

    sigmaHGM = all_params['sigmaHGM']

    __noiseH = all_params['__noiseH']

    __noise_zero = all_params['__noise_zero']

    ####################################
    
    ####################################
    print('start testing')
    start_time = time.time()
    log_file_path = os.getcwd() +  test_params.log_file_path
    log_file = open(log_file_path, 'a', encoding='utf-8')
    attacks_and_benign = test_params.attacks + ['benign']
    #===================adv samples=====================
    # for each eps setting
    for fgsm_eps in test_params.fgsm_eps_list:
      adv_acc_dict = {}
      robust_adv_acc_dict = {}
      robust_adv_utility_dict = {}
      log_str = ''
      eps_start_time = time.time()
      # cover all test data
      for i in range(test_params.test_epochs):
        test_batch = TIN_data.test.next_batch(test_params.test_batch_size)
        adv_images_dict = {}
        # test for each attack
        for atk in attacks_and_benign:
          start_time = time.time()
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
                                                                                            mu_alpha:[fgsm_eps],
                                                                                            keep_prob: 1.0})
            testing_img = adv_images_dict[atk]
          else:
            continue
          print('adv gen time: {}s'.format(time.time() - start_time))
          start_time = time.time()

          ### PixelDP Robustness ###
          predictions_form_argmax = np.zeros([test_params.test_batch_size, train_params.num_classes])
          softmax_predictions = sess.run(y_softmax_test, feed_dict={x_sb: testing_img, noise: (__noiseE + __noiseH)/2, keep_prob:1.0})
          argmax_predictions = np.argmax(softmax_predictions, axis=1)
          for n_draws in range(1, test_params.num_samples+1):
            if n_draws % 100 == 0:
              print('current draws: {}, avg draw time: {}s'.format(n_draws, (time.time()-start_time) / n_draws))
            _noiseE = np.random.normal(0.0, sigmaEGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
            _noiseE = np.reshape(_noiseE, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])
            _noise = np.random.normal(0.0, sigmaHGM**2, train_params.enc_h_size*train_params.enc_h_size*train_params.enc_filters).astype(np.float32)
            _noise = np.reshape(_noise, [-1, train_params.enc_h_size, train_params.enc_h_size, train_params.enc_filters])*grad_redis
            for j in range(test_params.test_batch_size):
              pred = argmax_predictions[j]
              predictions_form_argmax[j, pred] += 1
            softmax_predictions = sess.run(y_softmax_test, feed_dict={x_sb: testing_img, noise: (__noiseE + __noiseH)/2 + (_noiseE + _noise)/4, keep_prob:1.0})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
          final_predictions = predictions_form_argmax
          is_correct = []
          is_robust = []
          for j in range(test_params.test_batch_size):
            is_correct.append(np.argmax(test_batch[1][j]) == np.argmax(final_predictions[j]))
            robustness_from_argmax = robustnessGGaussian.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=train_params.dp_epsilon, dp_delta=0.05, dp_mechanism='gaussian') / dp_mult
            is_robust.append(robustness_from_argmax >= fgsm_eps)
          adv_acc_dict[atk] += np.sum(is_correct)*1.0/test_params.test_batch_size
          robust_adv_acc_dict[atk] += np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
          robust_adv_utility_dict[atk] += np.sum(is_robust)*1.0/test_params.test_batch_size
          
          dt = time.time() - start_time
          print('atk test time: {}s'.format(dt), flush=True)
      ##############################
      # average all acc for whole test data
      log_str += datetime.now().strftime("%Y-%m-%d_%H:%M:%S\n")
      log_str += 'model trained epoch: {}\n'.format(test_params.epoch_to_test)
      log_str += 'fgsm_eps: {}\n'.format(fgsm_eps)
      log_str += 'iter_step_testing: {}\n'.format(test_params.iter_step_testing)
      log_str += 'num_samples: {}\n'.format(test_params.num_samples)
      for atk in attacks_and_benign:
        adv_acc_dict[atk] = adv_acc_dict[atk] / test_params.test_epochs
        robust_adv_acc_dict[atk] = robust_adv_acc_dict[atk] / test_params.test_epochs
        robust_adv_utility_dict[atk] = robust_adv_utility_dict[atk] / test_params.test_epochs
        # added robust prediction
        log_str += " {}: {:.6f} {:.6f} {:.6f} {:.6f}\n".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk] * robust_adv_utility_dict[atk])
      dt = time.time() - eps_start_time
      print('total test time: {}s'.format(dt), flush=True)
      print(log_str, flush=True)
      print('*******************')

      log_file.write(log_str)
      log_file.write('*******************\n')
      log_file.flush()

      dt = time.time() - start_time
    log_file.close()

def get_lr(epoch, lr_list):
  for epoch_thres in lr_list:
    if epoch <= epoch_thres:
      return epoch_thres
  return lr_list[-1]

class testParams():
  def __init__(self, train_params):
    self.epoch_to_test = 82
    self.step_to_load = int(self.epoch_to_test * train_params.epoch_steps)
    self.test_size = train_params.num_classes * 50
    self.test_batch_size = 300
    self.test_epochs = int(self.test_size / self.test_batch_size)
    self.n_ensemble = 3

    self.check_point_dir = '/models/tinyimagenet_SSGD_lr_1'
    # self.fgsm_eps_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    self.fgsm_eps_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    self.iter_step_testing = 100 #[100, 500, 1000, 2000]
    self.attacks = ['ifgsm','mim','madry']
    self.num_samples = 1000
    self.log_file_path = '/results/pretrain_resnet_SSGD_lr_1_epoch_{}_steps_{}.txt'.format(self.epoch_to_test, self.iter_step_testing)


def main(argv=None):  # pylint: disable=unused-argument
  # load params
  params_save_path = '/SSGD_params.pik'
  with open(os.getcwd() + params_save_path, 'rb') as inf:
    all_params = pickle.load(inf)
  train_params = all_params['train_params']
  resnet_params = all_params['resnet_params']
  test_params = testParams(train_params)

  # load TinyImageNet data
  data_dir = '/data/public/TinyImageNet/tiny-imagenet-200'
  # filename = '/tinyimagenet_{}_classes_augment_onehot.npz'.format(train_params.num_classes)
  filename = '/tinyimagenet_{}_classes_augment_{}_onehot.npz'.format(train_params.num_classes, 3)
  TIN_data = read_data_sets_parallel(data_dir, filename, train_params.batch_size, train_params.n_ensemble)

  SSGD_resnet_testing(TIN_data, resnet_params, train_params, test_params, all_params)

if __name__ == '__main__':
  tf.app.run()