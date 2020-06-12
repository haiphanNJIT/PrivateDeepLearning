########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow, Cleverhans
########################################################################

"""
Testing Scaleable DP on CIFAR10
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
GPU_INDEX = 3
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(GPU_INDEX))

from datetime import datetime
import time
import math
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import itertools
import re

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
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
image_size = 28;
padding = 4;
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

def test(cifar10_data, checkpoint_path, epochs, L, learning_rate, scale3, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, parameter_dict, testing_step):
  # logfile.write("fgsm_eps \t %g, LR \t %g, alpha \t %d , epsilon \t %d \n"%(fgsm_eps, learning_rate, alpha, total_eps))
    
  """Train CIFAR-10 for a number of steps."""
  # make sure variables are placed on cpu
  # TODO: for AWS version, check if put variables on GPU will be better
  with tf.Graph().as_default():
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

    x = tf.placeholder(tf.float32, [None,image_size,image_size,3]) # input is the bunch of n_batchs

    y = tf.placeholder(tf.float32, [None, 10]) # input is the bunch of n_batchs

    # benign conv output
    bi = 0
    x_image = x + noise
    # with tf.device('/gpu:0'):
    y_conv = inference(x_image, FM_h, params, scopes, training=True, bn_index=bi)
    softmax_y_conv = tf.nn.softmax(y_conv)

    # start a session with memory growth
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    print("session created")
    
    dp_epsilon=1.0

    epsilon2_update = parameter_dict['epsilon2_update']

    delta_r = parameter_dict['delta_r']

    _sensitivityW = parameter_dict['_sensitivityW']

    delta_h = parameter_dict['delta_h']

    dp_mult = parameter_dict['dp_mult']

    # ============== attacks ================
    iter_step_training = parameter_dict['iter_step_training']


    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_input_probs, output_layer='probs', params=params, scopes=scopes, image_size=image_size, adv_noise=adv_noise)
    attack_tensor_dict = {}

    # define each attack method's tensor
    mu_alpha = tf.placeholder(tf.float32, [1])


    # build each attack
    for atk in attacks:
      print('building attack {} tensors'.format(atk))
      # for each gpu assign to each attack
      if atk == 'ifgsm':
        ifgsm_obj = BasicIterativeMethod(model=ch_model_probs, sess=sess)
        attack_tensor_dict[atk] = ifgsm_obj.generate(x=x, 
                                                      eps=mu_alpha, 
                                                      eps_iter=mu_alpha/testing_step, 
                                                      nb_iter=testing_step, 
                                                      clip_min=-1.0, clip_max=1.0)
      elif atk == 'mim':
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        attack_tensor_dict[atk] = mim_obj.generate(x=x, 
                                                    eps=mu_alpha, 
                                                    eps_iter=mu_alpha/testing_step, 
                                                    nb_iter=testing_step, 
                                                    decay_factor=1.0, 
                                                    clip_min=-1.0, clip_max=1.0)
      elif atk == 'madry':
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        attack_tensor_dict[atk] = madry_obj.generate(x=x,
                                                      eps=mu_alpha, 
                                                      eps_iter=mu_alpha/testing_step, 
                                                      nb_iter=testing_step, 
                                                      clip_min=-1.0, clip_max=1.0)

    # Create a saver and load checkpoint
    saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=1000)
    saver.restore(sess, checkpoint_path)
    
    T = int(int(math.ceil(D/L))*epochs + 1) # number of steps
    
    step_for_epoch = parameter_dict['step_for_epoch'] #number of steps for one epoch
    
    # load some fixed noise
    perturbH_test = parameter_dict['perturbH_test']
    
    perturbFM_h = parameter_dict['perturbFM_h']
    
    Noise = parameter_dict['Noise']
    
    Noise_test = parameter_dict['Noise_test']

    # test on testing dataset
    adv_acc_dict = {}
    robust_adv_acc_dict = {}
    robust_adv_utility_dict = {}
    test_batch_size = 5000
    n_draw = 1000
    begin_time = time.time()
    print('on testing set')
    print('test_batch_size: {}'.format(test_batch_size))
    print('testing iteration: {}'.format(testing_step))
    print('testing n_draw: {}'.format(n_draw))
    atk_index = -1
    for _ in [0,1]:
      for atk in attacks:
          print(atk)
          if atk not in adv_acc_dict:
              adv_acc_dict[atk] = -1
              robust_adv_acc_dict[atk] = -1
              robust_adv_utility_dict[atk] = -1
          # generate test samples
          test_batch = cifar10_data.test.next_batch(test_batch_size)
          adv_images = sess.run(attack_tensor_dict[atk], feed_dict={x:test_batch[0], 
                                                                        adv_noise: Noise_test, 
                                                                        mu_alpha:[fgsm_eps]})
          print("Done adversarial examples")
          ### PixelDP Robustness ###
          predictions_form_argmax = np.zeros([test_batch_size, 10])
          softmax_predictions = sess.run(softmax_y_conv, feed_dict={x: adv_images, 
                                                                    noise: Noise, 
                                                                    FM_h: perturbFM_h})
          argmax_predictions = np.argmax(softmax_predictions, axis=1)
          argmax_labels = np.argmax(test_batch[1], axis=1)
          print('labels')
          print(argmax_labels[0:100])
          print('init predictions')
          print(argmax_predictions[0:100])
          for _n_draws in range(0, n_draw):
              _BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L)
              _perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*128)
              _perturbFM_h = np.reshape(_perturbFM_h, [-1, 14, 14, 128])
              if _n_draws == 500 or _n_draws == 1000:
                  print("n_draws = 500/1000")
                  print('time passed: {}s'.format(time.time() - begin_time))
              for j in range(test_batch_size):
                  pred = argmax_predictions[j]
                  predictions_form_argmax[j, pred] += 1
              softmax_predictions = sess.run(softmax_y_conv, 
                                              feed_dict={x: adv_images, 
                                                        noise: (_BenignLNoise/10 + Noise), 
                                                        FM_h: perturbFM_h}) * sess.run(softmax_y_conv, 
                                                                                      feed_dict={x: adv_images, 
                                                                                                noise: Noise, 
                                                                                                FM_h: (_perturbFM_h/10 + perturbFM_h)})
              argmax_predictions = np.argmax(softmax_predictions, axis=1)
          final_predictions = predictions_form_argmax
          print('final predictions')
          print(np.argmax(final_predictions, axis=1)[0:100])
          is_correct = []
          is_robust = []
          for j in range(test_batch_size):
              is_correct.append(np.argmax(test_batch[1][j]) == np.argmax(final_predictions[j]))
              robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],
                                                                          eta=0.05, dp_attack_size=fgsm_eps, 
                                                                          dp_epsilon=dp_epsilon, dp_delta=0.05, 
                                                                          dp_mechanism='laplace') / dp_mult
              is_robust.append(robustness_from_argmax >= fgsm_eps)
          adv_acc_dict[atk] = np.sum(is_correct)*1.0/test_batch_size
          robust_adv_acc_dict[atk] = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
          robust_adv_utility_dict[atk] = np.sum(is_robust)*1.0/test_batch_size
          ##############################
      log_str = 'testing, eps: {}; steps: {};'.format(fgsm_eps, testing_step)
      for atk in attacks:
          log_str += "\n{}: {:.4f} {:.4f} {:.4f} {:.4f} ".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk] * robust_adv_utility_dict[atk])
      print(log_str, flush=True)
  tf.reset_default_graph()


def read_parameter_dict(p):
    begin_flag = False
    raw_data = []
    with open(p, 'r', encoding='utf-8') as inf:
        for line in inf:
            if line.strip() == '*=*=*=*=*':
                if begin_flag:
                    break
                elif not begin_flag:
                    begin_flag = True
            else:
                if begin_flag:
                    raw_data.append(line.strip())
    data_str = ''.join(raw_data)
    pattern = re.compile('float32')
    data_str = re.sub('float32', "'float32'", data_str)
    parameter_dict = eval(data_str)
    print(parameter_dict.keys())
    return parameter_dict


def main(argv=None):  # pylint: disable=unused-argument
  # load and setup parameters
  log_path = "GPU_4_NO_TESTING_50_3.txt"
  parameter_dict = read_parameter_dict(log_path)

  train_size = parameter_dict['train_size']
  # print('train_size: {}'.format(train_size))
  
  target_batch_size = parameter_dict['target_batch_size']
  # print('target_batch_size: {}'.format(target_batch_size))
  
  n_batchs = parameter_dict['n_batchs']
  # print('n_batchs: {}'.format(n_batchs))
  
  n_ensemble = parameter_dict['n_ensemble']
  # print('n_ensemble: {}'.format(n_ensemble))

  stair_L = parameter_dict['L'] #batch size # stair_L = [1851] #batch size
  # print('stair_L: {}'.format(stair_L))

  stair_epochs = parameter_dict['epochs'] #number of epoch for one iteration
  print('stair_epochs: {}'.format(stair_epochs))

  stair_iters = [1] #number of staired iterations

  stair_learning_rate = parameter_dict['learning_rate'] #learning rate
  print('stair_learning_rate: {}'.format(stair_learning_rate))

  gen_ratio = 0 # [0, 2, 4, 6, 8]
  
  epsilon3 = parameter_dict['epsilon3'] #epsilon for the last hidden layer, for total epsilon = 10, we used [epsilon3, epsilon2] = [4, 6]
  print('epsilon3: {}'.format(epsilon3))
  
  epsilon2 = parameter_dict['epsilon2'] #epsilon for the first hidden layer
  print('epsilon2: {}'.format(epsilon2))

  total_eps = parameter_dict['total_eps'] # stay fixed
  print('total_eps: {}'.format(total_eps))

  
  # cifar10_data = cifar10_read.read_data_sets("cifar-10-batches-bin/", one_hot = True)
  cifar10_data = cifar10_read.read_data_sets_parallel("cifar-10-batches-bin/", target_batch_size, n_ensemble)
  print('Done getting images')

  Delta2 = parameter_dict['Delta2'] #global sensitivity for the first hidden layer
  print('Delta2: {}'.format(Delta2))

  alpha = parameter_dict['alpha']
  print('alpha: {}'.format(alpha))

  eps2_ratio = parameter_dict['eps2_ratio']
  print('eps2_ratio: {}'.format(eps2_ratio))

  Delta3_adv = parameter_dict['Delta3_adv'] #10*(hk + 1/4 * hk**2); #global sensitivity for the output layer
  print('Delta3_adv: {}'.format(Delta3_adv))

  Delta3_benign = parameter_dict['Delta3_benign'] #10*(hk); #global sensitivity for the output layer
  print('Delta3_benign: {}'.format(Delta3_benign))

  eps3_benign = parameter_dict['eps3_benign']
  print('eps3_benign: {}'.format(eps3_benign))

  eps3_adv = parameter_dict['eps3_adv']
  print('eps3_adv: {}'.format(eps3_adv))

  # fgsm_eps = parameter_dict['fgsm_eps']
  # print('fgsm_eps: {}'.format(fgsm_eps))

  scale3_benign = parameter_dict['scale3_benign']
  scale3_adv = parameter_dict['scale3_adv']
  print('scale3_benign: {}'.format(scale3_benign))
  print('scale3_adv: {}'.format(scale3_adv))
  
  perturbFM = parameter_dict['perturbFM']
  # print('perturbFM: {}'.format(perturbFM))

  dt = datetime.now().strftime("%y%m%d%H%M%S")
  
  # change this path
  checkpoint_path = '/ckpts/cifar10_train/model.ckpt-16200'
  
  testing_steps = [100, 500, 1000, 2000]
  # testing_steps = [1000]
  # testing_steps = [2000]
  # testing_steps = [100, 500]
  # fgsm_eps_list = [0.05, 0.3, 0.5]
  fgsm_eps_list = [0.05, 0.3, 0.5]
  fgsm_eps = fgsm_eps_list[GPU_INDEX - 1]
  for testing_step in testing_steps:
    test(cifar10_data, checkpoint_path, stair_epochs, stair_L, stair_learning_rate, scale3_benign, Delta2, epsilon2, eps2_ratio, alpha, perturbFM, fgsm_eps, total_eps, parameter_dict, testing_step)



if __name__ == '__main__':
  tf.app.run()