# -*- coding: utf-8 -*-
'''
An implementation of the StoBatch
'''
#import matplotlib.pyplot as plt
import numpy as np;
import random
import tensorflow as tf;
from tensorflow.python.framework import ops;
#from tensorflow.examples.tutorials.mnist import input_data;
import argparse;
import pickle;
from datetime import datetime
import time
import os
import math
import input_data
import smtplib
from mlp import EncLayer
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
from more_attack import *
from cleverhans.attacks import BasicIterativeMethod, CarliniWagnerL2, DeepFool, FastGradientMethod, MadryEtAl, MomentumIterativeMethod, SPSA, SpatialTransformationMethod
from cleverhans.attacks_tf import fgm, fgsm
from cleverhans import utils_tf
from cleverhans.model import CallableModelWrapper, CustomCallableModelWrapper
from cleverhans.utils import set_log_level
import logging
import copy
import robustness

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
set_log_level(logging.ERROR)

AECODER_VARIABLES = 'AECODER_VARIABLES'
CONV_VARIABLES = 'CONV_VARIABLES'

def parse_time(time_sec):
    time_string = "{} hours {} minutes".format(int(time_sec/3600), int((time_sec%3600)/60))
    return time_string

#def weight_variable(shape):
#    initial = tf.truncated_normal(shape, stddev=0.1);
#    return tf.Variable(initial);

def weight_variable(name, shape, collect):
    #initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0/(5*5*32))/math.ceil(5 / 2));
    #return tf.Variable(initial, collections=[collect]);
    var = tf.get_variable(name = name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    tf.add_to_collections(collect, var)
    return var

def bias_variable(name, shape, collect):
    #initial = tf.constant(0.1, shape=shape);
    #return tf.Variable(initial, collections=[collect]);
    var =  tf.get_variable(name = name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0001))
    tf.add_to_collections(collect, var)
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME');

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME');

def generateIdLMNoise(image_size, Delta2, epsilon2, L):
    #Initiate the noise for the first hidden layer#
    W_conv1Noise = np.random.laplace(0.0, Delta2/(L*epsilon2), image_size**2).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 1]);
    return W_conv1Noise;

def generateHkNoise(hk, Delta, epsilon, L):
    perturbFM = np.random.laplace(0.0, Delta/(epsilon*L), hk)
    perturbFM = np.reshape(perturbFM, [hk]);
    return perturbFM;

def generateNoise(image_size, Delta2, _beta, epsilon2, L):
    #Initiate the noise for the first hidden layer#
    W_conv1Noise = np.random.laplace(0.0, 0.1, image_size**2).astype(np.float32);
    W_conv1Noise = np.reshape(W_conv1Noise, [-1, image_size, image_size, 1]);
    #Redistribute the noise#
    for i in range(0, image_size):
        for j in range(0, image_size):
            W_conv1Noise[0][i][j] = np.random.laplace(0.0, Delta2/(L*(_beta[i+2][j+2])*epsilon2), 1);
    ###
    return W_conv1Noise;

FLAGS = None;

def avg_pool(input, s):
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

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

def batch_adv(sess, adv_op, x_images, y_, test_input, test_labels):
    '''
        Generate adv samples in batch
        '''
    feed_dict = {x_images:test_input, y_:test_labels}
    adv_list = sess.run(adv_op, feed_dict=feed_dict)
    return adv_list

def inference(x_image, perturbFM, hk, perturbFM_h, params):
    z = x_image;
    """W_conv1 = weight_variable([5, 5, 1, 32]);
    b_conv1 = bias_variable([32]);"""
    W_conv1 = params[0];
    b_conv1 = params[1];

    h_conv1 = tf.nn.relu(conv2d(z, W_conv1) + b_conv1 + perturbFM_h);
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1, scale=True, is_training=True, updates_collections=[CONV_VARIABLES])
    #h_pool1 = max_pool_2x2(h_conv1);
    h_pool1 = avg_pool(h_conv1, 2);
    print(h_pool1)

    """W_conv2 = weight_variable([5, 5, 32, 64]);
    b_conv2 = bias_variable([64]);"""
    W_conv2 = params[2];
    b_conv2 = params[3];
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
    #h_pool2 = max_pool_2x2(h_conv2);
    #h_pool2 = avg_pool(h_conv2, 2);
    print(h_conv2)

    """W_fc1 = weight_variable([7 * 7 * 64, hk]);
    b_fc1 = bias_variable([hk]);"""
    W_fc1 = params[4];
    b_fc1 = params[5];
    h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*64]);
    z2 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
    #Applying normalization for the flat connected layer h_fc1#
    #batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
    ################################################################################################################### changed this
    # scale2 = tf.Variable(tf.ones([hk]))
    # beta2 = tf.Variable(tf.zeros([hk]))
    #scale2 = params[8]
    #beta2 = params[9]
    ################################################################################################################### changed this
    BN_norm = tf.contrib.layers.batch_norm(z2, scale=True, is_training=True, updates_collections=[CONV_VARIABLES]) #tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
    ###
    #h_fc1 = tf.nn.relu(BN_norm);
    h_fc1 = max_out(BN_norm, hk)
    h_fc1 = tf.clip_by_value(h_fc1, -1, 1) #hidden neurons must be bounded in [-1, 1]
    #perturbFM = tf.placeholder(tf.float32, [hk]);
    #h_fc1 += perturbFM;
    #Sometime bound the 2-norm of h_fc1 can help to stablize the training process. Use with care.
    #h_fc1 = tf.clip_by_norm(h_fc1, c[2], 1);
    #place holder for dropout, however we do not use dropout in this code#
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
    ###
    """W_fc2 = weight_variable([hk, 10]);
    b_fc2 = bias_variable([10]);"""
    W_fc2 = params[6];
    b_fc2 = params[7];
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2;
    ###
    
    return y_conv

def inference_robust_mask(y_conv, Delta2, L, eps, robustness_T):
    ##Robustness####
    softmax = tf.nn.softmax(y_conv)
    values, indices = tf.nn.top_k(softmax,2)
    values = tf.unstack(values, num=2, axis=1)
    print('nn_tok_k: ')
    print(values)
    print(indices)
    eps_r = tf.log(values[0] / values[1]) / 2.0
    print('eps_r')
    print(eps_r)
    kappa_phi = (Delta2 * eps_r * (1 + 2e-5))/(L * eps)
    print('kappa_phi')
    print(kappa_phi)
    robust_pos = tf.ones(shape=tf.shape(kappa_phi))
    robust_neg = tf.zeros(shape=tf.shape(kappa_phi))
    robust_mask = tf.where(eps_r >= robustness_T, robust_pos, robust_neg)
    print('robust_mask')
    print(robust_mask)
    #####
    return robust_mask

def inference_probs(x_image, perturbFM, hk, params):
    logits = inference(x_image, perturbFM, hk, params)
    return tf.nn.softmax(logits)

def inference_test_input(x, hk, params, image_size, adv_noise):
    z = tf.reshape(x, [-1,image_size,image_size,1]) + adv_noise

    W_conv1 = params[0];
    b_conv1 = params[1];

    h_conv1 = tf.nn.relu(conv2d(z, W_conv1) + b_conv1);
    h_conv1 = tf.contrib.layers.batch_norm(h_conv1, scale=True, is_training=False, updates_collections=None)

    h_pool1 = avg_pool(h_conv1, 2);
    print(h_pool1)

    W_conv2 = params[2];
    b_conv2 = params[3];
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
    #h_pool2 = max_pool_2x2(h_conv2);
    #h_pool2 = avg_pool(h_conv2, 2);
    print(h_conv2)

    W_fc1 = params[4];
    b_fc1 = params[5];
    h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*64]);
    z2 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
    #Applying normalization for the flat connected layer h_fc1#
    batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
    
    #scale2 = params[8]
    #beta2 = params[9]
    
    #BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
    BN_norm = tf.contrib.layers.batch_norm(z2, scale=True, is_training=False, updates_collections=None)
    ###
    #h_fc1 = tf.nn.relu(BN_norm);
    h_fc1 = max_out(BN_norm, hk)
    #h_fc1 = tf.clip_by_value(h_fc1, -1, 1) #hidden neurons must be bounded in [-1, 1]
    #perturbFM = tf.placeholder(tf.float32, [hk]);
    #h_fc1 += perturbFM;
    ###
    """W_fc2 = weight_variable([hk, 10]);
    b_fc2 = bias_variable([10]);"""
    W_fc2 = params[6];
    b_fc2 = params[7];
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2;
    ###
    return y_conv

def inference_test_input_probs(x, hk, params, image_size, adv_noise):
    logits = inference_test_input(x, hk, params, image_size, adv_noise)
    return tf.nn.softmax(logits)

def train(alpha, eps2_ratio, gen_ratio, fgsm_eps, LR, logfile):
    logfile.write("fgsm_eps \t %g, LR \t %g, alpha \t %d , eps2_ratio \t %d , gen_ratio \t %d \n"%(fgsm_eps, LR, alpha, eps2_ratio, gen_ratio))
    #############################
    ##Hyper-parameter Setting####
    #############################
    hk = 256; #number of hidden units at the last layer
    Delta2 = (14*14+2)*25; #global sensitivity for the first hidden layer
    Delta3_adv = 2*hk #10*(hk + 1/4 * hk**2) #10*(hk) #global sensitivity for the output layer
    Delta3_benign = 2*hk #10*(hk); #global sensitivity for the output layer
    D = 50000; #size of the dataset
    L = 2499; #batch size
    image_size = 28;
    padding = 4;
    #numHidUnits = 14*14*32 + 7*7*64 + M + 10; #number of hidden units
    #gen_ratio = 1
    epsilon1 = 0.0; #0.175; #epsilon for dpLRP
    epsilon2 = 0.1*(1 + gen_ratio); #epsilon for the first hidden layer
    epsilon3 = 0.1*(1); #epsilon for the last hidden layer
    total_eps = epsilon1 + epsilon2 + epsilon3
    print(total_eps)
    uncert = 0.1; #uncertainty modeling at the output layer
    infl = 1; #inflation rate in the privacy budget redistribution
    R_lowerbound = 1e-5; #lower bound of the LRP
    c = [0, 40, 50, 200] #norm bounds
    epochs = 200; #number of epochs
    preT_epochs = 50; #number of epochs
    T = int(D/L*epochs + 1); #number of steps T
    pre_T = int(D/L*preT_epochs + 1);
    step_for_epoch = int(D/L); #number of steps for one epoch
    
    broken_ratio = 1
    #alpha = 9.0 # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #eps2_ratio = 10; # [1/10, 1/8, 1/6, 1/4, 1/2, 1, 2, 4, 6, 8, 10]
    #eps_benign = 1/(1+eps2_ratio)*(2*epsilon2)
    #eps_adv = eps2_ratio/(1+eps2_ratio)*(2*epsilon2)
    
    #fgsm_eps = 0.1
    rand_alpha = 0.05
    
    ##Robustness##
    robustness_T = (fgsm_eps*18*18*L*epsilon2)/Delta2;
    ####
    
    LRPfile = os.getcwd() + '/Relevance_R_0_075.txt';
    #############################
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True);

    #############################
    ##Construct the Model########
    #############################
    #Step 4: Randomly initiate the noise, Compute 1/|L| * Delta3 for the output layer#

    #Compute the 1/|L| * Delta3 for the last hidden layer#
    """eps3_ratio = Delta3_adv/Delta3_benign;
    eps3_benign = 1/(1+eps3_ratio)*(epsilon3)
    eps3_adv = eps3_ratio/(1+eps3_ratio)*(epsilon3)"""
    loc, scale3_benign, scale3_adv = 0., Delta3_benign/(epsilon3*L), Delta3_adv/(epsilon3*L);
    ###
    #End Step 4#
    # Parameters Declarification
    W_conv1 = weight_variable('W_conv1', [5, 5, 1, 32], collect=[AECODER_VARIABLES]);
    b_conv1 = bias_variable('b_conv1', [32], collect=[AECODER_VARIABLES]);

    shape     = W_conv1.get_shape().as_list()
    w_t       = tf.reshape(W_conv1, [-1, shape[-1]])
    w         = tf.transpose(w_t)
    sing_vals = tf.svd(w, compute_uv=False)
    sensitivity = tf.reduce_max(sing_vals)
    gamma = 2*(14*14 + 2)*25/(L*sensitivity)
    
    dp_epsilon=1.0 #0.1
    delta_r = fgsm_eps*(image_size**2);
    #delta_h = 1.0 * delta_r; #sensitivity*(14**2) = sensitivity*(\beta**2) can also be used
    #dp_mult = (Delta2/(L*epsilon2))/(delta_r / dp_epsilon) + (2*Delta2/(L*epsilon2))/(delta_h / dp_epsilon)
    
    W_conv2 = weight_variable('W_conv2', [5, 5, 32, 64], collect=[CONV_VARIABLES]);
    b_conv2 = bias_variable('b_conv2', [64], collect=[CONV_VARIABLES]);

    W_fc1 = weight_variable('W_fc1', [4 * 4 * 64, hk], collect=[CONV_VARIABLES]);
    b_fc1 = bias_variable('b_fc1', [hk], collect=[CONV_VARIABLES]);

    W_fc2 = weight_variable('W_fc2', [hk, 10], collect=[CONV_VARIABLES]);
    b_fc2 = bias_variable('b_fc2', [10], collect=[CONV_VARIABLES]);

    """scale2 = tf.Variable(tf.ones([hk]))
    beta2 = tf.Variable(tf.zeros([hk]))
    tf.add_to_collections([CONV_VARIABLES], scale2)
    tf.add_to_collections([CONV_VARIABLES], beta2)"""

    params = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
    ###


    #Step 5: Create the model#
    noise = tf.placeholder(tf.float32, [None, image_size, image_size, 1]);
    adv_noise = tf.placeholder(tf.float32, [None, image_size, image_size, 1]);

    keep_prob = tf.placeholder(tf.float32);
    x = tf.placeholder(tf.float32, [None, image_size*image_size]);
    x_image = tf.reshape(x, [-1,image_size,image_size,1]);

    #perturbFMx = np.random.laplace(0.0, Delta2/(2*epsilon2*L), 28*28)
    #perturbFMx = np.reshape(perturbFMx, [-1, 28, 28, 1]);

    # pretrain ###
    #Enc_Layer1 = EncLayer(inpt=x_image, n_filter_in = 1, n_filter_out = 32, filter_size = 5, W=W_conv1, b=b_conv1, activation=tf.nn.relu)
    #pretrain = Enc_Layer1.get_train_ops2(xShape = tf.shape(x_image)[0], Delta = Delta2, epsilon = 2*epsilon2, batch_size = L, learning_rate= LR, W = W_conv1, b = b_conv1, perturbFMx = noise)
    ###########

    adv_x = tf.placeholder(tf.float32, [None, image_size*image_size]);
    adv_image = tf.reshape(adv_x, [-1,image_size,image_size,1]);

    #perturbFMx_adv = np.random.laplace(0.0, Delta2/(2*epsilon2*L), 28*28)
    #perturbFMx_adv = np.reshape(perturbFMx_adv, [-1, 28, 28, 1]);

    # pretrain adv ###
    #perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2*L), 14*14*32)
    #perturbFM_h = np.reshape(perturbFM_h, [-1, 14, 14, 32]);
    FM_h = tf.placeholder(tf.float32, [None, 14, 14, 32]);
    Enc_Layer2 = EncLayer(inpt=adv_image, n_filter_in = 1, n_filter_out = 32, filter_size = 5, W=W_conv1, b=b_conv1, activation=tf.nn.relu)
    pretrain_adv = Enc_Layer2.get_train_ops2(xShape = tf.shape(adv_image)[0], Delta = Delta2, batch_size = L, learning_rate= LR, W = W_conv1, b = b_conv1, perturbFMx = adv_noise, perturbFM_h = FM_h)
    Enc_Layer3 = EncLayer(inpt=x_image, n_filter_in = 1, n_filter_out = 32, filter_size = 5, W=W_conv1, b=b_conv1, activation=tf.nn.relu)
    pretrain_benign = Enc_Layer3.get_train_ops2(xShape = tf.shape(x_image)[0], Delta = Delta2, batch_size = L, learning_rate= LR, W = W_conv1, b = b_conv1, perturbFMx = noise, perturbFM_h = FM_h)
    ###########
    
    x_image += noise;
    x_image = tf.clip_by_value(x_image, -10, 10) #Clip the values of each input feature.
    
    adv_image += adv_noise;
    adv_image = tf.clip_by_value(adv_image, -10, 10) #Clip the values of each input feature.

    #perturbFM = np.random.laplace(0.0, scale3_benign, hk)
    #perturbFM = np.reshape(perturbFM, [hk]);
    perturbFM = np.random.laplace(0.0, scale3_benign, hk * 10)
    perturbFM = np.reshape(perturbFM, [hk, 10]);
    
    y_conv = inference(x_image, perturbFM, hk, FM_h, params);
    softmax_y_conv = tf.nn.softmax(y_conv)
    #robust_mask = inference_robust_mask(y_conv, Delta2, L, epsilon2, robustness_T)

    #perturbFM = np.random.laplace(0.0, scale3_adv, hk)
    #perturbFM = np.reshape(perturbFM, [hk]);
    y_adv_conv = inference(adv_image, perturbFM, hk, FM_h, params);
    #adv_robust_mask = inference_robust_mask(y_adv_conv, Delta2, L, epsilon2, robustness_T)

    # test model
    perturbFM_test = np.random.laplace(0.0, 0, hk)
    perturbFM_test = np.reshape(perturbFM_test, [hk]);
    x_test = tf.reshape(x, [-1,image_size,image_size,1]);
    y_test = inference(x_test, perturbFM_test, hk, FM_h, params);
    #test_robust_mask = inference_robust_mask(y_test, Delta2, L, epsilon2, robustness_T)

    #Define a place holder for the output label#
    y_ = tf.placeholder(tf.float32, [None, 10]);
    adv_y_ = tf.placeholder(tf.float32, [None, 10]);
    #End Step 5#
    #############################

    #############################
    ##Define loss and Optimizer##
    #############################
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
        
        `logits` and `labels` must have the same type and shape. Let denote neg_abs_logits = -abs(y_conv) = -abs(h_fc1 * W_fc2). By Applying Taylor Expansion, we have:
        
        Taylor = max(y_conv, 0) - y_conv * y_ + log(1 + exp(-abs(y_conv)));
        = max(h_fc1 * W_fc2, 0) - (y_ * h_fc1) * W_fc2 + (math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        = max(h_fc1 * W_fc2, 0) - (y_ * h_fc1) * W_fc2 + (math.log(2.0) + 0.5*(-abs(h_fc1 * W_fc2)) + 1.0/8.0*(-abs(h_fc1 * W_fc2))**2)
        = F1 + F2
        where: F1 = max(h_fc1 * W_fc2, 0) + (math.log(2.0) + 0.5*(-abs(h_fc1 * W_fc2)) + 1.0/8.0*(-abs(h_fc1 * W_fc2))**2) and F2 = - (y_ * h_fc1) * W_fc2
        
        To ensure that Taylor is differentially private, we need to perturb all the coefficients, including the term y_ * h_fc1 * W_fc2.
        Note that h_fc1 is differentially private, since its computation on top of the DP Affine transformation does not access the original data.
        Therefore, F1 should be differentially private. We need to preserve DP in F2, which reads the groundtruth label y_, as follows:
        
        By applying Funtional Mechanism, we perturb (y_ * h_fc1) * W_fc2 as ((y_ * h_fc1) + perturbFM) * W_fc2 = (y_ * h_fc1)*W_fc2 + (perturbFM * W_fc2):
        
        perturbFM = np.random.laplace(0.0, scale3, hk * 10)
        perturbFM = np.reshape(perturbFM/L, [hk, 10]);
        
        where scale3 = Delta3/(epsilon3) = 2*hk/(epsilon3);
        
        To allow computing gradients at zero, we define custom versions of max and abs functions [Tensorflow].
        
        Source: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_impl.py @ TensorFlow
    '''
    ### Taylor for benign x
    zeros = array_ops.zeros_like(y_conv, dtype=y_conv.dtype)
    cond = (y_conv >= zeros)
    relu_logits = array_ops.where(cond, y_conv, zeros)
    neg_abs_logits = array_ops.where(cond, -y_conv, y_conv)
    #Taylor = math_ops.add(relu_logits - y_conv * y_, math_ops.log1p(math_ops.exp(neg_abs_logits)))
    Taylor_benign = math_ops.add(relu_logits - y_conv * y_, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2) - tf.reduce_sum(perturbFM*W_fc2)
    #Taylor_benign = tf.abs(y_conv - y_)

    ### Taylor for adv_x
    zeros_adv = array_ops.zeros_like(y_adv_conv, dtype=y_conv.dtype)
    cond_adv = (y_adv_conv >= zeros_adv)
    relu_logits_adv = array_ops.where(cond_adv, y_adv_conv, zeros_adv)
    neg_abs_logits_adv = array_ops.where(cond_adv, -y_adv_conv, y_adv_conv)
    #Taylor = math_ops.add(relu_logits - y_conv * y_, math_ops.log1p(math_ops.exp(neg_abs_logits)))
    Taylor_adv = math_ops.add(relu_logits_adv - y_adv_conv * adv_y_, math.log(2.0) + 0.5*neg_abs_logits_adv + 1.0/8.0*neg_abs_logits_adv**2) - tf.reduce_sum(perturbFM*W_fc2)
    #Taylor_adv = tf.abs(y_adv_conv - adv_y_)

    ### Adversarial training loss
    adv_loss = (1/(L + L*alpha))*(Taylor_benign + alpha * Taylor_adv)

    '''Some time, using learning rate decay can help to stablize training process. However, use this carefully, since it may affect the convergent speed.'''
    global_step = tf.Variable(0, trainable=False)
    pretrain_var_list = tf.get_collection(AECODER_VARIABLES)
    train_var_list = tf.get_collection(CONV_VARIABLES)
    #print(pretrain_var_list)
    #print(train_var_list)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        pretrain_step = tf.train.AdamOptimizer(LR).minimize(pretrain_adv+pretrain_benign, global_step=global_step, var_list=pretrain_var_list);
        train_step = tf.train.AdamOptimizer(LR).minimize(adv_loss, global_step=global_step, var_list=train_var_list);
    sess = tf.InteractiveSession();

    # Define the correct prediction and accuracy
    # This needs to be changed to "Robust Prediction"
    correct_prediction_x = tf.equal(tf.argmax(y_test,1), tf.argmax(y_,1));
    accuracy_x = tf.reduce_mean(tf.cast(correct_prediction_x, tf.float32));

    #############
    # use these to get predictions wrt to robust conditions
    """robust_correct_prediction_x = tf.multiply(test_robust_mask, tf.cast(correct_prediction_x, tf.float32))
    accuracy_x_robust = tf.reduce_sum(robust_correct_prediction_x) / tf.reduce_sum(test_robust_mask)
    #certified_utility = 2/(1/accuracy_x_robust + 1/(tf.reduce_sum(test_robust_mask)/(1.0*tf.cast(tf.size(test_robust_mask), tf.float32))))
    certified_utility = (1.0*tf.reduce_sum(test_robust_mask))/(1.0*tf.cast(tf.size(test_robust_mask), tf.float32))"""
    #############

    # craft adversarial samples from x for training
    dynamic_eps = tf.placeholder(tf.float32);
    emsemble_L = int(L/3)
    softmax_y = tf.nn.softmax(y_test)
    #c_x_adv = fgsm(x, softmax_y, eps=fgsm_eps, clip_min=0.0, clip_max=1.0)
    c_x_adv = fgsm(x, softmax_y, eps=(dynamic_eps)/10, clip_min=-1.0, clip_max=1.0) # for I-FGSM
    x_adv = tf.reshape(c_x_adv, [emsemble_L,image_size*image_size]);

    #====================== attack =========================
    #attack_switch = {'randfgsm':True, 'fgsm':True, 'ifgsm':True, 'deepfool':True, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':True}
    #attack_switch = {'fgsm':True, 'ifgsm':True, 'deepfool':True, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':True}
    attack_switch = {'fgsm':True, 'ifgsm':True, 'deepfool':False, 'mim':True, 'spsa':False, 'cwl2':False, 'madry':True, 'stm':False}
    #other possible attacks:
        # ElasticNetMethod
        # FastFeatureAdversaries
        # LBFGS
        # SaliencyMapMethod
        # VirtualAdversarialMethod

    # y_test = logits (before softmax)
    # softmax_y_test = preds (probs, after softmax)
    softmax_y_test = tf.nn.softmax(y_test)

    # create saver
    saver = tf.train.Saver(tf.all_variables())
    
    sess.run(W_conv1.initializer)
    _gamma = sess.run(gamma)
    _gamma_x = Delta2/L
    epsilon2_update = epsilon2/(1.0 + 1.0/_gamma + 1/_gamma_x)
    print(epsilon2_update/_gamma + epsilon2_update/_gamma_x)
    print(epsilon2_update)
    _sensitivityW = sess.run(sensitivity)
    delta_h = _sensitivityW*(14**2)
    dp_mult = (Delta2/(L*epsilon2_update))/(delta_r / dp_epsilon) + (2*Delta2/(L*epsilon2_update))/(delta_h / dp_epsilon)
    #############################
    
    iterativeStep = 100
    
    # load the most recent models
    _global_step = 0
    ckpt = tf.train.get_checkpoint_state(os.getcwd() + './tmp/train')
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path);
        saver.restore(sess, ckpt.model_checkpoint_path)
        _global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('No checkpoint file found')

    start_time = time.time();

    # adv pretrain model (Auto encoder layer)
    cost = tf.reduce_sum(Enc_Layer2.cost);
    logfile.write("pretrain: \n")
    
    # define cleverhans abstract models for using cleverhans attacks
    ch_model_logits = CustomCallableModelWrapper(callable_fn=inference_test_input, output_layer='logits', hk=hk, params=params, image_size=image_size, adv_noise = adv_noise)
    ch_model_probs = CustomCallableModelWrapper(callable_fn=inference_test_input_probs, output_layer='probs', hk=hk, params=params, image_size=image_size, adv_noise = adv_noise)

    # rand+fgsm
    # if attack_switch['randfgsm']:
    #     randfgsm_obj = FastGradientMethod(model=ch_model_probs, sess=sess)
    #     x_randfgsm_t = (fgsm_eps - rand_alpha) * randfgsm_obj.generate(x=x, eps=fgsm_eps, clip_min=-1.0, clip_max=1.0)
    #     x_rand_t = rand_alpha * tf.sign(tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0))

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
        x_adv_test_ifgsm = ifgsm_obj.generate(x=x, eps=mu_alpha, eps_iter=mu_alpha/iterativeStep, nb_iter=iterativeStep, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['ifgsm'] = x_adv_test_ifgsm

    # Deepfool
    if attack_switch['deepfool']:
        print('creating attack tensor of DeepFool')
        deepfool_obj = DeepFool(model=ch_model_logits, sess=sess)
        #x_adv_test_deepfool = deepfool_obj.generate(x=x, nb_candidate=10, overshoot=0.02, max_iter=50, nb_classes=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_deepfool = deepfool_obj.generate(x=x, nb_candidate=10, overshoot=0.02, max_iter=50, nb_classes=10, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['deepfool'] = x_adv_test_deepfool

    # MomentumIterativeMethod
    # default: eps_iter=0.06, nb_iter=10
    if attack_switch['mim']:
        print('creating attack tensor of MomentumIterativeMethod')
        mim_obj = MomentumIterativeMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_mim = mim_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, decay_factor=1.0, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_mim = mim_obj.generate(x=x, eps=mu_alpha, eps_iter=mu_alpha/iterativeStep, nb_iter=iterativeStep, decay_factor=1.0, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['mim'] = x_adv_test_mim

    # SPSA
    # note here the epsilon is the infinity norm instead of precent of perturb
    # Maybe exclude this method first, since it seems to have some constrain about the data value range
    if attack_switch['spsa']:
        print('creating attack tensor of SPSA')
        spsa_obj = SPSA(model=ch_model_logits, sess=sess)
        #x_adv_test_spsa = spsa_obj.generate(x=x, epsilon=fgsm_eps, num_steps=10, is_targeted=False, early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01,spsa_samples=1000, spsa_iters=1, ord=2)
        x_adv_test_spsa = spsa_obj.generate(x=x, epsilon=fgsm_eps, num_steps=10, is_targeted=False, early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01,spsa_samples=1000, spsa_iters=1)
        attack_tensor_dict['spsa'] = x_adv_test_spsa

    # CarliniWagnerL2
    # confidence=0 is fron their paper
    # it is said to be slow, maybe exclude first
    if attack_switch['cwl2']:
        print('creating attack tensor of CarliniWagnerL2')
        cwl2_obj = CarliniWagnerL2(model=ch_model_logits, sess=sess)
        #x_adv_test_cwl2 = cwl2_obj.generate(x=x, confidence=0, batch_size=1000, learning_rate=0.005, binary_search_steps=5, max_iterations=500, abort_early=True, initial_const=0.01, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_cwl2 = cwl2_obj.generate(x=x, confidence=0, batch_size=1000, learning_rate=0.005, binary_search_steps=5, max_iterations=500, abort_early=True, initial_const=0.01, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['cwl2'] = x_adv_test_cwl2

    # MadryEtAl (Projected Grdient with random init, same as rand+fgsm)
    # default: eps_iter=0.01, nb_iter=40
    if attack_switch['madry']:
        print('creating attack tensor of MadryEtAl')
        madry_obj = MadryEtAl(model=ch_model_probs, sess=sess)
        #x_adv_test_madry = madry_obj.generate(x=x, eps=fgsm_eps, eps_iter=fgsm_eps/10, nb_iter=10, clip_min=-1.0, clip_max=1.0, ord=2)
        x_adv_test_madry = madry_obj.generate(x=x, eps=mu_alpha, eps_iter=fgsm_eps/iterativeStep, nb_iter=iterativeStep, clip_min=-1.0, clip_max=1.0)
        attack_tensor_dict['madry'] = x_adv_test_madry

    # SpatialTransformationMethod
    # the params are pretty different from on the paper
    # so I use default
    # exclude since there's bug
    if attack_switch['stm']:
        print('creating attack tensor of SpatialTransformationMethod')
        stm_obj = SpatialTransformationMethod(model=ch_model_probs, sess=sess)
        #x_adv_test_stm = stm_obj.generate(x=x, batch_size=1000, n_samples=None, dx_min=-0.1, dx_max=0.1, n_dxs=2, dy_min=-0.1, dy_max=0.1, n_dys=2, angle_min=-30, angle_max=30, n_angles=6, ord=2)
        x_adv_test_stm = stm_obj.generate(x=x, batch_size=1000, n_samples=None, dx_min=-0.1, dx_max=0.1, n_dxs=2, dy_min=-0.1, dy_max=0.1, n_dys=2, angle_min=-30, angle_max=30, n_angles=6)
        attack_tensor_dict['stm'] = x_adv_test_stm
    #====================== attack =========================
    
    sess.run(tf.initialize_all_variables());

    ##perturb h for training
    perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*32)
    perturbFM_h = np.reshape(perturbFM_h, [-1, 14, 14, 32]);

    ##perturb h for testing
    perturbFM_h_test = np.random.laplace(0.0, 0, 14*14*32)
    perturbFM_h_test = np.reshape(perturbFM_h_test, [-1, 14, 14, 32]);

    '''for i in range(_global_step, _global_step + pre_T):
        d_eps = random.random();
        
        batch = mnist.train.next_batch(L); #Get a random batch.
        adv_images = sess.run(x_adv, feed_dict = {x:batch[0], y_:batch[1], FM_h: perturbFM_h_test, dynamic_eps: d_eps})
        for iter in range(0, 9):
            adv_images = sess.run(x_adv, feed_dict = {x:adv_images, y_:batch[1], FM_h: perturbFM_h_test, dynamic_eps: d_eps})
        """batch = mnist.train.next_batch(emsemble_L)
        adv_images_mim = sess.run(attack_tensor_dict['mim'], feed_dict = {x:batch[0], y_: batch[1]})
        batch = mnist.train.next_batch(emsemble_L)
        adv_images_madry = sess.run(attack_tensor_dict['mim'], feed_dict = {x:batch[0], y_: batch[1]})
        train_images = np.append(np.append(adv_images, adv_images_mim, axis = 0),adv_images_madry, axis = 0)"""

        batch_2 = mnist.train.next_batch(L);
        pretrain_step.run(feed_dict={adv_x: np.append(adv_images, batch_2[0], axis = 0), adv_noise: AdvLnoise, FM_h: perturbFM_h});
        if i % int(5*step_for_epoch) == 0:
            cost_value = sess.run(cost, feed_dict={adv_x:mnist.test.images, adv_noise: AdvLnoise_test, FM_h: perturbFM_h_test})/(test_size*32)
            logfile.write("step \t %d \t %g \n"%(i, cost_value))
            print(cost_value)

    pre_train_finish_time = time.time()
    print('pre_train finished in: ' + parse_time(pre_train_finish_time - start_time))'''

    # train and test model with adv samples
    max_benign_acc = -1;
    max_robust_benign_acc = -1
    #max_adv_acc = -1;

    test_size = len(mnist.test.images)
    AdvLnoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L);
    AdvLnoise_test = generateIdLMNoise(image_size, 0, epsilon2_update, test_size);

    Lnoise_empty = generateIdLMNoise(image_size, 0, epsilon2_update, L);
    BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L);
    last_eval_time = -1
    accum_time = 0
    accum_epoch = 0
    max_adv_acc_dict = {}
    max_robust_adv_acc_dict = {}
    #max_robust_adv_utility_dict = {}
    for atk in attack_switch.keys():
        if atk not in max_adv_acc_dict:
            max_adv_acc_dict[atk] = -1
            max_robust_adv_acc_dict[atk] = -1

    for i in range(_global_step, _global_step + T):
        # this batch is for generating adv samples
        batch = mnist.train.next_batch(emsemble_L); #Get a random batch.
        y_adv_batch = batch[1]
        #The number of epochs we print out the result. Print out the result every 5 epochs.
        if i % int(10*step_for_epoch) == 0 and i > int(10*step_for_epoch):
            cost_value = sess.run(cost, feed_dict={adv_x:mnist.test.images, adv_noise: AdvLnoise_test, FM_h: perturbFM_h_test})/(test_size*32)
            print(cost_value)
            
            if last_eval_time < 0:
                last_eval_time = time.time()
            #===================benign samples=====================
            predictions_form_argmax = np.zeros([test_size, 10])
            #test_bach = mnist.test.next_batch(test_size)
            softmax_predictions = softmax_y_conv.eval(feed_dict={x: mnist.test.images, noise: BenignLNoise, FM_h: perturbFM_h})
            argmax_predictions = np.argmax(softmax_predictions, axis=1)
            for n_draws in range(0, 1):
                _BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L);
                _perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*32)
                _perturbFM_h = np.reshape(_perturbFM_h, [-1, 14, 14, 32]);
                for j in range(test_size):
                    pred = argmax_predictions[j]
                    predictions_form_argmax[j, pred] += 1;
                softmax_predictions = softmax_y_conv.eval(feed_dict={x: mnist.test.images, noise: (BenignLNoise + _BenignLNoise/2), FM_h: (perturbFM_h + _perturbFM_h/2)})
                argmax_predictions = np.argmax(softmax_predictions, axis=1)
            final_predictions = predictions_form_argmax;
            is_correct = []
            is_robust = []
            for j in range(test_size):
                is_correct.append(np.argmax(mnist.test.labels[j]) == np.argmax(final_predictions[j]))
                robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=1.0, dp_delta=0.05, dp_mechanism='laplace') / (dp_mult)
                is_robust.append(robustness_from_argmax >= fgsm_eps)
            acc = np.sum(is_correct)*1.0/test_size
            robust_acc = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
            robust_utility = np.sum(is_robust)*1.0/test_size
            max_benign_acc = max(max_benign_acc, acc)
            max_robust_benign_acc = max(max_robust_benign_acc, robust_acc*robust_utility)
            log_str = "step: {:.1f}\t epsilon: {:.1f}\t benign: {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t".format(i, total_eps, acc, robust_acc, robust_utility, robust_acc*robust_utility)
            #===================adv samples=====================
            #log_str = "step: {:.1f}\t epsilon: {:.1f}\t".format(i, total_eps)
            """adv_images_dict = {}
            for atk in attack_switch.keys():
                if attack_switch[atk]:
                    adv_images_dict[atk] = sess.run(attack_tensor_dict[atk], feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
            print("Done with the generating of Adversarial samples")"""
            #===================adv samples=====================
            adv_acc_dict = {}
            robust_adv_acc_dict = {}
            robust_adv_utility_dict = {}
            for atk in attack_switch.keys():
                if atk not in adv_acc_dict:
                    adv_acc_dict[atk] = -1
                    robust_adv_acc_dict[atk] = -1
                    robust_adv_utility_dict[atk] = -1
                if attack_switch[atk]:
                    adv_images_dict = sess.run(attack_tensor_dict[atk], feed_dict = {x:mnist.test.images, y_: mnist.test.labels, adv_noise: AdvLnoise_test, mu_alpha:[fgsm_eps]})
                    ### PixelDP Robustness ###
                    predictions_form_argmax = np.zeros([test_size, 10])
                    softmax_predictions = softmax_y_conv.eval(feed_dict={x: adv_images_dict, noise: BenignLNoise, FM_h: perturbFM_h})
                    argmax_predictions = np.argmax(softmax_predictions, axis=1)
                    for n_draws in range(0, 2000):
                        if n_draws % 1000 == 0:
                            print(n_draws)
                        _BenignLNoise = generateIdLMNoise(image_size, Delta2, epsilon2_update, L);
                        _perturbFM_h = np.random.laplace(0.0, 2*Delta2/(epsilon2_update*L), 14*14*32)
                        _perturbFM_h = np.reshape(_perturbFM_h, [-1, 14, 14, 32]);
                        for j in range(test_size):
                            pred = argmax_predictions[j]
                            predictions_form_argmax[j, pred] += 1;
                        softmax_predictions = softmax_y_conv.eval(feed_dict={x: adv_images_dict, noise: BenignLNoise, FM_h: (perturbFM_h + _perturbFM_h/2)}) * softmax_y_conv.eval(feed_dict={x: adv_images_dict, noise: (BenignLNoise + _BenignLNoise/2), FM_h: perturbFM_h})
                        #softmax_predictions = softmax_y_conv.eval(feed_dict={x: adv_images_dict, noise: BenignLNoise, FM_h: (_perturbFM_h)}) * softmax_y_conv.eval(feed_dict={x: adv_images_dict, noise: (_BenignLNoise), FM_h: perturbFM_h})
                        argmax_predictions = np.argmax(softmax_predictions, axis=1)
                    final_predictions = predictions_form_argmax;
                    is_correct = []
                    is_robust = []
                    for j in range(test_size):
                        is_correct.append(np.argmax(mnist.test.labels[j]) == np.argmax(final_predictions[j]))
                        robustness_from_argmax = robustness.robustness_size_argmax(counts=predictions_form_argmax[j],eta=0.05,dp_attack_size=fgsm_eps, dp_epsilon=1.0, dp_delta=0.05, dp_mechanism='laplace') / (dp_mult)
                        is_robust.append(robustness_from_argmax >= fgsm_eps)
                    adv_acc_dict[atk] = np.sum(is_correct)*1.0/test_size
                    robust_adv_acc_dict[atk] = np.sum([a and b for a,b in zip(is_robust, is_correct)])*1.0/np.sum(is_robust)
                    robust_adv_utility_dict[atk] = np.sum(is_robust)*1.0/test_size
                    ##############################
            for atk in attack_switch.keys():
                if attack_switch[atk]:
                    # added robust prediction
                    log_str += " {}: {:.4f} {:.4f} {:.4f} {:.4f}".format(atk, adv_acc_dict[atk], robust_adv_acc_dict[atk], robust_adv_utility_dict[atk], robust_adv_acc_dict[atk]*robust_adv_utility_dict[atk])
                    max_adv_acc_dict[atk] = max(max_adv_acc_dict[atk], adv_acc_dict[atk])
                    max_robust_adv_acc_dict[atk] = max(max_robust_adv_acc_dict[atk], robust_adv_acc_dict[atk]*robust_adv_utility_dict[atk])
            print(log_str)
            logfile.write(log_str + '\n')

            # logfile.write("step \t %d \t %g \t %g \n"%(i, benign_acc, adv_acc))
            # print("step \t %d \t %g \t %g"%(i, benign_acc, adv_acc));

            # estimate end time
            """if i > 0 and i % int(10*step_for_epoch) == 0:
                current_time_interval = time.time() - last_eval_time
                last_eval_time = time.time()
                print('during last eval interval, {} epoch takes {}'.format(10, parse_time(current_time_interval)))
                accum_time += current_time_interval
                accum_epoch += 10
                estimate_time = ((_global_step + T - i) / step_for_epoch) * (accum_time / accum_epoch)
                print('estimate finish in: {}'.format(parse_time(estimate_time)))"""

            #print("step \t %d \t adversarial test accuracy \t %g"%(i, accuracy_x.eval(feed_dict={x: adv_images, y_: mnist.test.labels, noise: Lnoise_empty})));
            """checkpoint_path = os.path.join(os.getcwd() + '/tmp/train', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i);"""

        d_eps = random.random();
        y_adv = batch[1]
        adv_images = sess.run(attack_tensor_dict['ifgsm'], feed_dict = {x:batch[0], y_: batch[1], adv_noise: AdvLnoise, mu_alpha:[d_eps]})
        """for iter in range(0, 9):
            adv_images = sess.run(x_adv, feed_dict = {x:adv_images, y_:batch[1], FM_h: perturbFM_h_test, dynamic_eps: d_eps})"""
        batch = mnist.train.next_batch(emsemble_L)
        adv_images_mim = sess.run(attack_tensor_dict['mim'], feed_dict = {x:batch[0], y_: batch[1], adv_noise: AdvLnoise, mu_alpha:[d_eps]})
        y_adv = np.append(y_adv, batch[1], axis = 0)
        batch = mnist.train.next_batch(emsemble_L)
        adv_images_madry = sess.run(attack_tensor_dict['madry'], feed_dict = {x:batch[0], y_: batch[1], adv_noise: AdvLnoise, mu_alpha:[d_eps]})
        y_adv = np.append(y_adv, batch[1], axis = 0)
        train_images = np.append(np.append(adv_images, adv_images_mim, axis = 0),adv_images_madry, axis = 0)
        
        batch = mnist.train.next_batch(L); #Get a random batch.
        # train with benign and adv samples
        pretrain_step.run(feed_dict={adv_x: train_images, x: batch[0], adv_noise: AdvLnoise_test, noise: BenignLNoise, FM_h: perturbFM_h});
        train_step.run(feed_dict={x: batch[0], adv_x: train_images, y_: batch[1], adv_y_: y_adv, noise: BenignLNoise, adv_noise: AdvLnoise_test, FM_h: perturbFM_h});
    duration = time.time() - start_time;
    # print(parse_time(duration)); #print running time duration#

    max_acc_string = "max acc: benign: \t{:.4f} {:.4f}".format(max_benign_acc, max_robust_benign_acc)
    for atk in attack_switch.keys():
        if attack_switch[atk]:
            max_acc_string += " {}: \t{:.4f} {:.4f}".format(atk, max_adv_acc_dict[atk], max_robust_adv_acc_dict[atk])
    logfile.write(max_acc_string + '\n')
    logfile.write(str(duration) + '\n')
    # logfile.write("max accuracy \t %g \t %g \n"%(max_benign_acc, max_adv_acc))
    # logfile.write(str(float(duration)) + '\n')

    # send email
    #send_email(subject="Training of epsilon = {} is done in {}".format(total_eps, parse_time(duration)), content=max_acc_string)

def main(_):
    #logfile = open('./tmp/results/DPAL_PixelDP_run1_1.txt','w')
    LR = 1e-4; #learning rate
    for fgsm_eps in [5, 10, 20, 30, 40, 50, 60]:
        for gen_ratio in [0]: #[0, 8]: #range(0, 20, 2):
            for alpha in [1.0]:# fix
                for eps2_ratio in [1]:# fix
                    logfile = open('./tmp/results/StoBatch' + str(fgsm_eps) + '_eps_' + str(0.1*(1 + gen_ratio) + 0.1) + '.txt','w')
                    train(alpha, eps2_ratio, gen_ratio, (fgsm_eps*1.0)/100.0, LR, logfile)
                    logfile.flush()
                    logfile.close();

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    if tf.gfile.Exists('./tmp/mnist_logs'):
        tf.gfile.DeleteRecursively('./tmp/mnist_logs');
    tf.gfile.MakeDirs('./tmp/mnist_logs');

    parser = argparse.ArgumentParser();
    parser.add_argument('--data_dir', type=str, default='./tmp/data',
                      help='Directory for storing data');
    FLAGS = parser.parse_args();
    tf.app.run();
