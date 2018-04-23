'''
Differentially Private Logistic Regression
author: Hai Phan
'''
import numpy as np
import tensorflow as tf
import input_data
from tensorflow.python.framework import ops;
import argparse;
import pickle;
from datetime import datetime
import time
import os
import math
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

class LogisticRegression(object):
    '''Multi-class logistic regression class'''
    def __init__(self, inpt, n_in, n_out):
        '''
        inpt: tf.Tensor, (one minibatch) [None, n_in]
        n_in: int, number of input units
        n_out: int, number of output units
        '''
        # weight
        self.W = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32))
        # bias
        self.b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32)
        # activation output
        self.output = tf.nn.softmax(tf.matmul(inpt, self.W) + self.b)
        # prediction
        self.y_pred = tf.argmax(self.output, axis=1)
        # keep track of variables
        self.params = [self.W, self.b]

    def cost(self, y):
        '''
        y: tf.Tensor, the target of the input
        '''
        # cross_entropy
        return -tf.reduce_mean(tf.reduce_sum(y * tf.log(self.output), axis=1))

    def accuarcy(self, y):
        '''errors'''
        correct_pred = tf.equal(self.y_pred, tf.argmax(y, axis=1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class dpLogisticRegression(object):
    '''Multi-class logistic regression class'''
    def __init__(self, inpt, n_in, n_out, LaplaceNoise):
        '''
        inpt: tf.Tensor, (one minibatch) [None, n_in]
        n_in: int, number of input units
        n_out: int, number of output units
        LaplaceNoise: Laplace noise
        '''
        # weight
        self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), dtype=tf.float32, name="W")
        # bias
        self.b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32)
        # activation output
        inpt = tf.clip_by_value(inpt, 0, 1) # hidden neurons must be bounded in [0, 1], sigmoid activation function does not need this bound
        inpt += LaplaceNoise;
        self.output = tf.matmul(inpt, self.W) + self.b
        # prediction
        self.y_pred = tf.argmax(self.output, axis=1)
        # keep track of variables
        self.params = [self.W, self.b]
    
    def cost(self, y):
        '''
        y: tf.Tensor, the target of the input
        '''
        #############################
        ##Define loss and Optimizer##
        #############################
        '''
            Computes sigmoid cross entropy given `logits`.
            
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
            
            To ensure that Taylor is differentially private, we need to perturb all the coefficients, including the terms h * W, z * h * W.
            
            Since '* z' is an element-wise multiplication, h can be considered coefficients of the term z * h. By applying Funtional Mechanism, we perturb (z * h) * W as tf.matmul(h + LaplaceNoise, W) * z:
            
            h += LaplaceNoise; where
            
            LaplaceNoise = np.random.laplace(0.0, scale, hk)
            LaplaceNoise = np.reshape(LaplaceNoise, [hk]);
            
            This has been done in the previous code block:
            
            "LaplaceNoise = generateNoise(.)
            ...
            inpt += LaplaceNoise;"
            
            where scale = 10*(hk + 1/4 * hk**2)/(epsilon*batch_size); (Lemma 5)
            
            To allow computing gradients at zero, we define custom versions of max and abs functions [Tensorflow].
            
            Source: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_impl.py @ TensorFlow
        '''
        zeros = array_ops.zeros_like(self.output, dtype=self.output.dtype)
        cond = (self.output >= zeros)
        relu_logits = array_ops.where(cond, self.output, zeros)
        neg_abs_logits = array_ops.where(cond, -self.output, self.output)
        #Taylor = math_ops.add(relu_logits - y_conv * y_, math_ops.log1p(math_ops.exp(neg_abs_logits)))
        Taylor = math_ops.add(relu_logits - self.output * y, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        return Taylor
    
    def accuarcy(self, y):
        '''prediction accuracy'''
        correct_pred = tf.equal(self.y_pred, tf.argmax(y, axis=1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
