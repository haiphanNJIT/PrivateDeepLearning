########################################################################
# Author: NhatHai Phan, Han Hu
# License: Apache 2.0
########################################################################

'''
Multi-Layer Perceptron
'''

import numpy as np
import tensorflow as tf
#import input_data
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
#from logisticRegression import LogisticRegression
import math

AECODER_VARIABLES = 'AECODER_VARIABLES'
AECODER_UPDATES = 'AECODER_UPDATES'
CONV_VARIABLES = 'CONV_VARIABLES'
CONV_UPDATES = 'CONV_UPDATES'

class EncLayer(object):
    '''Typical convolutional layer of MLP'''
    def __init__(self, inpt, n_filter_in, n_filter_out, filter_size, W=None, b=None, strides=2, activation=tf.nn.sigmoid):
        '''
        :param inpt: input for the layer, i.e., images
        :param n_filter_in: the number of input feature maps
        :param n_filter_out: the number of input feature maps
        :param filter_size: the dimension of convolutional filter [filter_size, filter_size]
        '''
        # Initialize parameters: W and b
        if W is None:
            W = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_filter_in, n_filter_out], stddev=0.1), dtype=tf.float32, name="W")
        if b is None:
            b = tf.Variable(tf.zeros([n_filter_out]), dtype=tf.float32, name="b")
        self.W = W
        self.b = b
        self.input = inpt
        # the input's shape
        self.inputShape = inpt.get_shape().as_list()
        # params of the layers
        self.params = [self.W, self.b]
        self.strides = strides
            
    # define the Chebyshev Polinomial approximation
    def Chebyshev(self, x):
        return (-5*x**7 + 21*x**5 - 35*x**3 + 35*x + 16)/(2.0**5) # L = 7
    
    def dpChebyshev(self, x, Delta, epsilon, batch_size):
        coefficients = [-5.0, 21.0, -35.0, 35.0, 16.0] # L = 7
        for i in range(0, len(coefficients)):
            perturbFM = np.random.laplace(0.0, 1.0/(epsilon*batch_size), 1).astype(np.float32)
            perturbFM = tf.multiply(perturbFM, Delta)
            coefficients[i] += perturbFM
        return (tf.multiply(coefficients[0], x**7) + tf.multiply(coefficients[1], x**5) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**1) + coefficients[4])/(2.0**5) # L = 7
        
    # sampling hidden neurons given visible neurons
    def propup(self, v, W, b):
      '''Compute the sigmoid activation for hidden units given visible units'''
      h = tf.nn.relu(tf.nn.conv2d(v, W, strides=[1, self.strides, self.strides, 1], padding='SAME') + b)
      h = tf.layers.batch_normalization(h, scale=True, training=True, trainable=True, reuse=tf.AUTO_REUSE, name='bn_enc')        
      return tf.clip_by_value(h, -1, 1) # values of hidden neurons must be bounded [-1, 1]

    # differentially private hidden terms given visible neurons
    def dp_propup(self, v, Delta, epsilon, batch_size):
        '''Compute the differentially private activation for hidden terms given visible units'''
        h = tf.add(tf.nn.conv2d(v, self.W, strides=[1, self.strides, self.strides, 1], padding='SAME'), self.b)
        max = tf.reduce_max(h)
        h = h/max
        # hidden neurons have to be bounded in [0, 1] after the perturbation
        Chebyshev_h = tf.clip_by_value(self.dpChebyshev(h, Delta, epsilon, batch_size), 0.0, 1.0)

        return Chebyshev_h # return perturbed approximated polinomial coefficients h
    
    # transpose of convolutional RBM given hidden neurons, this is use for convolutional auto-encoder
    def decode(self, xShape, propup, activation=tf.nn.sigmoid):
        rc_input = activation(tf.add(tf.nn.conv2d_transpose(propup, self.W,
                    tf.stack([xShape, self.inputShape[1], self.inputShape[2], self.inputShape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), self.b))
        return rc_input
    
    # reconstruct visible units from convolutional feature maps
    def decode2(self, xShape, propup, W, b, activation=tf.nn.sigmoid):
        # upsampling given hidden feature maps to obtain input's size feature maps. This step can be considered an actual deconvolution
        upsample3 = tf.image.resize_images(propup, size=(self.inputShape[1],self.inputShape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # reconstruct the original inputs
        rc_input = activation(tf.nn.conv2d(input=upsample3, filter=tf.transpose(W, perm=[1, 0, 3, 2]), strides=[1, 1, 1, 1], padding='SAME'))
        ###
        return rc_input

    # get pre-training objective function, this is use for convolutional auto-encoder
    def get_train_ops(self, xShape, learning_rate=0.1):
        propup = self.propup(self.input)
        rc_v = self.decode(xShape, propup, activation=tf.nn.sigmoid)
        self.cost = tf.reduce_sum(tf.square(rc_v - self.input))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, var_list=self.params)
        return optimizer
    
    # get differentially private pre-training energy function for convolutional RBM
    def get_train_ops2(self, xShape, Delta, epsilon, batch_size, learning_rate, W, b, perturbFMx, perturbFM_h):
        # compute Laplace noise injected into coefficients h
        print('inside enc layer get_train_ops2')
        # compute h terms
        propup = self.propup(self.input + perturbFMx, W, b) + perturbFM_h
        print('propup')
        print(propup)
        # reconstruct v terms
        rc_v = self.decode2(xShape, propup, W, b, activation=tf.nn.relu)
        print('rc_v')
        print(rc_v)
        
        zeros = array_ops.zeros_like(rc_v, dtype=self.input.dtype)
        cond = (rc_v >= zeros)
        relu_logits = array_ops.where(cond, rc_v, zeros)
        neg_abs_logits = array_ops.where(cond, -rc_v, rc_v)
        self.cost = tf.abs(math_ops.add(relu_logits - rc_v * (self.input + perturbFMx), math.log(2.0) + 0.5*neg_abs_logits))
        
        return self.cost






