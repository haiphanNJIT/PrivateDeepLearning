'''
Differentially Private Layers
author: Hai Phan
'''
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
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

class EncLayer(object):
    '''Typical convolutional layer of MLP'''
    def __init__(self, inpt, n_filter_in, n_filter_out, filter_size, W=None, b=None, activation=tf.nn.sigmoid):
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
        self.input = inpt;
        # the output
        sum_W = tf.add(tf.nn.conv2d(inpt, self.W, strides=[1, 2, 2, 1], padding='SAME'), self.b)
        self.output = activation(sum_W) if activation is not None else sum_W
        # the input's shape
        self.inputShape = inpt.get_shape().as_list()
        # params of the layers
        self.params = [self.W, self.b]
            
    # define the Chebyshev Polinomial approximation
    def Chebyshev(self, x):
        return (-5*x**7 + 21*x**5 - 35*x**3 + 35*x + 16)/(2.0**5) # L = 7
    
    def dpChebyshev(self, x, Delta, epsilon, batch_size):
        coefficients = [-5.0, 21.0, -35.0, 35.0, 16.0] # L = 7
        for i in range(0, len(coefficients)):
            perturbFM = np.random.laplace(0.0, 1.0/(epsilon*batch_size), 1).astype(np.float32);
            perturbFM = tf.multiply(perturbFM, Delta);
            coefficients[i] += perturbFM;
        return (tf.multiply(coefficients[0], x**7) + tf.multiply(coefficients[1], x**5) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**1) + coefficients[4])/(2.0**5)
    
    # sampling hidden neurons given visible neurons
    def propup(self, v):
        '''Compute the sigmoid activation for hidden units given visible units'''
        h = tf.add(tf.nn.conv2d(v, self.W, strides=[1, 2, 2, 1], padding='SAME'), self.b)
        # Get the max value of hidden neurons
        max = tf.reduce_max(h)
        # Normalization so that h will satisfy the Riemann integrable condition on [−1, 1]
        h = h/max;
        return tf.clip_by_value(self.Chebyshev(h), 0.0, 1.0) # values of hidden neurons must be bounded [0, 1]

    # differentially private hidden terms given visible neurons
    def dp_propup(self, v, Delta, epsilon, batch_size):
        '''Compute the differentially private activation for hidden terms given visible units'''
        h = tf.add(tf.nn.conv2d(v, self.W, strides=[1, 2, 2, 1], padding='SAME'), self.b)
        max = tf.reduce_max(h)
        h = h/max;
        # hidden neurons have to be bounded in [0, 1] after the perturbation
        Chebyshev_h = tf.clip_by_value(self.dpChebyshev(h, Delta, epsilon, batch_size), 0.0, 1.0)

        return Chebyshev_h # return perturbed approximated polinomial coefficients h
    
    # transpose of convolutional RBM given hidden neurons
    def decode(self, xShape, propup, activation=tf.nn.sigmoid):
        rc_input = activation(tf.add(tf.nn.conv2d_transpose(propup, self.W,
                    tf.stack([xShape, self.inputShape[1], self.inputShape[2], self.inputShape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), self.b))
        return rc_input
    
    # reconstruct visible units from convolutional feature maps
    def decode2(self, xShape, propup, activation=tf.nn.sigmoid):
        # upsampling given hidden feature maps to obtain input's size feature maps. This step can be considered an actual deconvolution
        upsample3 = tf.image.resize_images(propup, size=(self.inputShape[1],self.inputShape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # reconstruct the original inputs
        rc_input = activation(tf.nn.conv2d(input=upsample3, filter=tf.transpose(self.W, perm=[1, 0, 3, 2]), strides=[1, 1, 1, 1], padding='SAME'))
        ###
        return rc_input
    
    # get pre-training objective function, this is use for convolutional auto-encoder
    def get_train_ops(self, xShape, learning_rate=0.1):
        propup = self.propup(self.input)
        rc_v = self.decode(xShape, propup, activation=tf.nn.sigmoid);
        self.cost = tf.reduce_sum(tf.square(rc_v - self.input))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, var_list=self.params)
        return optimizer
    
    # get differentially private pre-training energy function for convolutional RBM
    def get_train_ops2(self, xShape, Delta, epsilon = 0.25, batch_size = 3600, learning_rate=0.1):
        # compute h terms
        propup = self.dp_propup(self.input, Delta, epsilon, batch_size)
        # reconstruct v terms
        rc_v = self.decode2(xShape, propup, activation=tf.nn.sigmoid);
        # reconstruct h terms
        rc_propup = self.dp_propup(rc_v, Delta, epsilon, batch_size)
        # minimize the differentially private mean between the two energy functions: (1) contructed from the original input, (2) constructed from the reconstructed input. In other words, we use CD-1 to minimize the energy function.
        self.cost = tf.reduce_mean(tf.square(rc_propup - propup)) + tf.reduce_mean(tf.square(rc_v - self.input))
        # define AdamOptimizer optimization
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, var_list=self.params)
        return optimizer

class ConvLayer(object):
    '''Typical convolutional layer of MLP'''
    def __init__(self, inpt, filter_size, n_in, n_out, W=None, b=None, activation=tf.nn.sigmoid):
        '''
        :param inpt: input for the layer, i.e., images
        :param n_in: the number of input units
        :param n_out: the number of output units
        '''
        if W is None:
            W = tf.Variable(tf.truncated_normal([filter_size, filter_size, n_in, n_out], stddev=0.1), dtype=tf.float32, name="W")
        if b is None:
            b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32, name="b")

        self.W = W
        self.b = b
        # the output
        sum_W = tf.add(tf.nn.conv2d(inpt, self.W, strides=[1, 2, 2, 1], padding='SAME'), self.b)
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b]

class ConvFlat(object):
    '''Transforming layer from a convolutional layer to a fully connected layer'''
    def __init__(self, inpt, xShape, n_out, activation=tf.nn.sigmoid):
        '''
        inpt: tf.Tensor, (one minibatch) [None, n_in]
        xShape: the first dimention of the input, i.e., used to access the batch_size
        n_out: int, number of output units
        '''
        # Initialize parameters W and b
        shape = inpt.get_shape().as_list()
        n_in = shape[1]*shape[2]*shape[3]
        # weight
        self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), dtype=tf.float32, name="W")
        # bias
        self.b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32)
        ###
        # Compute the output
        h = tf.reshape(inpt, tf.stack([xShape, n_in]))
        sum_W = tf.matmul(h, self.W) + self.b
        # Applying normalization for the flat connected layer
        batch_mean2, batch_var2 = tf.nn.moments(sum_W,[0])
        scale2 = tf.Variable(tf.ones([n_out]))
        beta2 = tf.Variable(tf.zeros([n_out]))
        BN_norm = tf.nn.batch_normalization(sum_W,batch_mean2,batch_var2,beta2,scale2,1e-3)
        ###
        self.output = activation(BN_norm) if activation is not None else BN_norm
        #####
        # keep track of variables
        self.params = [self.W, self.b]

class HiddenLayer(object):
    '''Typical hidden layer of MLP'''
    def __init__(self, inpt, n_in, n_out, activation=tf.nn.sigmoid):
        '''
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_out: int, number of hidden units
        W, b: tf.Tensor, weight and bias
        activation: tf.op, activation function
        '''
        # weight
        self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), dtype=tf.float32, name="W")
        # bias
        self.b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32)
        # shape
        self.n_in = n_in;
        self.n_out = n_out;
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        # Applying normalization for the flat connected layer
        batch_mean2, batch_var2 = tf.nn.moments(sum_W,[0])
        scale2 = tf.Variable(tf.ones([n_out]))
        beta2 = tf.Variable(tf.zeros([n_out]))
        BN_norm = tf.nn.batch_normalization(sum_W,batch_mean2,batch_var2,beta2,scale2,1e-3)
        ###
        self.output = activation(BN_norm) if activation is not None else BN_norm
        # params
        self.params = [self.W, self.b]

class Autoencoder(object):
    '''Typical hidden layer of MLP'''
    def __init__(self, inpt, n_in, n_out, activation=tf.nn.sigmoid):
        '''
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_out: int, number of hidden units
        W, b: tf.Tensor, weight and bias
        activation: tf.op, activation function
        '''
        # weight
        self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), dtype=tf.float32, name="W")
        # bias
        self.b = tf.Variable(tf.zeros([n_out]), dtype=tf.float32)
        self.vbias = tf.Variable(tf.zeros([n_in]), dtype=tf.float32)
        self.input = inpt;
        # shape
        self.n_in = n_in;
        self.n_out = n_out;
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b, self.vbias]
    
    # compute hidden neurons given visible neurons
    def propup(self, v):
        '''Compute the sigmoid activation for hidden units given visible units'''
        h = tf.matmul(v, self.W) + self.b;
        return tf.sigmoid(h)
    
    # compute visible neurons given hidden neurons
    def propdown(self, h):
        """Compute the sigmoid activation for visible units given hidden units"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)
    
    # get differentially private pre-training cross-entropy error function
    def get_dp_train_ops(self, epsilon, data_size, first_h, learning_rate=0.1):
        # compute Laplace noise injected into coefficients h
        Delta = self.n_in*(self.n_out + 1/4 * self.n_out**2);
        perturbFM = np.random.laplace(0.0, Delta/(epsilon*data_size), self.n_out)
        perturbFM = np.reshape(perturbFM, [self.n_out]);
        
        # compute h terms
        activation_h = self.propup(self.input) + perturbFM;
        # reconstruct v terms
        activation_v = self.propdown(activation_h)
        
        '''
        Computes sigmoid cross entropy given `logits`, i.e., logits = activation_v, and z = self.input = labels.
            
        Measures the probability error in discrete classification tasks in which each
        class, i.e., each visible neuron, is independent and not mutually exclusive.
            
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
            
        `logits` and `labels` must have the same type and shape. Let denote neg_abs_logits = -abs(activation_v) = -abs(W^T * activation_h). By Applying Taylor Expansion, we have:
            
        Taylor = max(activation_v, 0) - activation_v * self.input + log(1 + exp(-abs(activation_v)));
            = max(activation_h * W^T, 0) -  self.input * (activation_h * W_fc2^T) + (math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
            = max(activation_h * W^T, 0) -  self.input * (activation_h * W_fc2^T) + (math.log(2.0) + 0.5*(-abs(activation_h * W^T)) + 1.0/8.0*(-abs(activation_h * W^T))**2)
            
        To ensure that Taylor is differentially private, we need to perturb all the coefficients, including the terms activation_h * W^T, self.input * (W^T * activation_h).
            
        Since 'self.input *' is an element-wise multiplication, activation_h can be considered coefficients of the term self.input * (W^T * activation_h). By applying Funtional Mechanism, we perturb self.input * (W^T * activation_h) as tf.matmul(activation_h + perturbFM, W^T) * self.input:
            
        activation_h += perturbFM; (Lemma 2) where
        
        Delta = self.n_in*(self.n_out + 1/4 * self.n_out**2);
        perturbFM = np.random.laplace(0.0, Delta/(epsilon*data_size), self.n_out)
        perturbFM = np.reshape(perturbFM, [self.n_out]);
            
        To allow computing gradients at zero, we define custom versions of max and abs functions [Tensorflow].
            
        Source: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_impl.py @ TensorFlow
        '''
        
        zeros = array_ops.zeros_like(activation_v, dtype=self.input.dtype)
        cond = (activation_v >= zeros)
        relu_logits = array_ops.where(cond, activation_v, zeros)
        neg_abs_logits = array_ops.where(cond, -activation_v, activation_v)
        #Taylor = math_ops.add(relu_logits - activation_v * self.input, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        if first_h == False:
            self.cost = math_ops.add(relu_logits - activation_v * self.input, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        else:
            '''In the first hidden layer, we need to perturb the activation_v * self.input term to ensure that the optimization function will not read the original data again. Instead, it only reads the perturbed data. activation_v can be considered parameters and self.input is input. This is a pair-wise multiplication operation: activation_v * self.input = {v_1*i_1, …, v_d*i_d}. The global sensitivity of an item v_j*i_j is Delta(v_j*i_j) = 2max|i_j| = 2, where j \in [1, d]. To protect v_j*i_j, we inject Laplace noise L(0, 2/(epsilon_i*data_size)) will be injected into i_j. The term becomes: v_j * (i_j + L(0, 2/(epsilon_i*data_size))).
                Since this is a pair-wise operation, the parallel composition will be applied to all the other pixels and the final privacy budget will be: epsilon_max = max(epsilon_1, …, epsilon_d).
                By doing this, the optimization of the first hidden layer will not read the private data again. In addition, the optimization of the upper layers do not use the original data to optimize the model. Instead, they use private hidden units from previous layers. Therefore, it will preserve differential privacy.
                '''
            Delta2 = 2;
            epsilon2 = 0.01
            perturbFM2 = np.random.laplace(0.0, Delta2/(epsilon2*data_size), self.n_in)
            perturbFM2 = np.reshape(perturbFM2, [self.n_in]);
            self.cost = math_ops.add(relu_logits - activation_v * (self.input + perturbFM2), math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, var_list=self.params)
        
        return optimizer


