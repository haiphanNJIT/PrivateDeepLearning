'''
Multi-Layer Perceptron Class
author: Hai Phan
'''
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
import math

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
        #return (-x**3 + 3*x + 2)/(2.0**2) # L = 3
        #return (5*x**5 + 2*x**4 - 14*x**3 - 4*x**2 + 17*x + 10)/(2.0) # L = 5
        #return (-x**2 + 2*x + 3)/(2.0**2) # L = 2
    
    def dpChebyshev(self, x, Delta, epsilon, batch_size):
        coefficients = [-5.0, 21.0, -35.0, 35.0, 16.0] # L = 7
        #coefficients = [-1.0, 3.0, 2.0] # L = 3
        #coefficients = [-1.0, 2.0, 3.0] # L = 2
        #coefficients = [5.0, 2.0, -14.0, -4.0, 17.0, 10.0] # L = 5
        for i in range(0, len(coefficients)):
            perturbFM = np.random.laplace(0.0, 1.0/(epsilon*batch_size), 1).astype(np.float32);
            perturbFM = tf.multiply(perturbFM, Delta);
            coefficients[i] += perturbFM;
        return (tf.multiply(coefficients[0], x**7) + tf.multiply(coefficients[1], x**5) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**1) + coefficients[4])/(2.0**5) # L = 7
        #return (tf.multiply(coefficients[0], x**3) + tf.multiply(coefficients[1], x) + coefficients[2])/(2.0**2) # L = 3
        #return (tf.multiply(coefficients[0], x**2) + tf.multiply(coefficients[1], x) + coefficients[2])/(2.0**2) # L = 2
        #return (tf.multiply(coefficients[0], x**5) + tf.multiply(coefficients[1], x**4) + tf.multiply(coefficients[2], x**3) + tf.multiply(coefficients[3], x**2) + tf.multiply(coefficients[4], x**1) + coefficients[5])/(2.0) # L = 5
    
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
    
    # transpose of convolutional RBM given hidden neurons, this is use for convolutional auto-encoder
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
        self.cost = tf.reduce_mean(tf.abs(rc_propup - propup)) + tf.reduce_mean(tf.abs(rc_v - self.input))
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
        self.input = inpt;
        # shape
        self.n_in = n_in;
        self.n_out = n_out;
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b]

class MLP(object):
    '''Multi-layer perceptron class'''
    def __init__(self, inpt, n_in, n_hidden, n_out):
        '''
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_hidden: int, number of hidden units
        n_out: int, number of output units
        '''
        # hidden layer
        self.hiddenLayer = HiddenLayer(inpt, n_in=n_in, n_out=n_hidden)
        # output layer (logistic layer)
        self.outputLayer = LogisticRegression(self.hiddenLayer.output, n_in=n_hidden,
                                              n_out=n_out)
        # L1 norm
        self.L1 = tf.reduce_sum(tf.abs(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.abs(self.outputLayer.W))
        # L2 norm
        self.L2 = tf.reduce_sum(tf.square(self.hiddenLayer.W)) + \
                  tf.reduce_sum(tf.square(self.outputLayer.W))
        # cross_entropy cost function
        self.cost = self.outputLayer.cost
        # accuracy function
        self.accuracy = self.outputLayer.accuarcy

        # params
        self.params = self.hiddenLayer.params + self.outputLayer.params
        # keep track of input
        self.input = inpt




