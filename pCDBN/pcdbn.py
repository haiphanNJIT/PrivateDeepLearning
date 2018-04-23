'''
Differentially Private Convolutional Deep Belief Network
author: Hai Phan
'''
import timeit
import numpy as np
import tensorflow as tf
import input_data
import math
import os
from logisticRegression import LogisticRegression
from logisticRegression import dpLogisticRegression
from mlp import ConvLayer
from mlp import EncLayer
from mlp import ConvFlat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class pCDBN(object):
    '''
    An implement of differentially private convolutional deep belief network
    '''
    def __init__(self, n_in=784, n_out=10, filter_size = 5, hidden_layers_sizes=[10, 10, 25], epsilon = 0.25, _batch_size = 3600, finetuneLR = 0.01):
        '''
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param filter_size: the dimension of convolutional filter [filter_size, filter_size]
        :param hidden_layers_sizes: list or tuple, the number of convolutional feature maps, the last item will be the number of hidden neurons in the last hidden layer
        :param epsilon: privacy budget epsilon
        :param _batch_size: the batch size
        :param finetuneLR: fine tunning learning rate
        '''
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # convolutional and hidden layers
        self.params = []       # keep track of params for training
        self.last_n_in = hidden_layers_sizes[-1] # the number of hidden neurons in the last hidden layer
        self.pretrain_ops = []; # list of pretrain objective functions for convolutional layers
        self.epsilon = epsilon; # privacy budget epsilon epsilon
        self.batch_size = _batch_size; # batch size

        # Define the input, output, Laplace noise for the output layer, and Delta for the pretrain convolutional layers
        self.x = tf.placeholder(tf.float32, shape=[None, n_in], name='x')
        # ensure 2-d is converted to square tensor.
        if len(self.x.get_shape()) == 2:
            x_dim = np.sqrt(self.x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(self.x, [-1, x_dim, x_dim, 1])
        elif len(self.x.get_shape()) == 4:
            x_tensor = self.x
        else:
            raise ValueError('Unsupported input dimensions')
        image = x_tensor
    
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])
        self.LaplaceNoise = tf.placeholder(tf.float32, 25);
        self.Delta = tf.placeholder(tf.float32, 1);
        ######
        
        #############################
        ##Construct the Model########
        #############################
        # Create the 1st convolutional restricted boltzmann layer
        Enc_Layer1 = EncLayer(inpt=image, n_filter_in = 1, n_filter_out = hidden_layers_sizes[0], filter_size = 5, activation=tf.nn.sigmoid)
        self.layers.append(Enc_Layer1)
        self.params.extend(Enc_Layer1.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Enc_Layer1.get_train_ops2(xShape = tf.shape(image)[0], Delta = self.Delta, epsilon = self.epsilon, batch_size = self.batch_size, learning_rate= 0.01))
        ###
        
        # Create the 2nd convolutional restricted boltzmann layer
        Enc_Layer2 = EncLayer(inpt=self.layers[-1].output, n_filter_in = hidden_layers_sizes[0], n_filter_out = hidden_layers_sizes[1], filter_size = 5, activation=tf.nn.sigmoid)
        self.layers.append(Enc_Layer2)
        self.params.extend(Enc_Layer2.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Enc_Layer2.get_train_ops2(xShape = tf.shape(image)[0], Delta = self.Delta, epsilon = self.epsilon, batch_size = self.batch_size, learning_rate= 0.01))
        ###
        
        # Create the flat connected hidden layer
        flat1 = ConvFlat(inpt=self.layers[-1].output, xShape = tf.shape(image)[0], n_out = self.last_n_in, activation=tf.nn.relu)
        self.layers.append(flat1)
        self.params.extend(flat1.params)
        ###
        
        # Create the output layer
        # We use the differentially private Logistic Regression (dpLogisticRegression) layer as the objective function
        self.output_layer = dpLogisticRegression(inpt=self.layers[-1].output, n_in = self.last_n_in, n_out=n_out, LaplaceNoise = self.LaplaceNoise)
        # We can also use the non-differentially private layer: LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1], n_out=n_out)
        self.params.extend(self.output_layer.params)
        ###

        #######################################
        ##Define Fine Tune Cost and Optimizer##
        #######################################
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # train_op for finetuning with AdamOptimizer
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(finetuneLR, global_step, 700, 0.96, staircase=True); # learning rate decay can be carefully used
        # Fine tune with AdamOptimizer. Note that we do not fine tune the pre-trained parameters at the convolutional layers
        self.train_op = tf.train.AdamOptimizer(finetuneLR).minimize(self.cost, var_list=[flat1.params, self.output_layer.params], global_step = global_step)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
        ###
        
    def getDelta(self, v, W, b):
        # Set _W and _b to be 1 with the shape of W and b
        _W = tf.constant(1.0, shape=W.get_shape())
        _b = tf.constant(1.0, shape=b.get_shape())
        ###
        # Compute hidden neurons in the convolutional layer
        h = tf.add(tf.nn.conv2d(v, _W, strides=[1, 2, 2, 1], padding='SAME'), _b)
        # Get the max value of hidden neurons
        max = tf.reduce_max(h)
        # Normalization so that h will satisfy the Riemann integrable condition on [−1, 1]
        h = h/max;
        # Approxiate hidden neurons by using Chebyshev Polinomial Approximations
        Chebyshev_h = tf.clip_by_value(EncLayer.Chebyshev(self = self, x = h), 0.0, 1.0)
        # Compute the global sensitivity Delta
        Delta = 2.0*tf.reduce_max(tf.abs(tf.reduce_sum(Chebyshev_h, axis=[1, 2])))
        # Compute max(v_terms)
        v_shape = v.get_shape().as_list()
        if (len(v_shape) > 2):
            Delta += 2.0*tf.reduce_max(tf.abs(tf.reduce_sum(v, axis=[1, 2])))
        else:
            Delta += 2.0*tf.reduce_max(tf.abs(tf.reduce_sum(v, axis=[1])))
        return Delta
    
    def generateNoise(n_in, epsilon, batch_size, test = False):
        Delta = 0.0;
        if test == True: # do not inject noise in the test phase
            Delta = 0.0;
        else:
            Delta = 10*(n_in + 1/4 * n_in**2); # global sensitivity for the output layer, note that 10 is the number of classes of the output layer
        # Generate the Laplace noise
        perturbFM = np.random.laplace(0.0, Delta/(epsilon*batch_size), n_in)
        perturbFM = np.reshape(perturbFM, [n_in]);
        return perturbFM;

    def pretrain(self, sess, X_train, batch_size=3600, pretraining_epochs=10, lr=0.1, k=1,
                    display_step=1):
        '''
        Pretrain the layers (just train the Convolutional RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modify this function if you do not use the designed mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        '''
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        batch_num = int(math.ceil(X_train.train.num_examples / batch_size)) # The number of batch per epoch
        # Pretrain layer by layer
        for i in range(self.n_layers-1):
            # Get the cost of the current Convolutional RBM layer
            cost = self.layers[i].cost;
            # Get the objective function of the current Convolutional RBM layer
            train_ops = self.pretrain_ops[i]
            # Get the Delta operation of the current Convolutional RBM layer
            delta = self.getDelta(v = self.layers[i].input, W = self.layers[i].W, b = self.layers[i].b)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch, _ = X_train.train.next_batch(batch_size)
                    # Compute the actual Delta with the current parameters of the current Convolutional RBM layer
                    _Delta = delta.eval(session=sess, feed_dict={self.x: x_batch});
                    #print(np.reshape(_Delta, [1]))
                    # training
                    sess.run(train_ops, feed_dict={self.x: x_batch, self.Delta: np.reshape(_Delta, [1])})
                    # cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch, self.Delta: np.reshape(_Delta, [1])}) / batch_num
                # print out the average cost every display_step
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, trainSet, training_epochs=2400, _epsilon = 0.25, _batch_size = 3600, display_step=5):
        '''
        Finetuing the network
        '''
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        
        for epoch in range(training_epochs):
            #avg_cost = 0.0
            batch_num = int(math.ceil(trainSet.train.num_examples / _batch_size)) # The number of batch per epoch
            for i in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(_batch_size)
                # training
                LapNoise = pCDBN.generateNoise(n_in = 25, epsilon = _epsilon, batch_size = _batch_size, test = False); #Add Laplace noise in training
                sess.run(self.train_op, feed_dict={self.x: x_batch, self.y: y_batch, self.LaplaceNoise: LapNoise})
            # print out the average cost
            if epoch % display_step == 0:
                LapNoise = pCDBN.generateNoise(n_in = 25, epsilon = _epsilon, batch_size = _batch_size, test = True); #Do not add noise when testing
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.images,
                                                       self.y: trainSet.validation.labels, self.LaplaceNoise: LapNoise})
                #print("\tEpoch {0} \t validation accuacy: \t {1}".format(epoch, val_acc))
                print(val_acc)

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dbn = pCDBN(n_in=784, n_out=10, filter_size = 5, hidden_layers_sizes=[32, 64, 25], epsilon = 0.2, _batch_size = 3600, finetuneLR = 2e-4)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # set random_seed for the reproducibility
    tf.set_random_seed(seed=1111)
    dbn.pretrain(sess, X_train=mnist)
    dbn.finetuning(sess, _epsilon = dbn.epsilon, _batch_size = dbn.batch_size, trainSet=mnist)
