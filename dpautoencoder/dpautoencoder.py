'''
Differentially Private Auto-Encoder
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
from dpLayers import HiddenLayer
from dpLayers import Autoencoder
from dpLayers import ConvFlat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class dpAutoEncoder(object):
    '''
    An implement of differentially private auto-encoder
    '''
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[25, 25], epsilon = 0.25, _batch_size = 280, finetuneLR = 0.01):
        '''
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the number of hidden neurons, the last item will be the number of hidden neurons in the last hidden layer
        :param epsilon: privacy budget epsilon
        :param _batch_size: the batch size
        :param finetuneLR: fine tunning learning rate
        '''
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # hidden layers
        self.params = []       # keep track of params for training
        self.last_n_in = hidden_layers_sizes[-1] # the number of hidden neurons in the last hidden layer
        self.pretrain_ops = []; # list of pretrain objective functions for hidden layers
        self.epsilon2 = 0.01; # epsilon for perturbation of the term (activation_v * self.input) in the cost function of the first hidden layer. This will ensure that private data will not be accessed again. Instead, it will optimize the function through a perturbed self.input #
        self.epsilon = epsilon - self.epsilon2; # privacy budget epsilon epsilon
        self.batch_size = _batch_size; # batch size

        # Define the input, output, Laplace noise for the output layer
        self.x = tf.placeholder(tf.float32, shape=[None, n_in], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])
        self.LaplaceNoise = tf.placeholder(tf.float32, self.last_n_in);
        ######
        
        #############################
        ##Construct the Model########
        #############################
        # Create the 1st auto-encoder layer
        Auto_Layer1 = Autoencoder(inpt=self.x, n_in = 784, n_out = hidden_layers_sizes[0], activation=tf.nn.sigmoid)
        self.layers.append(Auto_Layer1)
        self.params.extend(Auto_Layer1.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Auto_Layer1.get_dp_train_ops(epsilon = self.epsilon, data_size = 50000, first_h = True, learning_rate= 0.01))
        ###
        
        # Create the 2nd cauto-encoder layer
        Auto_Layer2 = Autoencoder(inpt=self.layers[-1].output, n_in = self.layers[-1].n_out, n_out = hidden_layers_sizes[1], activation=tf.nn.sigmoid)
        self.layers.append(Auto_Layer2)
        self.params.extend(Auto_Layer2.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Auto_Layer2.get_dp_train_ops(epsilon = self.epsilon, data_size = 50000, first_h = False, learning_rate= 0.01))
        ###
        
        # Create the flat connected hidden layer
        flat1 = HiddenLayer(inpt=self.layers[-1].output, n_in = self.layers[-1].n_out, n_out = self.last_n_in, activation=tf.nn.relu)
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
        # Fine tune with AdamOptimizer. Note that we do not fine tune the pre-trained parameters at the auto-encoder layers
        self.train_op = tf.train.AdamOptimizer(finetuneLR).minimize(self.cost, var_list=[flat1.params, self.output_layer.params], global_step = global_step)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
        ###
    
    def generateNoise(n_in, epsilon, data_size, test = False):
        Delta = 0.0;
        if test == True: # do not inject noise in the test phase
            Delta = 0.0;
        else:
            Delta = 10*(n_in + 1/4 * n_in**2); # global sensitivity for the output layer, note that 10 is the number of classes of the output layer
        # Generate the Laplace noise
        perturbFM = np.random.laplace(0.0, Delta/(epsilon*data_size), n_in)
        perturbFM = np.reshape(perturbFM, [n_in]);
        return perturbFM;

    def pretrain(self, sess, X_train, batch_size=600, pretraining_epochs=10, lr=0.01,
                    display_step=1):
        '''
        Pretrain the layers (just train the auto-encoder layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modify this function if you do not use the designed mnist)
        :param batch_size: int
        :param lr: float
        :param pretraining_epoch: int
        :param display_step: int
        '''
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        batch_num = int(math.ceil(X_train.train.num_examples / batch_size)) # The number of batch per epoch
        # Pretrain layer by layer
        for i in range(self.n_layers-1):
            # Get the cost of the current auto-encoder layer
            cost = tf.reduce_mean(self.layers[i].cost);
            # Get the objective function of the current auto-encoder layer
            train_ops = self.pretrain_ops[i]
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    x_batch, _ = X_train.train.next_batch(batch_size)
                    # training
                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch}) / batch_num
                # print out the average cost every display_step
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, trainSet, training_epochs=2400, _epsilon = 0.25, _batch_size = 280, display_step=5):
        '''
        Finetuing the network
        '''
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        LapNoise = dpAutoEncoder.generateNoise(n_in = 25, epsilon = _epsilon, data_size = 50000, test = False); #Add Laplace noise in training
        for epoch in range(training_epochs):
            #avg_cost = 0.0
            batch_num = int(math.ceil(trainSet.train.num_examples / _batch_size)) # The number of batch per epoch
            for i in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(_batch_size)
                # training
                sess.run(self.train_op, feed_dict={self.x: x_batch, self.y: y_batch, self.LaplaceNoise: LapNoise})
            # print out the average cost
            if epoch % display_step == 0:
                LapNoise_test = dpAutoEncoder.generateNoise(n_in = 25, epsilon = _epsilon, data_size = 50000, test = True); #Do not add noise when testing
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.images,
                                                       self.y: trainSet.validation.labels, self.LaplaceNoise: LapNoise_test})
                print("\tEpoch {0} \t validation accuacy: \t {1}".format(epoch, val_acc))
                #print(val_acc)

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

if __name__ == "__main__":
    # mnist examples
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dpn = dpAutoEncoder(n_in=784, n_out=10, hidden_layers_sizes=[25, 90, 25], epsilon = 8.0, _batch_size = 280, finetuneLR = 1e-3)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # set random_seed
    tf.set_random_seed(seed=1111)
    dpn.pretrain(sess, X_train=mnist)
    dpn.finetuning(sess, _epsilon = dpn.epsilon, _batch_size = dpn.batch_size, trainSet=mnist)
