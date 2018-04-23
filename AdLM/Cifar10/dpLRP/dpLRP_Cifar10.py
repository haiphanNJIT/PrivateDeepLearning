"""
Differentially Private LRP on Cifar-10 dataset
Author: Hai Phan
"""

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import lrp
import pandas as pd
from pylab import rcParams
import os
from keras.datasets import cifar10

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_index_in_epoch_train = 0;
_index_in_epoch_test = 0;
rcParams['figure.figsize'] = 8, 10
#mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
_num_examples = len(x_train)
_num_test_examples = len(x_test)
I = y_train[:,0].astype(int)
y_train = np.zeros([x_train.shape[0],np.unique(y_train).size])
y_train[np.arange(y_train.shape[0]),I] = 1
#print(y_train)
I = y_test[:,0].astype(int)
y_test = np.zeros([x_test.shape[0],np.unique(y_test).size])
y_test[np.arange(y_test.shape[0]),I] = 1
#print(y_test)

def next_batch_train(batch_size, _num_examples):
    global _index_in_epoch_train;
    global x_train
    global y_train
    """Return the next `batch_size` examples from this data set."""
    start = _index_in_epoch_train
    _index_in_epoch_train += batch_size
    if _index_in_epoch_train > _num_examples:
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        x_train = x_train[perm]
        y_train = y_train[perm]
        # Start next epoch
        start = 0
        _index_in_epoch_train = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch_train
    return x_train[start:end], y_train[start:end]

def next_batch_test(batch_size, _num_examples):
    global _index_in_epoch_test;
    global x_test
    global y_test
    """Return the next `batch_size` examples from this data set."""
    start = _index_in_epoch_test
    _index_in_epoch_test += batch_size
    if _index_in_epoch_test > _num_examples:
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        x_test = x_test[perm]
        y_test = y_test[perm]
        # Start next epoch
        start = 0
        _index_in_epoch_test = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch_test
    return x_test[start:end], y_test[start:end]

batch_size = 50
total_batch = int(_num_examples/batch_size)
num_epochs = 50

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10], name="truth")

#Set the weights for the network
xavier = tf.contrib.layers.xavier_initializer_conv2d()
conv1_weights = tf.get_variable(name="c1", initializer=xavier, shape=[5, 5, 3, 64])
conv1_biases = tf.Variable(tf.zeros([64]))
conv2_weights = tf.get_variable(name="c2", initializer=xavier, shape=[5, 5, 64, 64])
conv2_biases = tf.Variable(tf.zeros([64]))
conv3_weights = tf.get_variable(name="c3", initializer=xavier, shape=[4, 4, 64, 64])
conv3_biases = tf.Variable(tf.zeros([64]))
fc1_weights = tf.Variable(tf.truncated_normal([4 * 4 * 64, 192], stddev=0.1))
fc1_biases = tf.Variable(tf.zeros([192]))
fc2_weights = tf.Variable(tf.truncated_normal([192, 10], stddev=0.1))
fc2_biases = tf.Variable(tf.zeros([10]))

#Stack the Layers
reshaped_input = tf.reshape(x, [-1, 32, 32, 3], name="absolute_input")
#layer 1
conv1 = tf.nn.conv2d(reshaped_input, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#layer 2
conv2 = tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
#layer 3
conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
norm3 = tf.nn.lrn(relu3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#layer 4
pool_shape = pool3.get_shape().as_list()
reshaped = tf.reshape(pool3, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
relu4 = tf.add(tf.matmul(reshaped, fc1_weights), fc1_biases)
#layer 5
y = tf.add(tf.matmul(relu4, fc2_weights), fc2_biases, name="absolute_output")

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# Train the model
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
for i in tqdm(range(num_epochs)):
    for i in range(total_batch):
        batch_x, batch_y = next_batch_train(batch_size, _num_examples);
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_acc = []
train_acc = []
for i in tqdm(range(total_batch)):
    batch_x, batch_y = next_batch_test(batch_size, _num_test_examples);
    test_acc.append(sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))
    batch_x, batch_y = next_batch_train(batch_size, _num_examples);
    train_acc.append(sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))
print (np.mean(train_acc), np.mean(test_acc))

#Run LRP with Deep Taylor Decomposition on the output of the network
LRP_batch_size = 50000;
#batch_x, batch_y = next_batch_train(LRP_batch_size, _num_examples);
F_list = lrp.lrp(y*y_, 0, 10, return_flist=True)
im_list = lrp.get_lrp_im(sess, F_list[-1], reshaped_input, y_, np.reshape(x_train, (LRP_batch_size, 32,32, 3)), y_train)

########################################
##Compute Differentially Private LRP####
########################################
image_size = 32;
d = (image_size-6*2)**2; #The number of coefficients/features in each image; crop images to 20 instead of 32
epsilon = 0.5; #epsilon for dpLRP
D = 50000; #size of the dataset
inflation = 1.0
augmentation = 0.3

#Average LRP#
Final_AvgR = np.abs(np.mean(im_list, axis=(0,3)));

#Compute min and max in LRP#
min_R = np.min(Final_AvgR);
max_R = np.max(Final_AvgR);

#Final_AvgR = (Final_AvgR - np.mean(Final_AvgR))/np.std(Final_AvgR)
#Normalize and Perturb LRP to enforce Differential Privacy#
for k in tqdm(range(0, image_size)):
    for j in range(0, image_size):
        #(1) normalized LRP with inflation rate is 1, and (2) add Laplace noise (2*d/epsilon*D)
        Final_AvgR[k,j] = np.clip((Final_AvgR[k,j] - min_R)**inflation/(max_R - min_R)**inflation + augmentation, 0, 1) + np.random.laplace(0.0, 2*d/(epsilon*D));
#Augment and Export the differentially private LRP. This step does not affect differential priacy since we do not access the data again
with open(os.getcwd() + '/LRP_0_5_v1.txt', "w") as text_file:
    for k in range(0, image_size):
        for j in range(0, image_size):
            if ((k >= 4+2) and (k < image_size-4-2)) and ((j >= 4+2) and (j < image_size-4-2)):
                text_file.write(str(Final_AvgR[k,j] + augmentation) + " ");
            else:
                text_file.write(str(Final_AvgR[k,j]) + " ");
        text_file.write("\n");
    text_file.close();
########################################

#Visualize the produced heatmaps
'''for b, im in zip(batch_x, im_list):
    im = im.sum(axis=2)
    """for i in range(0, 32):
        print(im[i])"""
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(np.reshape(b, (32,32,3)))
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(im, (32,32)), cmap="gray")
    plt.show()'''

