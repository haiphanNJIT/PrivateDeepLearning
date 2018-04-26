# -*- coding: utf-8 -*-
'''
An implementation of the Adaptive Laplace Mechanism
author: Hai Phan
'''
import numpy as np;
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1);
  return tf.Variable(initial);

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape);
  return tf.Variable(initial);
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME');                          

def generateIdLMNoise(image_size, Delta2, _beta, epsilon2, L):
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

def main(_):
  #############################
  ##Hyper-parameter Setting####
  #############################
  hk = 25; #number of hidden units at the last layer
  Delta2 = 2*14*14*25; #global sensitivity for the first hidden layer
  Delta3 = 10*(hk + 1/4 * hk**2); #global sensitivity for the output layer
  D = 60000; #size of the dataset
  L = 1800; #batch size
  image_size = 28;
  padding = 4;
  #numHidUnits = 14*14*32 + 7*7*64 + M + 10; #number of hidden units
  epsilon1 = 0.075; #epsilon for dpLRP
  epsilon2 = 0.075; #epsilon for the first hidden layer
  epsilon3 = 0.15; #epsilon for the last hidden layer
  uncert = 0.1; #uncertainty modeling at the output layer
  infl = 1; #inflation rate in the privacy budget redistribution
  R_lowerbound = 1e-5; #lower bound of the LRP
  c = [0, 40, 50, 200] #norm bounds
  epochs = 2400; #number of epochs
  T = int(D/L*epochs + 1); #number of steps T
  step_for_epoch = int(D/L); #number of steps for one epoch
  LR = 5e-4; #learning rate
  LRPfile = os.getcwd() + '/Relevance_R_0_075.txt';
  #############################
  mnist = input_data.read_data_sets("MNIST_data/", one_hot = True);
  
  #############################
  ##Redistribute the noise#####
  #############################
  #Step 1: Load differentially private LRP (dpLRP)#
  R = [];
  with open(LRPfile, "r") as ins:
      for line in ins:
          array = [];
          array = line.split();
          tempArray = [];
          for i in range(0, len(array)):
            tempArray.append(float(array[i]));
          R.append(tempArray);
  #End Step 1#
  
  #Step 2: Redistribute the noise#
  sum_R = 0;
  for k in range(0,image_size+padding):
    for j in range(0, image_size+padding):
       if R[k][j] < R_lowerbound:
           R[k][j] = R_lowerbound;
       sum_R += R[k][j]**infl;
  _beta = [[0.0 for x in range(image_size+padding)] for y in range(image_size+padding)];
  for k in range(0,image_size+padding):
    for j in range(0, image_size+padding):
       #Compute Privacy Budget Redistribution Vector#
       _beta[k][j] = ((image_size+padding)**2)*(R[k][j]**infl)/sum_R;
       if _beta[k][j] < R_lowerbound:
           _beta[k][j] = R_lowerbound;
  #############################
  
  #############################
  ##Construct the Model########
  #############################
  #Step 4: Randomly initiate the noise, Compute 1/|L| * Delta3 for the output layer#
  
  #Compute the 1/|L| * Delta3 for the last hidden layer#
  loc, scale3, scale4 = 0., Delta3/(epsilon3*L), Delta3/(uncert*L);
  ###
  #End Step 4#
  
  #Step 5: Create the model#
  x = tf.placeholder(tf.float32, [None, image_size*image_size]);
  x_image = tf.reshape(x, [-1,image_size,image_size,1]);
  z = x_image;
  W_conv1 = weight_variable([5, 5, 1, 32]);
  b_conv1 = bias_variable([32]);
  #inject noise into the input#
  noise = tf.placeholder(tf.float32, [None, image_size, image_size, 1]);
  z += noise;
  z = tf.clip_by_value(z, -10, 10) #Clip the values of each input feature.
  ###
  h_conv1 = tf.nn.relu(conv2d(z, W_conv1) + b_conv1);
  h_pool1 = max_pool_2x2(h_conv1);

  W_conv2 = weight_variable([5, 5, 32, 64]);
  b_conv2 = bias_variable([64]);
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
  h_pool2 = max_pool_2x2(h_conv2);
  
  W_fc1 = weight_variable([7 * 7 * 64, hk]);
  b_fc1 = bias_variable([hk]);
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]);
  z2 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1;
  #Applying normalization for the flat connected layer h_fc1#
  batch_mean2, batch_var2 = tf.nn.moments(z2,[0])
  scale2 = tf.Variable(tf.ones([hk]))
  beta2 = tf.Variable(tf.zeros([hk]))
  BN_norm = tf.nn.batch_normalization(z2,batch_mean2,batch_var2,beta2,scale2,1e-3)
  ###
  h_fc1 = tf.nn.relu(BN_norm);
  h_fc1 = tf.clip_by_value(h_fc1, 0, 1) #hidden neurons must be bounded in [0, 1]
  '''perturbFM = np.random.laplace(0.0, scale3, hk)
  perturbFM = np.reshape(perturbFM, [hk]);'''
  perturbFM = tf.placeholder(tf.float32, [hk]);
  h_fc1 += perturbFM;
  #Sometime bound the 2-norm of h_fc1 can help to stablize the training process. Use with care.
  #h_fc1 = tf.clip_by_norm(h_fc1, c[2], 1);
  
  #place holder for dropout, however we do not use dropout in this code#
  keep_prob = tf.placeholder(tf.float32);
  #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
  ###

  W_fc2 = weight_variable([hk, 10]);
  b_fc2 = bias_variable([10]);
  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2;
  '''Model uncertainty by injecting noise into the output layer. This step does not relate to differential privacy preservation. It simply improve the robustness of the model and force the model to reach flatter optimal areas.'''
  '''perturb = np.random.laplace(0.0, scale4, 10) #10 is the number of classes [0, 1, ..., 9]
  perturb = np.reshape(perturb, [10]);
  y_conv += perturb;
  #After modeling the uncertainty, we apply norm bound to bound 2-norm of the output#
  y_conv = tf.clip_by_norm(y_conv, c[3], 1)'''
  ###
  #Define a place holder for the output label#
  y_ = tf.placeholder(tf.float32, [None, 10]);
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
    
    To ensure that Taylor is differentially private, we need to perturb all the coefficients, including the terms h_fc1 * W_fc2, y_ * h_fc1 * W_fc2.
    
    Since '* y_' is an element-wise multiplication and 'y_' is one_hot encoding, h_fc1 can be considered coefficients of the term y_ * h_fc1. By applying Funtional Mechanism, we perturb (y_ * h_fc1) * W_fc2 as tf.matmul(h_fc1 + perturbFM, W_fc2) * y_:
    
    h_fc1 += perturbFM; where
    
    perturbFM = np.random.laplace(0.0, scale3, hk)
    perturbFM = np.reshape(perturbFM, [hk]);
    
    This has been done in the previous code block:
    
    "perturbFM = np.random.laplace(0.0, scale3, hk)
    perturbFM = np.reshape(perturbFM, [hk]);
    h_fc1 += perturbFM;"
    
    where scale3 = Delta3/(epsilon3*L) = 10*(hk + 1/4 * hk**2)/(epsilon3*L); (Lemma 5)
    
    To allow computing gradients at zero, we define custom versions of max and abs functions [Tensorflow].
    
    Source: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/nn_impl.py @ TensorFlow
  '''
  zeros = array_ops.zeros_like(y_conv, dtype=y_conv.dtype)
  cond = (y_conv >= zeros)
  relu_logits = array_ops.where(cond, y_conv, zeros)
  neg_abs_logits = array_ops.where(cond, -y_conv, y_conv)
  #Taylor = math_ops.add(relu_logits - y_conv * y_, math_ops.log1p(math_ops.exp(neg_abs_logits)))
  Taylor = math_ops.add(relu_logits - y_conv * y_, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
  '''Some time, using learning rate decay can help to stablize training process. However, use this carefully, since it may affect the convergent speed.'''
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(LR, global_step, 30000, 0.3, staircase=True)
  train_step = tf.train.AdamOptimizer(LR).minimize(Taylor, global_step=global_step);
  sess = tf.InteractiveSession();

  #Define the correct prediction and accuracy#
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1));
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
  sess.run(tf.initialize_all_variables());
  #############################

  start_time = time.time();
  for i in range(T):
    batch = mnist.train.next_batch(L); #Get a random batch.
    #The number of epochs we print out the result. Print out the result every 5 epochs.
    if i % int(5*step_for_epoch) == 0:
      #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], noise: LapNoise, keep_prob: 1.0});
      #print("step \t %d \t training accuracy \t %g"%(i, train_accuracy));
      Lnoise1 = generateNoise(image_size, 0, _beta, epsilon2, L);
      Lnoise2 = generateHkNoise(hk, 0, epsilon3, L);
      print("step \t %d \t test accuracy \t %g"%(i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, noise: Lnoise1, perturbFM: Lnoise2, keep_prob: 1.0})));
    LapNoise = generateNoise(image_size, Delta2, _beta, epsilon2, L);
    LapNoise2 = generateHkNoise(hk, Delta3, epsilon3, L); #Add noise when training
    train_step.run(feed_dict={x: batch[0], y_: batch[1], noise: LapNoise, perturbFM: Lnoise2,keep_prob: 0.5});
  duration = time.time() - start_time;
  Lnoise1 = generateNoise(image_size, 0, _beta, epsilon2, L);
  Lnoise2 = generateHkNoise(hk, 0, epsilon3, L);
  print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, noise: Lnoise1, perturbFM: Lnoise2, keep_prob: 1.0})); #print accuracy on test data#
  print(float(duration)); #print running time duration#

if __name__ == '__main__':
  if tf.gfile.Exists('/tmp/mnist_logs'):
    tf.gfile.DeleteRecursively('/tmp/mnist_logs');
  tf.gfile.MakeDirs('/tmp/mnist_logs');
  
  parser = argparse.ArgumentParser();
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data');
  FLAGS = parser.parse_args();
  tf.app.run();
    
    



