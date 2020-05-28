# -*- coding: utf-8 -*-
import numpy as np;
import tensorflow as tf;
from tensorflow.python.framework import ops;
from tensorflow.examples.tutorials.mnist import input_data;
import argparse;
import numpy as np
import math
import random
import scipy.integrate as integrate
import scipy.stats
import mpmath as mp
from gaussian_moments import *;
from tensorflow.python.platform import flags
from datetime import datetime
import time
from tensorflow.python.training import optimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import accountant, utils

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

def idle():
    return

# compute sigma using strong composition theory given epsilon
def compute_sigma(epsilon, delta):
    return 1/epsilon * np.sqrt(np.log(2/math.pi/np.square(delta))+2*epsilon)

# compute sigma using moment accountant given epsilon
def comp_sigma(q,T,delta,epsilon):
    c_2 = 4 * 1.26 / (0.01 * np.sqrt(10000 * np.log(100000))) # c_2 = 1.485
    return c_2 * q * np.sqrt(T * np.log(1 / delta)) / epsilon

# compute epsilon using abadi's code given sigma
def comp_eps(lmbda,q,sigma,T,delta):
    lmbds = range(1, lmbda+1)
    log_moments = []
    for lmbd in lmbds:
        log_moment = compute_log_moment(q, sigma, T, lmbd)
        log_moments.append((lmbd, log_moment))
    
    eps, delta = get_privacy_spent(log_moments, target_delta=delta)
    return eps

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

FLAGS = None;

#target_eps = [0.125,0.25,0.5,1,2,4,8]
target_eps = [0.5];

def main(_):
  clip_bound = 0.01 # 'the clip bound of the gradients'
  clip_bound_2 = 1/1.5 #'the clip bound for r_kM'

  small_num = 1e-5 # 'a small number'
  large_num = 1e5 # a large number'
  num_images = 60000 # 'number of images N'

  batch_size = 600 # 'batch_size L'
  sample_rate = 600/60000 # 'sample rate q = L / N'
  num_steps = 160000 # 'number of steps T = E * N / L = E / q'
  num_epoch = 24 # 'number of epoches E'

  sigma = 5 # 'sigma'
  delta = 1e-5 # 'delta'

  lambd = 1e3 # 'exponential distribution parameter'

  iterative_clip_step = 2 # 'iterative_clip_step'

  clip = 1 # 'whether to clip the gradient'
  noise = 0 # 'whether to add noise'
  redistribute = 0 # 'whether to redistribute the noise'

  D = 60000;
  
  '''from tensorflow.examples.tutorials.mnist import input_data;
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True);'''
  
  sess = tf.InteractiveSession();
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784]);
  x_image = tf.reshape(x, [-1,28,28,1]);
  W_conv1 = weight_variable([5, 5, 1, 32]);
  b_conv1 = bias_variable([32]);
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1);
  h_pool1 = max_pool_2x2(h_conv1);

  W_conv2 = weight_variable([5, 5, 32, 64]);
  b_conv2 = bias_variable([64]);
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
  h_pool2 = max_pool_2x2(h_conv2);
  
  W_fc1 = weight_variable([7 * 7 * 64, 25]);
  b_fc1 = bias_variable([25]);
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]);
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1);
  
  keep_prob = tf.placeholder(tf.float32);
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
  
  W_fc2 = weight_variable([25, 10]);
  b_fc2 = bias_variable([10]);
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10]);

  d = 25*10 + 25*7*7*64 + 5*5*32*64 + 5*5*32; # number of parameters
  M = d
  
  priv_accountant = accountant.GaussianMomentsAccountant(D)
  privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], sigma, batch_size)
  
  #sess.run(tf.initialize_all_variables())
  sess.run(tf.global_variables_initializer())
        
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv));
  #train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy);
  #train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

  opt = GradientDescentOptimizer(learning_rate=1e-2)
  
  #compute gradient
  gw_W1 = tf.gradients(cross_entropy,W_conv1)[0] # gradient of W1
  gb1 = tf.gradients(cross_entropy,b_conv1)[0] # gradient of b1

  gw_W2 = tf.gradients(cross_entropy,W_conv2)[0] # gradient of W2
  gb2 = tf.gradients(cross_entropy,b_conv2)[0] # gradient of b2

  gw_Wf1 = tf.gradients(cross_entropy,W_fc1)[0] # gradient of W_fc1
  gbf1 = tf.gradients(cross_entropy,b_fc1)[0] # gradient of b_fc1
        
  gw_Wf2 = tf.gradients(cross_entropy,W_fc2)[0] # gradient of W_fc2
  gbf2 = tf.gradients(cross_entropy,b_fc2)[0] # gradient of b_fc2

  #clip gradient
  gw_W1 = tf.clip_by_norm(gw_W1,clip_bound)
  gw_W2 = tf.clip_by_norm(gw_W2,clip_bound)
  gw_Wf1 = tf.clip_by_norm(gw_Wf1,clip_bound)
  gw_Wf2 = tf.clip_by_norm(gw_Wf2,clip_bound)
  
  #sigma = FLAGS.sigma # when comp_eps(lmbda,q,sigma,T,delta)==epsilon

  #sensitivity = 2 * FLAGS.clip_bound #adjacency matrix with one tuple different
  sensitivity = clip_bound #adjacency matrix with one more tuple
  
  gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gw_Wf1 += tf.random_normal(shape=tf.shape(gw_Wf1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gbf1 += tf.random_normal(shape=tf.shape(gbf1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gw_Wf2 += tf.random_normal(shape=tf.shape(gw_Wf2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
  gbf2 += tf.random_normal(shape=tf.shape(gbf2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)

  train_step = opt.apply_gradients([(gw_W1,W_conv1),(gb1,b_conv1),(gw_W2,W_conv2),(gb2,b_conv2),(gw_Wf1,W_fc1),(gbf1,b_fc1),(gw_Wf2,W_fc2),(gbf2,b_fc2)]);

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1));
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

  start_time = time.time();
  for i in range(num_steps):
    batch = mnist.train.next_batch(batch_size);
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5});

    if i%100 == 0:
      #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0});
      #print("step \t %d \t training accuracy \t %g"%(i, train_accuracy));
      print("step \t %d \t test accuracy \t %g"%(i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})));
      #epsilon = comp_eps(32, sample_rate, sigma, i, delta)
      #print("epsilon: {}".format(epsilon))
    sess.run([privacy_accum_op])
    spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
    #print(i, spent_eps_deltas)
    _break = False;
    for _eps, _delta in spent_eps_deltas:
      if _delta >= delta:
        _break = True;
        break;
    if _break == True:
        break;
  duration = time.time() - start_time;
  print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}));
  print(float(duration));
  ###

if __name__ == '__main__':
  if tf.gfile.Exists('/tmp/mnist_logs'):
    tf.gfile.DeleteRecursively('/tmp/mnist_logs');
  tf.gfile.MakeDirs('/tmp/mnist_logs');
  
  parser = argparse.ArgumentParser();
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data');
  FLAGS = parser.parse_args();
  tf.app.run();
    
    



