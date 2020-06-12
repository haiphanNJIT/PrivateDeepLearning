import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.attacks_tf import fgm, fgsm

def weight_variable(name, shape, scope, stddev=0.1, trainable=True):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(mean=0.0, stddev=stddev)
        variable = tf.get_variable(name, shape, initializer=init, trainable=trainable)
    return variable

def bias_variable(name, shape, scope, value, trainable=True):
    with tf.variable_scope(scope):
        init = tf.constant_initializer(value, dtype=tf.float32)
        variable = tf.get_variable(name, shape, initializer=init, trainable=trainable)
    return variable

def variable_with_weight_decay(name, shape, scope, stddev=0.1, trainable=True, wd=None):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(mean=0.0, stddev=stddev)
        variable = tf.get_variable(name, shape, initializer=init, trainable=trainable)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)
    return variable

def conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def softmax_loss(labels, logits, name):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name=name)

def accuracy_tensor(labels, logits, name):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)),tf.float32), name=name)

def accuracy(labels, logits):
    return np.mean(np.equal(np.argmax(logits, axis=1), tf.argmax(labels, axis=1)).astype(float))

def op_inputs_10(data_dir, batch_size,eval_data):
    '''
    Call cifar10_input.inputs to create a input op that read and preprocess the data
    '''
    return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                batch_size=batch_size)

def op_distorted_inputs_10(data_dir, batch_size):
    '''
    Call cifar10_input.inputs to create a input op that read and preprocess the data
    with random distortions
    '''
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                          batch_size=batch_size)

def batch_norm(input_t, bias, scope, is_test, step, decay=0.99, reuse=True):
    # Perform a batch normalization after a conv layer or a fc layer
    # bias: bias term
    # is_teat: bool tensor
    # step: tensor, current step count
    # epsilon: the variance epsilon - a small float number to avoid dividing by 0
    with tf.variable_scope(scope, reuse=reuse) as bn_scope:
        shape = input_t.get_shape().as_list()
        #beta = tf.get_variable('beta', shape[-1], initializer=tf.constant_initializer(0.1), trainable=True)
        exp_moving_avg = tf.train.ExponentialMovingAverage(decay, step)
        m, v = tf.nn.moments(input_t, range(len(shape)-1))
        update_moving_averages = exp_moving_avg.apply([m, v])
        mean = tf.cond(is_test, lambda: exp_moving_avg.average(m), lambda: m)
        var = tf.cond(is_test, lambda: exp_moving_avg.average(v), lambda: v)
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(input_t, mean, var,
                offset=bias, scale=None, variance_epsilon=1e-5)
    return output, update_moving_averages

def batch_eval(sess, acc_op, x_input_t, labels_t, test_input, test_labels, batch_size):
    '''
    Do evaluation in batch
    '''
    n_samples = test_input.shape[0]
    acc_list = []
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        batch_test_input = test_input[start:end]
        batch_test_labels = test_labels[start:end]
        feed_dict = {x_input_t:batch_test_input, labels_t:batch_test_labels}
        acc_list.append(sess.run(acc_op, feed_dict=feed_dict))
    return np.mean(acc_list)

def batch_adv(sess, adv_op, x_input_t, labels_t, test_input, test_labels, batch_size):
    '''
    Generate adv samples in batch
    '''
    n_samples = test_input.shape[0]
    adv_list = []
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        batch_test_input = test_input[start:end]
        batch_test_labels = test_labels[start:end]
        feed_dict = {x_input_t:batch_test_input, labels_t:batch_test_labels}
        adv_list.append(sess.run(adv_op, feed_dict=feed_dict))
    return np.concatenate(adv_list)
