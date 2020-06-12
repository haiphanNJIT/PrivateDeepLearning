import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.attacks_tf import fgm, fgsm

from build_utils import *


'''========================================'''
# These are model builders functions to build desiredsmodel
# Modify these to change model structure
# these functions take input tensor, and return logits tensor
# all necessary parameters should be contained in param_dict
'''========================================''' 

def mnist_mlp(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    # define weights
    fc_size = param_dict['fc_size']
    noise_amount = param_dict['noise_amount']
    keep_prob = param_dict['keep_prob']
    n = len(fc_size)
    w, b = [None for i in range(n)], [None for i in range(n)]
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        for i in range(1, n):
            with tf.variable_scope('layer_' + str(i)) as layer_scope:
                w[i] = weight_variable('weight', shape=[fc_size[i - 1], fc_size[i]],
                        scope=layer_scope, stddev=0.04, trainable=True)
                b[i] = bias_variable('bias', shape=[fc_size[i]],
                        scope=layer_scope, value=0.1, trainable=True)
    # define ops
    z, h = [None for i in range(n)], [None for i in range(n)]
    for i in range(n):
        if i == 0:
            # input layer
            z[i] = x_input
            if noise:
                input_noise = tf.random_normal(shape = tf.shape(z[i]),
                    mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
                #input_noise = tf.clip_by_value(input_noise, param_dict["input_low"], param_dict["input_high"])
                z[i] += input_noise
                z[i] = tf.clip_by_value(z[i], param_dict["input_low"], param_dict["input_high"])
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            if noise:
                z[i] += tf.random_normal(shape = tf.shape(z[i]),
                    mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
            if is_training:
                h[i] = tf.nn.dropout(tf.nn.relu(z[i]), keep_prob[i-1])
            else:
                h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            if noise:
                z[i] += tf.random_normal(shape = tf.shape(z[i]),
                    mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
            h[i] = z[i]
    logits = tf.identity(h[n - 1], name=name)
    return logits

def mnist_conv_A(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = [tf.constant(i) for i in param_dict['keep_prob']]
    noise_amount = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [-1,28,28,1])
        # add noise to input
        if noise:
            print(x_input)
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            print(input_noise)
            x_input = tf.clip_by_value(x_input + input_noise, param_dict["input_low"], param_dict["input_high"])
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv2))
            # dropout 1
            if is_training:
                h_conv2 = tf.nn.dropout(h_conv2, keep_prob=keep_prob[0])
            flat_shape = h_conv2.get_shape().as_list()
            h_flat = tf.reshape(h_conv2, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i]))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/fc_size[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    # dropout 2
                    if is_training:
                        h_fc_list[i] = tf.nn.dropout(h_fc_list[i], keep_prob=keep_prob[1])
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits

def mnist_conv_B(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = [tf.constant(i) for i in param_dict['keep_prob']]
    noise_amount = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [-1,28,28,1])
        # add noise to input
        if noise:
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            x_input = tf.clip_by_value(x_input + input_noise, param_dict["input_low"], param_dict["input_high"])
        # dropout 1
        if is_training:
            x_input = tf.nn.dropout(x_input, keep_prob=keep_prob[0])
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 2, 2, 1], padding='SAME'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 2, 2, 1], padding='VALID'), b_conv2))

        with tf.variable_scope('conv_3') as conv2_scope:
            w_conv3 = weight_variable('weight',
                shape=[conv_width[2], conv_width[2], conv_size[1], conv_size[2]],
                scope=conv2_scope, stddev=5e-2)
            b_conv3 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv2, w_conv3,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv3))
            # dropout 2
            if is_training:
                h_conv3 = tf.nn.dropout(h_conv3, keep_prob=keep_prob[1])
            flat_shape = h_conv3.get_shape().as_list()
            h_flat = tf.reshape(h_conv3, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i])))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/fc_size[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits

def mnist_conv_C(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = [tf.constant(i) for i in param_dict['keep_prob']]
    noise_amount = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [-1,28,28,1])
        # add noise to input
        if noise:
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            x_input = tf.clip_by_value(x_input + input_noise, param_dict["input_low"], param_dict["input_high"])
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv2))
            # dropout 1
            if is_training:
                h_conv2 = tf.nn.dropout(h_conv2, keep_prob=keep_prob[0])
            flat_shape = h_conv2.get_shape().as_list()
            h_flat = tf.reshape(h_conv2, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i]))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/fc_size[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    # dropout 2
                    if is_training:
                        h_fc_list[i] = tf.nn.dropout(h_fc_list[i], keep_prob=keep_prob[1])
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits

'''================== cifar10 ================='''
# All the cifar model builders here are placeholders

# def cifar_mlp(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
#     # define weights
#     fc_size = param_dict['fc_size']
#     noise_amount = param_dict['noise_amount']
#     keep_prob = param_dict['keep_prob']
#     n = len(fc_size)
#     w, b = [None for i in range(n)], [None for i in range(n)]
#     with tf.variable_scope(scope, reuse=reuse) as model_scope:
#         for i in range(1, n):
#             with tf.variable_scope('layer_' + str(i)) as layer_scope:
#                 w[i] = weight_variable('weight', shape=[fc_size[i - 1], fc_size[i]],
#                         scope=layer_scope, stddev=0.04, trainable=True)
#                 b[i] = bias_variable('bias', shape=[fc_size[i]],
#                         scope=layer_scope, value=0.1, trainable=True)
#     # define ops
#     z, h = [None for i in range(n)], [None for i in range(n)]
#     for i in range(n):
#         if i == 0:
#             # input layer
#             z[i] = x_input
#             if noise:
#                 input_noise = tf.random_normal(shape = tf.shape(z[i]),
#                     mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
#                 z[i] += input_noise
#                 z[i] = tf.clip_by_value(z[i], 0, 1)
#             h[i] = z[i]
#         if i > 0 and i < n - 1:
#             z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
#             if noise:
#                 z[i] += tf.random_normal(shape = tf.shape(z[i]),
#                     mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
#             if is_training:
#                 h[i] = tf.nn.dropout(tf.nn.relu(z[i]), keep_prob[i-1])
#             else:
#                 h[i] = tf.nn.relu(z[i])
#         if i == n - 1:
#             z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
#             if noise:
#                 z[i] += tf.random_normal(shape = tf.shape(z[i]),
#                     mean=0.0, stddev=noise_amount[i], dtype=tf.float32)
#             h[i] = z[i]
#     logits = tf.identity(h[n - 1], name=name)
#     return logits

def cifar_conv_A(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = param_dict['keep_prob']
    keep_prob = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [-1,28,28,1])
        # add noise to input
        if noise:
            print(x_input)
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            print(input_noise)
            x_input = tf.clip_by_value(x_input + input_noise, 0, 1)
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv2))
            # dropout 1
            if is_training:
                h_conv2 = tf.nn.dropout(h_conv2, keep_prob=keep_prob[0])
            flat_shape = h_conv2.get_shape().as_list()
            h_flat = tf.reshape(h_conv2, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i]))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/FC_LAYERS[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    # dropout 2
                    if is_training:
                        h_fc_list[i] = tf.nn.dropout(h_fc_list[i], keep_prob=keep_prob[1])
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits

def cifar_conv_B(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = param_dict['keep_prob']
    keep_prob = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [None,28,28,1])
        # add noise to input
        if noise:
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            x_input = tf.clip_by_value(x_input + input_noise, 0, 1)
        # dropout 1
        if is_training:
            x_input = tf.nn.dropout(x_input, keep_prob=keep_prob[0])
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 2, 2, 1], padding='SAME'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 2, 2, 1], padding='VALID'), b_conv2))

        with tf.variable_scope('conv_3') as conv2_scope:
            w_conv3 = weight_variable('weight',
                shape=[conv_width[2], conv_width[2], conv_size[1], conv_size[2]],
                scope=conv2_scope, stddev=5e-2)
            b_conv3 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv2, w_conv3,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv3))
            # dropout 2
            if is_training:
                h_conv3 = tf.nn.dropout(h_conv3, keep_prob=keep_prob[1])
            flat_shape = h_conv3.get_shape().as_list()
            h_flat = tf.reshape(h_conv3, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i])))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/FC_LAYERS[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits

def cifar_conv_C(x_input, param_dict, scope, name, reuse=False, noise=False, is_training=True):
    conv_width = param_dict['conv_width']
    conv_channel = param_dict['conv_channel']
    conv_size = param_dict['conv_size']
    fc_size = param_dict['fc_size']
    keep_prob = param_dict['keep_prob']
    keep_prob = param_dict['noise_amount']
    # define conv layers
    with tf.variable_scope(scope, reuse=reuse) as model_scope:
        # reshape the [None, 784] input to [None, 28,28,1]
        x_input = tf.reshape(x_input, [None,28,28,1])
        # add noise to input
        if noise:
            input_noise = tf.random_normal(shape = tf.shape(x_input),
                mean=0.0, stddev=noise_amount[0], dtype=tf.float32)
            x_input = tf.clip_by_value(x_input + input_noise, 0, 1)
        # define conv layers
        with tf.variable_scope('conv_1') as conv1_scope:
            w_conv1 = weight_variable('weight',
                shape=[conv_width[0], conv_width[0], conv_channel, conv_size[0]],
                scope=conv1_scope, stddev=5e-2)
            b_conv1 = bias_variable('bias', [conv_size[0]],
                scope=conv1_scope, value=0.1, trainable=True)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input, w_conv1,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv1))

        with tf.variable_scope('conv_2') as conv2_scope:
            w_conv2 = weight_variable('weight',
                shape=[conv_width[1], conv_width[1], conv_size[0], conv_size[1]],
                scope=conv2_scope, stddev=5e-2)
            b_conv2 = bias_variable('bias', [conv_size[1]],
                scope=conv2_scope, value=0.1, trainable=True)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_conv1, w_conv2,
                strides=[1, 1, 1, 1], padding='VALID'), b_conv2))
            # dropout 1
            if is_training:
                h_conv2 = tf.nn.dropout(h_conv2, keep_prob=keep_prob[0])
            flat_shape = h_conv2.get_shape().as_list()
            h_flat = tf.reshape(h_conv2, [-1, flat_shape[1] * flat_shape[2] * flat_shape[3]]) # flat the input

        # define fc layers
        fc_size[0] = h_flat.get_shape().as_list()[1]
        num_fc_layers = len(fc_size) - 1
        w_fc_list = [None for i in range(num_fc_layers)]
        b_fc_list = [None for i in range(num_fc_layers)]
        h_fc_list = [None for i in range(num_fc_layers + 1)]
        h_fc_list[0] = h_flat
        for i in range(num_fc_layers):# 0 1 2
            with tf.variable_scope('fc_' + str(i+1)) as layer_scope:
                if i < num_fc_layers - 1:
                    # front layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=0.04)
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    h_fc_list[i+1] = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i]))
                else:
                    # output layers
                    w_fc_list[i] = weight_variable('weight',
                        shape=[fc_size[i], fc_size[i+1]],
                        scope=layer_scope, stddev=1.0/FC_LAYERS[-2])
                    b_fc_list[i] = bias_variable('bias', [fc_size[i+1]],
                        scope=layer_scope, value=0.1, trainable=True)
                    # dropout 2
                    if is_training:
                        h_fc_list[i] = tf.nn.dropout(h_fc_list[i], keep_prob=keep_prob[1])
                    h_pre_noise = tf.nn.bias_add(tf.matmul(h_fc_list[i], w_fc_list[i]), b_fc_list[i])
                    if noise:
                        output_noise = tf.random_normal(shape=tf.shape(h_pre_noise),
                            mean=0.0, stddev=noise_amount[1], dtype=tf.float32)
                        h_pre_noise += output_noise
                    h_fc_list[i+1] = h_pre_noise
    logits = tf.identity(h_fc_list[-1], name=name)
    return logits
