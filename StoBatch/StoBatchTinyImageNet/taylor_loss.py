########################################################################
# Author: NhaiHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow
########################################################################

'''
Taylor expansion of loss function, derived from Tensorflow implementation
'''

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import math
import tensorflow as tf

def TaylorExp(logits, labels, adv_logits, b_labels, L, alpha, perturbW):
    """You can also add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
        of shape [batch_size]
    
        Returns:
        Loss tensor of type float.
        """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.float32)
    b_labels = tf.cast(b_labels, tf.float32)
    
    # Differentially private sparse cross entropy error based on Taylor Expansion
    zeros = array_ops.zeros_like(adv_logits, dtype=adv_logits.dtype)
    cond = (adv_logits >= zeros)
    relu_logits = array_ops.where(cond, adv_logits, zeros)
    neg_abs_logits = array_ops.where(cond, -adv_logits, adv_logits)
    Taylor_adv = math_ops.add(relu_logits - adv_logits * b_labels, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
    
    ### Taylor for benign x
    zeros2 = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond2 = (logits >= zeros2)
    relu_logits_benign = array_ops.where(cond2, logits, zeros2)
    neg_abs_logits_benign = array_ops.where(cond2, -logits, logits)
    Taylor_benign = math_ops.add(relu_logits_benign - logits * labels, math.log(2.0) + 0.5*neg_abs_logits_benign + 1.0/8.0*neg_abs_logits_benign**2)
    
    zeros1 = array_ops.zeros_like(perturbW, dtype=perturbW.dtype)
    cond1 = (perturbW >= zeros1)
    perturbW = array_ops.where(cond1, perturbW, -perturbW)
    
    ### Adversarial training loss
    #adv_loss = (1/(L + alpha*L))*(Taylor_benign + alpha * Taylor_adv)
    # adv_loss = (1/(1 + alpha))*(Taylor_benign + alpha * Taylor_adv)
    adv_loss = (Taylor_benign + alpha * Taylor_adv)

    cross_entropy_mean = tf.reduce_mean(adv_loss, name='cross_entropy') + tf.reduce_mean(perturbW, name = 'perturbW')
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def TaylorExp_no_noise(logits, labels, adv_logits, b_labels, L, alpha):
    """
      You can also add L2Loss to all the trainable variables.
      Add summary for "Loss" and "Loss/avg".
      Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
      of shape [batch_size]
  
      Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.float32)
    b_labels = tf.cast(b_labels, tf.float32)
    
    # Differentially private sparse cross entropy error based on Taylor Expansion
    zeros = array_ops.zeros_like(adv_logits, dtype=adv_logits.dtype)
    cond = (adv_logits >= zeros)
    relu_logits = array_ops.where(cond, adv_logits, zeros)
    neg_abs_logits = array_ops.where(cond, -adv_logits, adv_logits)
    Taylor_adv = math_ops.add(relu_logits - adv_logits * b_labels, math.log(2.0) + 0.5*neg_abs_logits + 1.0/8.0*neg_abs_logits**2)
    # Taylor_adv = tf.abs(math_ops.add(relu_logits - adv_logits * b_labels, math.log(2.0) + 0.5*neg_abs_logits))
    
    ### Taylor for benign x
    zeros2 = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond2 = (logits >= zeros2)
    relu_logits_benign = array_ops.where(cond2, logits, zeros2)
    neg_abs_logits_benign = array_ops.where(cond2, -logits, logits)
    Taylor_benign = math_ops.add(relu_logits_benign - logits * labels, math.log(2.0) + 0.5*neg_abs_logits_benign + 1.0/8.0*neg_abs_logits_benign**2)
    # Taylor_benign = tf.abs(math_ops.add(relu_logits_benign - logits * labels, math.log(2.0) + 0.5*neg_abs_logits_benign))
    
    ### Adversarial training loss
    # adv_loss = (1/(L + alpha*L))*(Taylor_benign + alpha * Taylor_adv)
    # adv_loss = (1/(1 + alpha))*(Taylor_benign + alpha * Taylor_adv)
    adv_loss = (Taylor_benign + alpha * Taylor_adv)

    cross_entropy_mean = tf.reduce_mean(adv_loss, name='cross_entropy')# + tf.reduce_sum(perturbW, name = 'perturbW');
    
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

