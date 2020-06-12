########################################################################
# Author: NhaiHai Phan, Han Hu
# License: Apache 2.0
# source code snippets from: Tensorflow
########################################################################

'''
Loss function of SecureSGD
'''

import tensorflow as tf

def lossDPSGD(logits, labels):
    """Add L2Loss to all the trainable variables.
        
    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
    of shape [batch_size]
    
    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.argmax(tf.cast(labels, tf.int64), 1)
    print(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
        
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')