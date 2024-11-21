"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class ZicoSVHN(object):
    def __init__(self, images):
        self.x_image = images  # Input images of shape [batch_size, 32, 32, 3]

        # Adjusted weight and bias shapes for SVHN
        W0 = tf.get_variable('W0', dtype=tf.float32, shape=(4, 4, 3, 16))
        B0 = tf.get_variable('B0', dtype=tf.float32, shape=(16,))
        W2 = tf.get_variable('W2', dtype=tf.float32, shape=(4, 4, 16, 32))
        B2 = tf.get_variable('B2', dtype=tf.float32, shape=(32,))
        W5 = tf.get_variable('W5', dtype=tf.float32, shape=(2048, 100))
        B5 = tf.get_variable('B5', dtype=tf.float32, shape=(100,))
        W7 = tf.get_variable('W7', dtype=tf.float32, shape=(100, 10))
        B7 = tf.get_variable('B7', dtype=tf.float32, shape=(10,))

        # First convolutional layer
        y = tf.pad(self.x_image, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # After padding: [batch_size, 34, 34, 3]
        y = tf.nn.conv2d(y, W0, strides=[1, 2, 2, 1], padding='VALID')
        # After convolution: [batch_size, 16, 16, 16]
        y = tf.nn.bias_add(y, B0)
        y = tf.nn.relu(y)
        
        # Second convolutional layer
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # After padding: [batch_size, 18, 18, 16]
        y = tf.nn.conv2d(y, W2, strides=[1, 2, 2, 1], padding="VALID")
        # After convolution: [batch_size, 8, 8, 32]
        y = tf.nn.bias_add(y, B2)
        y = tf.nn.relu(y)
        
        # Reshape and fully connected layers
        y = tf.transpose(y, [0, 3, 1, 2])  # Shape: [batch_size, 32, 8, 8]
        y = tf.reshape(y, [tf.shape(y)[0], -1])  # Flatten: [batch_size, 2048]
        y = tf.matmul(y, W5) + B5  # Fully connected layer: [batch_size, 100]
        y = tf.nn.relu(y)
        y = tf.matmul(y, W7) + B7  # Output logits: [batch_size, 10]

        self.logits = y
