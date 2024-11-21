"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class ZicoCelebA(object):
    def __init__(self, images):
        self.x_image = images  # Input placeholder of shape [batch_size, 64, 64, 3]
        # First convolutional layer variables
        W0 = tf.get_variable('W0', dtype=tf.float32, shape=(4, 4, 3, 16))
        B0 = tf.get_variable('B0', dtype=tf.float32, shape=(16,))
        # Second convolutional layer variables
        W2 = tf.get_variable('W2', dtype=tf.float32, shape=(4, 4, 16, 32))
        B2 = tf.get_variable('B2', dtype=tf.float32, shape=(32,))
        # Third convolutional layer variables
        W3 = tf.get_variable('W3', dtype=tf.float32, shape=(4, 4, 32, 64))
        B3 = tf.get_variable('B3', dtype=tf.float32, shape=(64,))
        # Fully connected layer variables
        W5 = tf.get_variable('W5', dtype=tf.float32, shape=(4096, 100))
        B5 = tf.get_variable('B5', dtype=tf.float32, shape=(100,))
        W7 = tf.get_variable('W7', dtype=tf.float32, shape=(100, 2))
        B7 = tf.get_variable('B7', dtype=tf.float32, shape=(2,))

        # First convolutional layer
        y = tf.pad(self.x_image, [[0, 0], [1, 1], [1, 1], [0, 0]])  # [batch_size, 66, 66, 3]
        y = tf.nn.conv2d(y, W0, strides=[1, 2, 2, 1], padding='VALID')  # [batch_size, 32, 32, 16]
        y = tf.nn.bias_add(y, B0)
        y = tf.nn.relu(y)

        # Second convolutional layer
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]])  # [batch_size, 34, 34, 16]
        y = tf.nn.conv2d(y, W2, strides=[1, 2, 2, 1], padding='VALID')  # [batch_size, 16, 16, 32]
        y = tf.nn.bias_add(y, B2)
        y = tf.nn.relu(y)

        # Third convolutional layer
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]])  # [batch_size, 18, 18, 32]
        y = tf.nn.conv2d(y, W3, strides=[1, 2, 2, 1], padding='VALID')  # [batch_size, 8, 8, 64]
        y = tf.nn.bias_add(y, B3)
        y = tf.nn.relu(y)

        # Flatten the output
        y = tf.reshape(y, [tf.shape(y)[0], -1])  # [batch_size, 8*8*64 = 4096]

        # Fully connected layer
        y = tf.matmul(y, W5) + B5
        y = tf.nn.relu(y)

        # Output layer
        y = tf.matmul(y, W7) + B7  # Output logits [batch_size, 1]

        self.logits = y  # Output logits
