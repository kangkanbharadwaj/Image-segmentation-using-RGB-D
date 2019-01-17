import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

DEFAULT_PADDING = 'SAME'

def make_var(name, shape, trainable):
	'''Creates a new TensorFlow variable.'''
	return tf.get_variable(name, shape, trainable=trainable)


def validate_padding(padding):
	'''Verifies that the padding is one of the supported ones.'''
	assert padding in ('SAME', 'VALID')


def atrous_conv(input, k_h, k_w, c_o, dilation, name, relu=True, padding=DEFAULT_PADDING, trainable=True, group=1, biased=True):
	# Verify that the padding is acceptable
	validate_padding(padding)
	# Get the number of channels in the input
	c_i = input.get_shape()[-1]
	# Verify that the grouping parameter is valid
	assert c_i % group == 0
	assert c_o % group == 0
	# Convolution for a given input and kernel
	convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
	with tf.variable_scope(name) as scope:
		kernel = make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=trainable)
		if group == 1:
			# This is the common-case. Convolve the input without any further complications.
		    output = convolve(input, kernel)
		else:
			# Split the input into groups and then convolve each of them independently
		    input_groups = tf.split(input, group, 3)
		    kernel_groups = tf.split(kernel, group, 3)
		    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
		    # Concatenate the groups
		    output = tf.concat(output_groups, 3)
		# Add the biases
		if biased:
			name2 = name + 'biases'
			biases = make_var(name2, [c_o], trainable)
			output = tf.nn.bias_add(output, biases)
		if relu:
			# ReLU non-linearity
		    output = tf.nn.elu(output, name=scope.name)
		return output