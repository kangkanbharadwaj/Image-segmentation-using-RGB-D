from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
import math



def TransitionUp_elu_Grouped(input, filters, groups, name, Final=False):
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  #output_shape = [1, 75, 75, 128]
  #upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=2,  scope=name)
  kernel_size = 3
  stride = 2 
  upconv = groupDeconvolution(current, kernel_size, kernel_size, filters, stride, stride, name, padding='SAME', groups=groups)
  #groupConvolution(RELU, kernel_size, kernel_size, out_features, stride, stride, name, padding='SAME', groups=groups)
  upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv

def groupDeconvolution(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  #Round to group
  new_size_multiplier = math.ceil(input_channels/groups)
  #print("New size mulltiplier")
  #print(new_size_multiplier)
  new_size = groups * new_size_multiplier
  #print("New size")
  #print(new_size)
  if(new_size!=input_channels):
    name2=name+'transform_layer'
    x = slim.conv2d(x, new_size, [1, 1], stride=1, scope=name2)
    input_channels = int(x.get_shape()[-1])


  output_shape = x.get_shape().as_list()
  output_shape[1] *= 2
  output_shape[2] *= 2
  output_shape[3] = num_filters//groups

  #output_shape = tf.convert_to_tensor(output_shape)
  #output_shape = tf.convert_to_tensor(output_shape)
  #output_shape = tf.stack([output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
  #Create lambda function for the convolution
  #output_shape = tf.TensorShape([None, output_shape[1], output_shape[1], output_shape[3]])
  #output_shape = tf.convert_to_tensor(output_shape)
  # print('output_shape')
  # print(output_shape)

  deconv_shape = tf.stack([1, output_shape[1], output_shape[2], output_shape[3]])
  # print('deconv_shape')
  # print(deconv_shape)

  
  #convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1], padding = padding)
  convolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape=deconv_shape,  strides=[1, stride_y, stride_x, 1], padding=padding)
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    #print("input channels")
    #print(input_channels)
    weights = tf.get_variable('weights',
                              shape = [filter_height, filter_width,
                              num_filters, input_channels//groups])
    # print('weights')
    # print(weights)
    biases = tf.get_variable('biases', shape = [num_filters])

    # out_shapeV = tf.get_variable('out_shape',
    #                           shape = [int(output_shape[0]), int(output_shape[1]), int(output_shape[2]), int(output_shape[3])], initializer=tf.constant([int(output_shape[0]), int(output_shape[1]),int(output_shape[2]), int(output_shape[3])]))
    # print('out_shapeV')
    # print(out_shapeV)

    if groups == 1:
      conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      # print('input_groups')
      # print(input_groups)
      weight_groups = tf.split(axis = 2, num_or_size_splits=groups, value=weights)
      # print('weight_groups')
      # print(weight_groups)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      # print('output_groups')
      # print(output_groups)
      # Concat the convolved output together again
      conv = tf.concat(output_groups, 3)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    #relu = tf.nn.relu(bias, name = scope.name)

    return bias


def groupConvolution(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  #Round to group
  new_size_multiplier = math.ceil(input_channels/groups)
  #print("New size mulltiplier")
  #print(new_size_multiplier)
  new_size = groups * new_size_multiplier
  #print("New size")
  #print(new_size)
  if(new_size!=input_channels):
    name2=name+'transform_layer'
    x = slim.conv2d(x, new_size, [1, 1], stride=1, scope=name2)
    input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1], padding = padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    #print("input channels")
    #print(input_channels)
    weights = tf.get_variable('weights',
                              shape = [filter_height, filter_width,
                              input_channels//groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])


    if groups == 1:
      conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

      # Concat the convolved output together again
      conv = tf.concat(output_groups, 3)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    #relu = tf.nn.relu(bias, name = scope.name)

    return bias


def groupConvolution_slim(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  #convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1], padding = padding)
  convolve = lambda i, k: slim.conv2d(i, k, [filter_height, filter_width], stride=stride_x, scope=name, padding=padding)
  

  # with tf.variable_scope(name) as scope:
  #   # Create tf variables for the weights and biases of the conv layer
  #   weights = tf.get_variable('weights',
  #                             shape = [filter_height, filter_width,
  #                             input_channels/groups, num_filters])
  #   biases = tf.get_variable('biases', shape = [num_filters])


  if groups == 1:
    weights = num_filters
    conv = convolve(x, weights)

  # In the cases of multiple groups, split inputs & weights and
  else:
    # Split input and weights and convolve them separately
    input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
    weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
    print(weight_groups)
    output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

    # Concat the convolved output together again
    conv = tf.concat(output_groups, 3)

    # Add biases
    #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    #relu = tf.nn.relu(bias, name = scope.name)

  return conv


