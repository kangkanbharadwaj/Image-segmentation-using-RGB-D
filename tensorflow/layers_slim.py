from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
from atrous_convolution import * 


def Swish_ActivationFunction(curr):
  curr_out = tf.nn.sigmoid(curr) * curr
  return curr_out

# def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
#   W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
#   conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
#   if with_bias:
#     return conv + bias_variable([ out_features ])
#   return conv

# def max_pool(input, s):
#   return tf.nn.max_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID'
# )


#Transition down module now with mixtured 
# of max and Average pooling
def TransitionDown_v2(input, filters, keep_prob, name):
  current = input
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  name3=name+'BN'
  current = slim.batch_norm(current, scope=name3)
  current = tf.nn.elu(current)
  #current = conv2d(current, in_features, filters, 1)
  current2 = slim.conv2d(current, filters, [1, 1], scope=name)
  current3= tf.nn.dropout(current2, keep_prob)
  name2=name+'pool'
  #print(name2)
  #current = max_pool(current, 2)
  
  #Before
  #probability=0.5
  #mix_max_Avg = Mix_Max_Avg_Pooling(current3, name=name2, mix_portion=probability)
  
  name2=name+'avg_portion'
  avgP = slim.avg_pool2d(current3, [2, 2], scope=name2)
  #current4 = slim.max_pool2d(current3, [2, 2], scope=name2)
  #current4 = slim.max_pool2d(current3, [2, 2], stride=2, scope=name2)
  return avgP


#Mix max and average pooling 
# 01/02/2018
def Mix_Max_Avg_Pooling(current, name, mix_portion):
  name2=name+'max_portion'
  maxP = slim.max_pool2d(current, [2, 2], scope=name2)
  name2=name+'avg_portion'
  avgP = slim.avg_pool2d(current, [2, 2], scope=name2)
  #Now fmix= alpha*max_pool + (1-alpha)avg_pool
  f_mix= tf.add(tf.multiply(maxP,mix_portion), tf.multiply(avgP,(1-mix_portion)))
  return f_mix  



def Conv_BN_ReLU(current, out_features, kernel_size, stride, name):
  if stride > 1:
    CONV = slim.conv2d(current, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')
  else:
    CONV = slim.conv2d(current, out_features, [kernel_size, kernel_size], stride=stride, scope=name)
  #conv_drop = tf.nn.dropout(CONV, dropout)
  BN = slim.batch_norm(CONV,activation_fn=None)
  RELU = tf.nn.relu(BN)
  return RELU



####################################################
### 01/11/17 Gated Backward connections ############
####################################################
def Gate_BackWard_skipConnection(highRSkip, lowRSkip, size, name):
  #each stream is BN, Elu, 3x3 conv
  #low resolution stream is upsampled
  #then both streams are fused using Element-wise product

  #High resolution block 
  name2 = name + 'High_stream_skip'
  #Downsample 2x High skip resolution stream
  #name2 = name + 'Low_stream_up'
  
  current = slim.batch_norm(highRSkip,activation_fn=None)
  current2 = tf.nn.elu(current)
  current3 = slim.conv2d(current2, size, [3, 3], scope=name2)
  name2= name + 'Pool'
  current_Pool_high = slim.max_pool2d(current3, [2, 2], scope=name2)
  #Low_streamUp = slim.conv2d_transpose(Low_stream, size, [3, 3], stride=2,  scope=name2)
  #if name=='GateSkip4':
  #  pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
  #  Low_streamUp = tf.pad(Low_streamUp, pattern)
  

  #Low resolution block 
  name2 = name + 'Low_stream_skip'
  #High_stream = Conv_BN_ReLU(highRSkip, out_features=size, kernel_size=3, stride=1, name=name2)
  current_low = slim.batch_norm(lowRSkip,activation_fn=None)
  current_low = tf.nn.elu(current_low)
  Low_stream = slim.conv2d(current_low, size, [3, 3], scope=name2)

  #Fusion 
  FusedStreams = tf.multiply(current_Pool_high, Low_stream)

  return FusedStreams

####################################################
### 24/10/17 Gating Skip connections ###############
####################################################
def Gate_skip_Connection(highRSkip, lowRSkip, size, name):
  #each stream is 3x3 conv, BN and Relu
  #low resolution stream is upsampled
  #then both streams are fused using Element-wise product

  #Low resolution block 
  name2 = name + 'Low_stream_skip'
  Low_stream = Conv_BN_ReLU(lowRSkip, out_features=size, kernel_size=3, stride=1, name=name2)
  #upsample 2x Low skip resolution stream
  name2 = name + 'Low_stream_up'
  Low_streamUp = slim.conv2d_transpose(Low_stream, size, [3, 3], stride=2,  scope=name2)
  if name=='GateSkip4':
    pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    Low_streamUp = tf.pad(Low_streamUp, pattern)
  

  #High resolution block 
  name2 = name + 'High_stream_skip'
  High_stream = Conv_BN_ReLU(highRSkip, out_features=size, kernel_size=3, stride=1, name=name2)

  #Fusion 
  FusedStreams = tf.multiply(High_stream, Low_streamUp)

  return FusedStreams



#####################################################
## Implementing Global Convolutional Network Block ##
##################################################### 
def GCN_block(current, out_features, kernel_size, stride, name):
  name2 = name + '_l1'
  conv_l1 = slim.conv2d(current, out_features, [kernel_size, 1], stride=stride, scope=name2, padding='SAME')
  name2 = name + '_l2'
  conv_l2 = slim.conv2d(conv_l1, out_features, [1, kernel_size], stride=stride, scope=name2, padding='SAME')

  name2 = name + '_r1'
  conv_r1 = slim.conv2d(current, out_features, [1, kernel_size], stride=stride, scope=name2, padding='SAME')
  name2 = name + '_r2'
  conv_r2 = slim.conv2d(conv_r1, out_features, [kernel_size ,1], stride=stride, scope=name2, padding='SAME')

  Conv_GCN = tf.add(conv_l2, conv_r2)

  return Conv_GCN

#Boundary Refinement Block
def BR_block(current, out_features, kernel_size, stride, name):
  name2 = name + 'CV1_bn'
  bn1 = slim.batch_norm(current,activation_fn=None, scope=name2)
  relu1 = tf.nn.relu(bn1)
  name2 = name + 'CV1'
  conv1 = slim.conv2d(relu1, out_features, [kernel_size, kernel_size], stride=stride, scope=name2, padding='SAME')
  name2 = name + 'CV2_bn'
  bn2 = slim.batch_norm(conv1,activation_fn=None, scope=name2)
  relu2 = tf.nn.relu(bn2)
  name2 = name + 'CV2'
  conv1 = slim.conv2d(relu2, out_features, [kernel_size, kernel_size], stride=stride, scope=name2, padding='SAME')

  conv_BR = tf.add(current , conv1)

  return conv_BR

def BN_ReLU_Conv(current, out_features, kernel_size, stride, name):
  BN = slim.batch_norm(current,activation_fn=None)
  RELU = tf.nn.relu(BN)
  if stride > 1:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')

  else:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name)
  #conv_drop = tf.nn.dropout(CONV, dropout)

  return CONV

def BN_ReLU_Conv_drop(current, out_features, kernel_size, stride, name, dropout):
  BN = slim.batch_norm(current,activation_fn=None)
  RELU = tf.nn.relu(BN)
  if stride > 1:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')

  else:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name)
  conv_drop = tf.nn.dropout(CONV, dropout)

  return conv_drop

def BN_ReLU_Conv_atrous(current, out_features, kernel_size, stride, name, rate, group):
  BN = slim.batch_norm(current,activation_fn=None)
  RELU = tf.nn.relu(BN)
  CONV = atrous_conv(RELU, kernel_size, kernel_size, out_features, rate, name=name, padding='SAME', biased=False, relu=False, trainable=True, group=group)

  return CONV

def Atrous_Spatial_pyramid_pooling(current, out_features, kernel_size, stride, name, rate1, rate2, rate3):
  
  #Batch Normalization
  BN = slim.batch_norm(current,activation_fn=None)
  #Convolutions 
  name2 = name + 'c1x1-SPP'
  conv1x1 = slim.conv2d(BN, out_features, [1, 1], stride=stride, scope=name2, padding='SAME')
  #conv1x1 = atrous_conv(BN, kernel_size, kernel_size, out_features, 1, name=name2, padding='SAME', biased=True, relu=False, trainable=True, group=1)
  #name2 = name + 'c3x3-SPP_rate0'
  #conv_rate0 = atrous_conv(BN, kernel_size, kernel_size, out_features, rate0, name=name2, padding='SAME', biased=False, relu=False, trainable=True, group=1)
  name2 = name + 'c3x3-SPP_rate1'
  conv_rate6 = atrous_conv(BN, kernel_size, kernel_size, out_features, rate1, name=name2, padding='SAME', biased=False, relu=False, trainable=True, group=1)
  name2 = name + 'c3x3-SPP_rate2'
  conv_rate12 = atrous_conv(BN, kernel_size, kernel_size, out_features, rate2, name=name2, padding='SAME', biased=False, relu=False, trainable=True, group=1)
  name2 = name + 'c3x3-SPP_rate3'
  conv_rate18 = atrous_conv(BN, kernel_size, kernel_size, out_features, rate3, name=name2, padding='SAME', biased=False, relu=False, trainable=True, group=1)

  concat_convs = tf.concat((conv1x1, conv_rate6, conv_rate12, conv_rate18), 3)

  return concat_convs

# Better Implementation
def Atrous_Spatial_pyramid_pooling_v2(current, out_features, kernel_size, stride, name, rate1, rate2, rate3):
  name2 = name + 'PoolingASPP'
  current_Pool = slim.max_pool2d(current, [2, 2], scope=name2)
  #Batch Normalization
  #BN = slim.batch_norm(current,activation_fn=None)
  #Convolutions 
  name2 = name + 'c1x1-SPP'
  conv1x1 = slim.conv2d(current_Pool, out_features, [1, 1], stride=stride, scope=name2, padding='SAME')
  name2 = name + 'c1x1-SPPup'
  upconv1x1 = slim.conv2d_transpose(conv1x1, out_features, [1, 1], stride=2,  scope=name2)
  pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
  upconv1x1 = tf.pad(upconv1x1, pattern)
  #conv1x1 = atrous_conv(BN, kernel_size, kernel_size, out_features, 1, name=name2, padding='SAME', biased=True, relu=False, trainable=True, group=1)
  #name2 = name + 'c3x3-SPP_rate0'
  #conv_rate0 = atrous_conv(BN, kernel_size, kernel_size, out_features, rate0, name=name2, padding='SAME', biased=False, relu=False, trainable=True, group=1)
  name2 = name + 'c3x3-SPP_rate1'
  conv_rate6 = atrous_conv(current_Pool, kernel_size, kernel_size, out_features, rate1, name=name2, padding='SAME', biased=True, relu=False, trainable=True, group=1)
  name2 = name + 'rate6-SPPup'
  upconv_rate6 = slim.conv2d_transpose(conv_rate6, out_features, [3, 3], stride=2,  scope=name2)
  pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
  upconv_rate6 = tf.pad(upconv_rate6, pattern)

  name2 = name + 'c3x3-SPP_rate2'
  conv_rate12 = atrous_conv(current_Pool, kernel_size, kernel_size, out_features, rate2, name=name2, padding='SAME', biased=True, relu=False, trainable=True, group=1)
  name2 = name + 'rate12-SPPup'
  upconv_rate12 = slim.conv2d_transpose(conv_rate12, out_features, [3, 3], stride=2,  scope=name2)
  pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
  upconv_rate12 = tf.pad(upconv_rate12, pattern)

  name2 = name + 'c3x3-SPP_rate3'
  conv_rate18 = atrous_conv(current_Pool, kernel_size, kernel_size, out_features, rate3, name=name2, padding='SAME', biased=True, relu=False, trainable=True, group=1)
  name2 = name + 'rate18-SPPup'
  upconv_rate18 = slim.conv2d_transpose(conv_rate18, out_features, [3, 3], stride=2,  scope=name2)
  pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
  upconv_rate18 = tf.pad(upconv_rate18, pattern)

  concat_convs = tf.concat((current, upconv1x1, upconv_rate6, upconv_rate12, upconv_rate18), 3)

  return concat_convs

def DPDB_Block(input, num_1x1_a, num_3x3_b, num_1x1_c, inc, name, _type):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  stride=1

  if isinstance(input, list):
    init = tf.concat((input[0], input[1]), 3)
  else:
    init = input

  if _type is 'create':
    create = True
  if _type is 'dense':
    create = False

  if create:
    name2 = name + 'residual'
    projection_path = BN_ReLU_Conv(init, out_features=num_1x1_c+2*inc, kernel_size=1, stride=stride, name=name2) 
    input_residual_path = projection_path[:,:,:,:num_1x1_c]   
    input_dense_path = projection_path[:,:,:,num_1x1_c:]
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]

  name2 = name + 'c1x1a'
  x = BN_ReLU_Conv(init, out_features=num_1x1_a, kernel_size=1, stride=1, name=name2)   
  name2 = name + 'c3x3b'
  x = BN_ReLU_Conv(x, out_features=num_3x3_b, kernel_size=3, stride=1, name=name2)
  name2 = name + 'c1xca'
  x = BN_ReLU_Conv(x, out_features=num_1x1_c+inc, kernel_size=1, stride=stride, name=name2)  

  output_residual_path = x[:,:,:,:num_1x1_c]
  output_dense_path = x[:,:,:,num_1x1_c:]

  residual_path =  tf.add(input_residual_path, output_residual_path)
  dense_path = tf.concat((input_dense_path, output_dense_path), 3)

  # if has_proj:
  #   concat_convs = tf.concat((c1x1_w, c1x1_c), 3)
  #   return concat_convs
  
  return [residual_path, dense_path]


#DualPathInterBlock(input, layers, growth=inc, layers=k_sec[0],  bw, R, name='conv1', residual_type='proj')
def DualPathInterBlock(input, growth, layers, bw, R, name, residual_type):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  #features = Din_features
  #Residual Convolution
  Conv_residual = DualPathBlock(input, R, R, bw, growth, name, _type=residual_type)

  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = DualPathBlock(Conv_residual, R, R, bw, growth, name2, _type='normal')
    #print(current)
    #print(tmp)
    current = tmp # tf.concat((current, tmp), 3)
    #features += growth
  current = tf.concat((Conv_residual, current), 3) 
  return current


#DualPathInterBlock(input, layers, growth=inc, layers=k_sec[0],  bw, R, name='conv1', residual_type='proj')
def DualPathInterBlock_atrous(input, growth, layers, bw, R, name, residual_type, rate):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  #features = Din_features
  #Residual Convolution
  Conv_residual = DualPathBlock(input, R, R, bw, growth, name, _type=residual_type)

  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = DualPathBlock(Conv_residual, R, R, bw, growth, name2, _type='normal')
    #print(current)
    #print(tmp)
    current = tmp # tf.concat((current, tmp), 3)
    #features += growth
  current = tf.concat((Conv_residual, current), 3) 
  return current


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def Concat_layers(conv1, conv2, nm='test'):
    #Concat values
    if(nm=='upconv2'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)
    if(nm=='upconv3'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)
    if(nm=='upconvV3'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)
        
    fused = tf.concat([conv1, conv2],3)
    return  fused


def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob, name):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current,activation_fn=None)
  current2 = tf.nn.elu(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  current3 = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)
  current4 = tf.nn.dropout(current3, keep_prob)
  
  return current4


def Denseblock(input, layers, Din_features, growth, is_training, keep_prob, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob, name2)
    #print(current)
    #print(tmp)
    current =  tf.concat((current, tmp), 3)
    features += growth
  return current


def batch_activ_conv_Swish(current, in_features, out_features, kernel_size, is_training, keep_prob, name):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current,activation_fn=None)
  current2 = Swish_ActivationFunction(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  current3 = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)
  current4 = tf.nn.dropout(current3, keep_prob)
  
  return current4


def Denseblock_Swish(input, layers, Din_features, growth, is_training, keep_prob, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv_Swish(current, features, growth, 3, is_training, keep_prob, name2)
    #print(current)
    #print(tmp)
    current =  tf.concat((current, tmp), 3)
    features += growth
  return current

def batch_activ_conv_atrous(current, in_features, out_features, kernel_size, is_training, keep_prob, trainable, Dilatation, name):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current,activation_fn=None)
  current2 = tf.nn.elu(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  current3 = atrous_conv(current2, kernel_size, kernel_size, out_features, Dilatation, name=name, padding='SAME', biased=True, relu=True, trainable=True)
  current4 = tf.nn.dropout(current3, keep_prob)
  
  return current4


def Denseblock_atrous(input, layers, Din_features, growth, is_training, keep_prob, trainable, Dilatation, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv_atrous(current, features, growth, 3, is_training, keep_prob, trainable, Dilatation, name2)
    #print(current)
    #print(tmp)
    current = tf.concat((current, tmp), 3)
    features += growth
  return current

#Dense Block for Segmentation Version 1.0 - 26/07/2017
#Produce a number of features Twice the layers*Groth - #growth to atrous and #growth to normal dense stream
# Main Stream is a normal Dense Block but the add a Dense Residual Atrous connection
def DSeg_Block_Wide(input, layers, Din_features, growth, is_training, keep_prob, trainable, Dilatation, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  
  #Dense Residual Atrous Connection
  DenseResidualAtrous_connection = batch_activ_conv_atrous(current, features, growth*layers, 3, is_training, keep_prob, trainable, Dilatation, name2)
  current = tf.concat((current, DenseResidual_connection), 3)
  features += growth*layers

  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob, trainable, name2)
    #print(current)
    #print(tmp)
    current = tf.concat((current, tmp), 3)
    features += growth
  return current

#Dense Block for Segmentation Version 1.0 - 26/07/2017
#Produce a number of features layers*Groth - #NO atrous and #growth to normal dense stream
# Main Stream is a normal Dense Block but the add a Dense Residual connection
def DSeg_Block_v2(input, layers, Din_features, growth, is_training, keep_prob, trainable, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  
  #Dense Residual Connection
  name2 = name + str('ResidualC')
  DenseResidual_connection = batch_activ_conv(current, features, (growth*layers)/2, 3, is_training, keep_prob, name2)
  current = tf.concat((current, DenseResidual_connection), 3)
  features += (growth*layers)/2

  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv(current, features, growth/2, 3, is_training, keep_prob, name2)
    #print(current)
    #print(tmp)
    current = tf.concat((current, tmp), 3)
    features += growth
  return current

def DSeg_Block_v3(input, layers, Din_features, growth, is_training, keep_prob, trainable, Dilatation, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  
  #Dense Residual Atrous Connection
  DenseResidualAtrous_connection = batch_activ_conv_atrous(current, features, growth*layers, 3, is_training, keep_prob, trainable, Dilatation, name2)
  current = tf.concat((current, DenseResidual_connection), 3)
  features += growth*layers

  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv(current, features, growth/2, 3, is_training, keep_prob, trainable, name2)
    tmp2 = batch_activ_conv_atrous(current, features, growth/2, 3, is_training, keep_prob, trainable, Dilatation, name2)
    aux = tf.concat((tmp, tmp2), 3)
    #print(current)
    #print(tmp)
    current = tf.concat((current, aux), 3)
    features += growth
  return current

#Not possible to remove the conv 1x1 to adjust feature size
def TransitionDownOnlyPooling(input, filters, keep_prob, name):
  current = input
  # #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  # name3=name+'BN'
  # current = slim.batch_norm(current, scope=name3)
  # current = tf.nn.elu(current)
  # #current = conv2d(current, in_features, filters, 1)
  current2 = slim.conv2d(current, filters, [1, 1], scope=name)
  # current3= tf.nn.dropout(current2, keep_prob)
  name2=name+'pool'
  #print(name2)
  #current = max_pool(current, 2)
  #Before
  current4 = slim.max_pool2d(current2, [2, 2], scope=name2)
  #current4 = slim.max_pool2d(current3, [2, 2], stride=2, scope=name2)
  return current4

def TransitionDown(input, filters, keep_prob, name):
  current = input
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  name3=name+'BN'
  current = slim.batch_norm(current, scope=name3)
  current = tf.nn.elu(current)
	#current = conv2d(current, in_features, filters, 1)
  current2 = slim.conv2d(current, filters, [1, 1], scope=name)
  current3= tf.nn.dropout(current2, keep_prob)
  name2=name+'pool'
  #print(name2)
  #current = max_pool(current, 2)
  #Before
  current4 = slim.max_pool2d(current3, [2, 2], scope=name2)
  #current4 = slim.max_pool2d(current3, [2, 2], stride=2, scope=name2)
  return current4

def TransitionDown_Swish(input, filters, keep_prob, name):
  current = input
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  name3=name+'BN'
  current = slim.batch_norm(current, scope=name3)
  current = Swish_ActivationFunction(current)
  #current = conv2d(current, in_features, filters, 1)
  current2 = slim.conv2d(current, filters, [1, 1], scope=name)
  current3= tf.nn.dropout(current2, keep_prob)
  name2=name+'pool'
  #print(name2)
  #current = max_pool(current, 2)
  #Before
  current4 = slim.max_pool2d(current3, [2, 2], scope=name2)
  #current4 = slim.max_pool2d(current3, [2, 2], stride=2, scope=name2)
  return current4


def TransitionUp_Final(self, input, filters):
  current = input
  pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
  current_pad = tf.pad(input, pattern)
  print("Current_pad")
  print(current_pad.shape)
  output_shape = [self.batch_size, self.height, self.width, filters]
  print("Output Shape")
  print(output_shape)
  print("filter shape")
  filter_shape = [3, 3, filters, current.shape[3]]
  print(filter_shape)
  print("input shape")
  print(current.shape)
  strides = [1, 2, 2, 1]
  with tf.variable_scope("g_h2"):
    w = tf.get_variable('w', filter_shape, initializer=tf.random_normal_initializer(stddev=0.02))
    h2 = tf.nn.conv2d_transpose(current, w, output_shape=output_shape,  strides=strides, padding='SAME')

  print("Get out of it") 
  print(h2.shape) 
  return h2


def TransitionUp_shape(self, input, filters, size, name):
  current = input
  output_shape = [self.batch_size,size, size, filters]
  filter_shape = [3, 3, filters, current.shape[3]]
  strides = [1, 2, 2, 1]
  filter_shape = [3, 3, filters, current.shape[3]]
  with tf.variable_scope(name):
    w = tf.get_variable(name, filter_shape, initializer=tf.random_normal_initializer(stddev=0.02))
    h2 = tf.nn.conv2d_transpose(current, w, output_shape=output_shape,  strides=strides, padding='SAME')
  #print("Get out of it") 
  #print(h2.shape) 
  return h2

def TransitionUp_shapeV2(self, input, filters, size1, size2, name):
  current = input
  output_shape = [self.batch_size,size1, size2, filters]
  filter_shape = [3, 3, filters, current.shape[3]]
  strides = [1, 2, 2, 1]
  filter_shape = [3, 3, filters, current.shape[3]]
  with tf.variable_scope(name):
    w = tf.get_variable(name, filter_shape, initializer=tf.random_normal_initializer(stddev=0.02))
    h2 = tf.nn.conv2d_transpose(current, w, output_shape=output_shape,  strides=strides, padding='SAME')
  #print("Get out of it") 
  #print(h2.shape) 
  return h2 

def TransitionUp_shape_stride(self, input, filters, size, stride, name):
  current = input
  output_shape = [self.batch_size,size, size, filters]
  filter_shape = [3, 3, filters, current.shape[3]]
  strides = [1, stride, stride, 1]
  filter_shape = [3, 3, filters, current.shape[3]]
  with tf.variable_scope(name):
    w = tf.get_variable(name, filter_shape, initializer=tf.random_normal_initializer(stddev=0.02))
    h2 = tf.nn.conv2d_transpose(current, w, output_shape=output_shape,  strides=strides, padding='SAME')
  #print("Get out of it") 
  #print(h2.shape) 
  return h2 

   
       
 

def TransitionUp(input, filters, name, Final=False):
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=2,  scope=name)
  upconv = tf.nn.elu(upconv)
  #print(upconv)
  return upconv

def TransitionUp_elu(input, filters, name, Final=False):
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  #output_shape = [1, 75, 75, 128]
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=2,  scope=name)
  upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv

def TransitionUp_elu_quarter(input, filters, name, Final=False):
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  #output_shape = [1, 75, 75, 128]
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=4,  scope=name)
  upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv

def TransitionUp_swish(input, filters, name, Final=False):
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  #output_shape = [1, 75, 75, 128]
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=2,  scope=name)
  upconv = Swish_ActivationFunction(upconv)
  #upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv

############################################################
############################################################
#PSPNET MODULES
############################################################
############################################################

def empty_branch(prev):
  return prev

def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
  lvl = str(lvl)
  sub_lvl = str(sub_lvl)
  names = ["conv"+lvl+"_"+ sub_lvl +"_1x1_reduce" ,
          "conv"+lvl+"_"+ sub_lvl +"_1x1_reduce_bn",
          "conv"+lvl+"_"+ sub_lvl +"_3x3",
          "conv"+lvl+"_"+ sub_lvl +"_3x3_bn",
          "conv"+lvl+"_"+ sub_lvl +"_1x1_increase",
          "conv"+lvl+"_"+ sub_lvl +"_1x1_increase_bn"]
  #conv1 = slim.conv2d(inp, 64, [3, 3], stride=2, scope='COnv1', padding='SAME', biases_initializer=None)
  #bn1 = slim.batch_norm(conv1,activation_fn=None)
  #relu1 = tf.nn.relu(bn1
  if modify_stride == False:
      #prev = Conv2D(64 * level, (1,1), strides=(1,1), name=names[0], use_bias=False)(prev)
      prev = slim.conv2d(prev, 64 * level, [1, 1], stride=1, scope=names[0], padding='SAME', biases_initializer=None)
  elif modify_stride == True:
      #prev = Conv2D(64 * level, (1,1), strides=(2,2), name=names[0], use_bias=False)(prev)
      prev = slim.conv2d(prev, 64 * level, [1, 1], stride=2, scope=names[0], padding='SAME', biases_initializer=None)

  #prev = BN(name=names[1])(prev)
  #prev = Activation('relu')(prev)
  bn = slim.batch_norm(prev,activation_fn=None, scope=names[1])
  relu = tf.nn.relu(bn)

  #prev = ZeroPadding2D(padding=(pad,pad))(prev)
  
  #prev = Conv2D(64 * level, (3,3), strides=(1,1), dilation_rate=pad, name=names[2], use_bias=False)(prev)
  CONV = atrous_conv(relu, 3, 3, 64 * level, dilation=pad, name=names[2], padding='SAME', biased=False, relu=False, trainable=True, group=1)

  #prev = BN(name=names[3])(prev)
  #prev = Activation('relu')(prev)
  bn = slim.batch_norm(CONV,activation_fn=None, scope=names[3])
  relu = tf.nn.relu(bn)

  #prev = Conv2D(256 * level, (1,1), strides=(1,1), name=names[4], use_bias=False)(prev)
  #prev = BN(name=names[5])(prev)
  prev = slim.conv2d(relu, 256 * level, [1, 1], stride=1, scope=names[4], padding='SAME', biases_initializer=None)
  prev = slim.batch_norm(prev,activation_fn=None, scope=names[5])
  return prev

def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
  lvl = str(lvl)
  sub_lvl = str(sub_lvl)
  names = ["conv"+lvl+"_"+ sub_lvl +"_1x1_proj",
          "conv"+lvl+"_"+ sub_lvl +"_1x1_proj_bn"]

  if modify_stride == False:      
      #prev = Conv2D(256 * level ,(1,1), strides=(1,1), name=names[0], use_bias=False)(prev)
      prev = slim.conv2d(prev, 256 * level, [1, 1], stride=1, scope=names[0], padding='SAME', biases_initializer=None)
  elif modify_stride == True:
      #prev = Conv2D(256 * level, (1,1), strides=(2,2), name=names[0], use_bias=False)(prev)
      prev = slim.conv2d(prev, 256 * level, [1, 1], stride=2, scope=names[0], padding='SAME', biases_initializer=None)

  #prev = BN(name=names[1])(prev)
  prev = slim.batch_norm(prev,activation_fn=None, scope=names[1])
  return prev



def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
  #prev_layer = Activation('relu')(prev_layer)
  prev_layer = tf.nn.relu(prev_layer)

  block_1 = residual_conv(prev_layer, level, pad=pad, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride)

  block_2 = short_convolution_branch(prev_layer, level, lvl=lvl, sub_lvl=sub_lvl, modify_stride=modify_stride)

  added = tf.add(block_1, block_2)
  return added

def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
  #prev_layer = Activation('relu')(prev_layer)
  prev_layer = tf.nn.relu(prev_layer)
  block_1 = residual_conv(prev_layer, level, pad=pad, lvl=lvl, sub_lvl=sub_lvl)
  block_2 = empty_branch(prev_layer)
  #added = Add()([block_1, block_2])
  added = tf.add(block_1, block_2)
  return added

def ResNet(self, inp):
  #Names for the first couple layers of model
  # names = ["conv1_1_3x3_s2",
  #         "conv1_1_3x3_s2_bn",
  #         "conv1_2_3x3",
  #         "conv1_2_3x3_bn",
  #         "conv1_3_3x3",
  #         "conv1_3_3x3_bn"]

  #---Short branch(only start of network)

  # cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp) # "conv1_1_3x3_s2"
  # bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
  # relu1 = Activation('relu')(bn1)             #"conv1_1_3x3_s2/relu"

  conv1 = slim.conv2d(inp, 64, [3, 3], stride=2, scope='COnv1', padding='SAME', biases_initializer=None)
  bn1 = slim.batch_norm(conv1,activation_fn=None, scope='COnv1_bn')
  relu1 = tf.nn.relu(bn1)
  

  #cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1) #"conv1_2_3x3"
  #bn1 = BN(name=names[3])(cnv1)  #"conv1_2_3x3/bn"
  #relu1 = Activation('relu')(bn1)                 #"conv1_2_3x3/relu"
  conv1_2 = slim.conv2d(relu1, 64, [3, 3], stride=1, scope='COnv1_2', padding='SAME', biases_initializer=None)
  bn1_2 = slim.batch_norm(conv1_2,activation_fn=None, scope='COnv1_2_bn')
  relu1_2 = tf.nn.relu(bn1_2)

  #cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4], use_bias=False)(relu1) #"conv1_3_3x3"
  #bn1 = BN(name=names[5])(cnv1)      #"conv1_3_3x3/bn"
  #relu1 = Activation('relu')(bn1)             #"conv1_3_3x3/relu"
  conv1_3 = slim.conv2d(relu1_2, 64, [3, 3], stride=1, scope='COnv1_3', padding='SAME', biases_initializer=None)
  bn1_3 = slim.batch_norm(conv1_3,activation_fn=None, scope='COnv1_3_bn')
  relu1_3 = tf.nn.relu(bn1_3)

  #res = MaxPooling2D(pool_size=(3,3), padding='same', strides=(2,2))(relu1)  #"pool1_3x3_s2"
  res = slim.max_pool2d(bn1, 3, stride=2, padding='SAME', scope='pool1_3x3_s2')
  res_skip=res
  #---Residual layers(body of network)

  """
  Modify_stride --Used only once in first 3_1 convolutions block.
  changes stride of first convolution from 1 -> 2
  """

  #2_1- 2_3
  res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1) 
  for i in range(3):
      res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2) 

  #3_1 - 3_3
  res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True) 
  for i in range(4):
      res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2) 

  #4_1 - 4_6
  res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1) 
  for i in range(20):
      res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2) 

  #5_1 - 5_3
  res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1) 
  for i in range(3):
      res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2)

  #res = Activation('relu')(res)
 # res = tf.nn.relu(res)


  current_up1 = TransitionUp_shape_stride(self, res, 512, 75, stride=2, name='Upconv1')
  skipConnection1 = Concat_layers(current_up1, res_skip)
  convup1_3 = slim.conv2d(skipConnection1, 576, [3, 3], stride=1, scope='COnv7', padding='SAME', biases_initializer=None)
  bnup1_3 = slim.batch_norm(convup1_3,activation_fn=None, scope='COnv7_bn')
  reluup1_3 = tf.nn.relu(bnup1_3)

  current_up2 = TransitionUp_elu(reluup1_3, 256, name='Upconv2')
  skipConnection2 = Concat_layers(current_up2, conv1_3)
  conv2up_3 = slim.conv2d(skipConnection2, 320, [3, 3], stride=1, scope='COnv8', padding='SAME', biases_initializer=None)
  bn2up_3 = slim.batch_norm(conv2up_3,activation_fn=None, scope='COnv8_bn')
  reluup2_3 = tf.nn.relu(bn2up_3)
  print("BEfore FInal upsampling")
  print(reluup2_3)
  shape = tf.shape(inp)

  height = shape[1]
  print("Height")
  print(height)
  up_final = TransitionUp_elu(reluup2_3, 256, name='Upconvfinal') 
  res_conv = slim.conv2d(up_final, 256, [3, 3], stride=1, scope='COnv9', padding='SAME', biases_initializer=None)
  res = slim.batch_norm(res_conv,activation_fn=None, scope='COnv9_bn')
  res = tf.nn.relu(res)

  print("END RESNET END END END END")
  return res

def interp_block(self, prev_layer, level, str_lvl=1):

  str_lvl = str(str_lvl)
  shape = tf.shape(prev_layer)
  names = [
      "conv5_3_pool"+str_lvl+"_conv",
      "conv5_3_pool"+str_lvl+"_conv_bn",
      "POOL5_3_pool"+str_lvl+"_conv_bn",
      "CC"+str_lvl+"_conv_UP",
      ]

  kernel = (10*level, 10*level)
  strides = (10*level, 10*level)
  #prev_layer = AveragePooling2D(kernel,strides=strides)(prev_layer)
  prev_layer = slim.avg_pool2d(prev_layer, kernel_size=10*level, stride=10*level, padding='SAME', scope=names[2])
  #prev_layer = Conv2D(512, (1,1), strides=(1,1), name=names[0], use_bias=False)(prev_layer)
  prev_layer = slim.conv2d(prev_layer, 512, [1, 1], stride=1, scope=names[0], padding='SAME', biases_initializer=None)

  #prev_layer = BN(name=names[1])(prev_layer)
  prev_layer = slim.batch_norm(prev_layer,activation_fn=None, scope=names[1])

  #prev_layer = Activation('relu')(prev_layer)
  prev_layer = tf.nn.relu(prev_layer)
  #Upsample data
  #prev_layer = Lambda(Interp)(prev_layer)
  height = shape[1]
  print("Height")
  print(height)
  prev_layer = TransitionUp_shape_stride(self, prev_layer, 512, height, stride=10*level, name=names[3])
  return prev_layer

def PSPNet_build(self, res):

  #---PSPNet concat layers with Interpolation
  print("BEFORE STARTING PSP ")
  interp_block1 = interp_block(self,res, 6, str_lvl=1)
  print("Block1 ")
  interp_block2 = interp_block(self, res, 3, str_lvl=2)
  interp_block3 = interp_block(self, res, 2, str_lvl=3)
  interp_block6 = interp_block(self, res, 1, str_lvl=6)

  #concat all these layers. resulted shape=(1,60,60,4096)
  # res = Concatenate()([res,
  #                 interp_block6,
  #                 interp_block3,
  #                 interp_block2,
  #                 interp_block1])
  concat_convs = tf.concat((res, interp_block1, interp_block2, interp_block3, interp_block6), 3)

  return res