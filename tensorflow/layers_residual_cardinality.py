from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
from layers_slim import * 
from group_conv import *
from SE_layers import *


def batch_activ_conv_Group(current, in_features, out_features, kernel_size, is_training, keep_prob, name, Groups):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current, activation_fn=None)
  #current2 = Swish_ActivationFunction(current)
  current2 = tf.nn.elu(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  #current3 = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)
  stride = 1
  current3 = groupConvolution(current2, kernel_size, kernel_size, out_features, stride, stride, name, padding='SAME', groups=Groups)
  current4 = tf.nn.dropout(current3, keep_prob)
  
  return current4

def batch_activ_conv_Group_swish(current, in_features, out_features, kernel_size, is_training, keep_prob, name, Groups):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current, activation_fn=None)
  current2 = Swish_ActivationFunction(current)
  #current2 = tf.nn.elu(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  #current3 = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)
  stride = 1
  current3 = groupConvolution(current2, kernel_size, kernel_size, out_features, stride, stride, name, padding='SAME', groups=Groups)
  current4 = tf.nn.dropout(current3, keep_prob)
  
  return current4


def Denseblock_Group(input, layers, Din_features, growth, is_training, keep_prob, name, Groups):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv_Group(current, features, growth, 3, is_training, keep_prob, name2, Groups)
    #print(current)
    #print(tmp)
    current =  tf.concat((current, tmp), 3)
    features += growth
  return current

def Denseblock_Group_SWISH(input, layers, Din_features, growth, is_training, keep_prob, name, Groups):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in xrange(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv_Group_swish(current, features, growth, 3, is_training, keep_prob, name2, Groups)
    #print(current)
    #print(tmp)
    current =  tf.concat((current, tmp), 3)
    features += growth
  return current


def Up_Conv(input, filters, name, scale, Final=False):
  stride=scale
  #print("Transition UP")
  current = input
  if Final:
    pattern = [[0, 0], [2, 2], [0, 0], [0, 0]]
    current = tf.pad(current, pattern)
  #print(name)
  #output_shape = [1, 75, 75, 128]
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=stride,  scope=name)
  upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv


def concatenation(layers):
  return tf.concat(layers, 3)


def BN_ReLU_Conv_drop(current, out_features, kernel_size, stride, name, dropout):
  BN = slim.batch_norm(current,activation_fn=None)
  RELU = tf.nn.relu(BN)
  if stride > 1:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')

  else:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name)
  conv_drop = tf.nn.dropout(CONV, dropout)

  return conv_drop

#internal operation - 1x1 conv followed by a 3x3 conv 
def transform(x, depth, card_step, stride, name):
  name2=name +'_'+ str(card_step) +'_'+ 'c1x1'
  #Bottleneck
  x = slim.batch_norm(x,activation_fn=None)
  x = tf.nn.relu(x)
  x = slim.conv2d(x, depth, [1, 1], stride=stride, scope=name2)

  name2=name +'_'+ str(card_step) +'_'+ 'c3x3'
  #Bottleneck
  x = slim.batch_norm(x,activation_fn=None)
  x = tf.nn.relu(x)
  x = slim.conv2d(x, depth, [3, 3], stride=stride, scope=name2)

  return x


def FusionInternalBlocks(x, out_features, kernel_size, stride, name):
  name2=name +'_T_'+ 'c1x1'
  #Bottleneck
  x = slim.batch_norm(x,activation_fn=None)
  #x = tf.nn.relu(x)
  x = slim.conv2d(x, out_features, [1, 1], stride=stride, scope=name2)

  return x


def internalBlock(x, out_features, kernel_size, stride, name, cardinality):
  #setting depth to 4
  depth=2

  layers_split=list()

  for i in range(cardinality):
    splits = transform(x, depth, card_step=i, stride=1, name=name)
    layers_split.append(splits)

  return concatenation(layers_split)



def Residual_cardinality(x, out_features, kernel_size, stride, name, Group):
  #cardinality equal to groups, in this case is 32 groups     
  # So first compute the main internal block - 1x1 conv followed by 3x3 
  
  x = internalBlock(x, out_features, kernel_size, stride, name, cardinality=Group)

  #x = FusionInternalBlocks(x, out_features, kernel_size, stride=stride, name=name) 

  return x

def BN_ReLU_Conv_group(current, out_features, kernel_size, stride, name, groups=1):
  BN = slim.batch_norm(current,activation_fn=None)
  RELU = tf.nn.relu(BN)
  if stride > 1:
    CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')
  else:
    #name2 = name + 'compress'
    #CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name2)
    #print("out-feature size")
    #print(out_features)
    CONV = groupConvolution(RELU, kernel_size, kernel_size, out_features, stride, stride, name, padding='SAME', groups=groups)
    
    #name2 = name + 'decompress'
    #CONV = slim.conv2d(CONV, out_features, [kernel_size, kernel_size], stride=stride, scope=name2)
    
  #conv_drop = tf.nn.dropout(CONV, dropout)

  return CONV

def BN_ReLU_Conv_group_Swish(current, out_features, kernel_size, stride, name, groups=1):
  BN = slim.batch_norm(current,activation_fn=None)
  #RELU = tf.nn.relu(BN)
  Swish = Swish_ActivationFunction(BN)

  if stride > 1:
    CONV = slim.conv2d(Swish, out_features, [kernel_size, kernel_size], stride=stride, scope=name, padding='VALID')
  else:
    #name2 = name + 'compress'
    #CONV = slim.conv2d(RELU, out_features, [kernel_size, kernel_size], stride=stride, scope=name2)
    #print("out-feature size")
    #print(out_features)
    CONV = groupConvolution(Swish, kernel_size, kernel_size, out_features, stride, stride, name, padding='SAME', groups=groups)
    
    #name2 = name + 'decompress'
    #CONV = slim.conv2d(CONV, out_features, [kernel_size, kernel_size], stride=stride, scope=name2)
    
  #conv_drop = tf.nn.dropout(CONV, dropout)

  return CONV


def SE_block(input_v, out_dim, name, G):
  stride=1 
  
  # name2 = name + 'split'
  # x = split(input_v, out_dim, stride, name=name2)
  # name2 = name + 'transition'
  # x = transition(x, out_dim=out_dim, name=name2)
  name2 = name + 'group'
  x = BN_ReLU_Conv_group(input_v, out_features=out_dim, kernel_size=3, stride=1, name=name2, groups=G)
  name2 = name + "SE_"
  x = squeeze_excitation_layer(x, out_dim, ratio=reduction_ratio, name=name2) 

  # print("X ")
 #    print(x)

 #    print("input_v ")
 #    print(input_v)

  # input_v = Relu(x + input_v)

  return x


##########################################################
##########################################################
###################Multi-Path Block ######################
######## Fusing Residual, Dense and SE  ################## 
############ capabilities 30/01/18########################
##########################################################

def M_Block(input, num_1x1_a, num_3x3_b, num_1x1_c, num_SE, inc, name, _type):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  stride=1
  
  #Just for debugging purposes 
  # Group == cardinality == 32
  G=32  

  if isinstance(input, list):
    init = tf.concat((input[0], input[1], input[2]), 3)
  else:
    init = input

  if _type is 'create':
    create = True
  if _type is 'dense':
    create = False

  if create:
    name2 = name + 'residual'
    projection_path = BN_ReLU_Conv_group(init, out_features=num_1x1_c+2*inc+num_SE, kernel_size=1, stride=stride, name=name2) 
    input_residual_path = projection_path[:,:,:,:num_1x1_c]   
    input_dense_path = projection_path[:,:,:,num_1x1_c:num_1x1_c+2*inc]
    input_SE_path = projection_path[:,:,:,num_1x1_c+2*inc:]
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]
    input_SE_path = input[2]
  ###################################DENSE PATH ##################################################################
  name2 = name + 'c1x1a'
  x = BN_ReLU_Conv_group(init, out_features=num_1x1_a, kernel_size=1, stride=1, name=name2)   
  name2 = name + 'c3x3b'
  x = BN_ReLU_Conv_group(x, out_features=num_3x3_b, kernel_size=3, stride=1, name=name2, groups=G)
  name2 = name + 'c1xca'
  x = BN_ReLU_Conv_group(x, out_features=num_1x1_c+inc, kernel_size=1, stride=stride, name=name2)

  ###################################SE PATH #####################################################################  
  name2 = name + 'SE_block'
  x2 = SE_block(init, num_SE, name2, G)
  name2 = name + 'c1_SE'
  #x2 = BN_ReLU_Conv_group(x2, out_features=num_SE, kernel_size=1, stride=stride, name=name2)
  output_SE_path = x2
  # print("X2 ")
  # print(x2)

  # print("input_SE_path ")
  # print(input_SE_path)

  output_residual_path = x[:,:,:,:num_1x1_c]
  output_dense_path = x[:,:,:,num_1x1_c:num_1x1_c+2*inc]





  residual_path =  tf.add(input_residual_path, output_residual_path)
  dense_path = tf.concat((input_dense_path, output_dense_path), 3)
  
  SE_path = tf.add(input_SE_path,output_SE_path)
  
  # if has_proj:
  #   concat_convs = tf.concat((c1x1_w, c1x1_c), 3)
  #   return concat_convs
  
  return [residual_path, dense_path, SE_path]


##########################################################
##########################################################
#######New version of DPDB block with ####################
######## Squeeze and excitation modeles ################## 
#####################29/01/18#############################
##########################################################

def DPDB_Block_SE(input, num_1x1_a, num_3x3_b, num_1x1_c, inc, name, _type):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  stride=1
  
  #Just for debugging purposes 
  # Group == cardinality == 32
  G=32  

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
    projection_path = BN_ReLU_Conv_group(init, out_features=num_1x1_c+2*inc, kernel_size=1, stride=stride, name=name2) 
    input_residual_path = projection_path[:,:,:,:num_1x1_c]   
    input_dense_path = projection_path[:,:,:,num_1x1_c:]
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]

  name2 = name + 'c1x1a'
  x = BN_ReLU_Conv_group(init, out_features=num_1x1_a, kernel_size=1, stride=1, name=name2)   
  name2 = name + 'c3x3b'
  x = BN_ReLU_Conv_group(x, out_features=num_3x3_b, kernel_size=3, stride=1, name=name2, groups=G)

  #Squeeze and excitation modules
  name2 = name + 'sq_ex' 
  x = squeeze_excitation_layer(x, out_dim=num_3x3_b, ratio=16, name=name2)


  name2 = name + 'c1xca'
  x = BN_ReLU_Conv_group(x, out_features=num_1x1_c+inc, kernel_size=1, stride=stride, name=name2)  

  output_residual_path = x[:,:,:,:num_1x1_c]
  output_dense_path = x[:,:,:,num_1x1_c:]

  residual_path =  tf.add(input_residual_path, output_residual_path)
  dense_path = tf.concat((input_dense_path, output_dense_path), 3)

  # if has_proj:
  #   concat_convs = tf.concat((c1x1_w, c1x1_c), 3)
  #   return concat_convs
  
  return [residual_path, dense_path]




##########################################################
##########################################################
#######New version of DPDB block with cadinality########## 
#########################18/01/18#########################
##########################################################

def DPDB_Block_V2(input, num_1x1_a, num_3x3_b, num_1x1_c, inc, name, _type, Groups):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  stride=1
  
  #Just for debugging purposes 
  # Group == cardinality == 32
  G=Groups  

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
    projection_path = BN_ReLU_Conv_group(init, out_features=num_1x1_c+2*inc, kernel_size=1, stride=stride, name=name2) 
    input_residual_path = projection_path[:,:,:,:num_1x1_c]   
    input_dense_path = projection_path[:,:,:,num_1x1_c:]
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]

  name2 = name + 'c1x1a'
  x = BN_ReLU_Conv_group(init, out_features=num_1x1_a, kernel_size=1, stride=1, name=name2)   
  name2 = name + 'c3x3b'
  x = BN_ReLU_Conv_group(x, out_features=num_3x3_b, kernel_size=3, stride=1, name=name2, groups=G)
  name2 = name + 'c1xca'
  x = BN_ReLU_Conv_group(x, out_features=num_1x1_c+inc, kernel_size=1, stride=stride, name=name2)  

  output_residual_path = x[:,:,:,:num_1x1_c]
  output_dense_path = x[:,:,:,num_1x1_c:]

  residual_path =  tf.add(input_residual_path, output_residual_path)
  dense_path = tf.concat((input_dense_path, output_dense_path), 3)

  # if has_proj:
  #   concat_convs = tf.concat((c1x1_w, c1x1_c), 3)
  #   return concat_convs
  
  return [residual_path, dense_path]

##########################################################
##########################################################
#######New version of DPDB block with cadinality and SWish########## 
#########################12/03/18#########################
##########################################################

def DPDB_Block_V3(input, num_1x1_a, num_3x3_b, num_1x1_c, inc, name, _type, Groups):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  stride=1
  
  #Just for debugging purposes 
  # Group == cardinality == 32
  G=Groups  

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
    projection_path = BN_ReLU_Conv_group_Swish(init, out_features=num_1x1_c+2*inc, kernel_size=1, stride=stride, name=name2) 
    input_residual_path = projection_path[:,:,:,:num_1x1_c]   
    input_dense_path = projection_path[:,:,:,num_1x1_c:]
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]

  name2 = name + 'c1x1a'
  x = BN_ReLU_Conv_group_Swish(init, out_features=num_1x1_a, kernel_size=1, stride=1, name=name2)   
  name2 = name + 'c3x3b'
  x = BN_ReLU_Conv_group_Swish(x, out_features=num_3x3_b, kernel_size=3, stride=1, name=name2, groups=G)
  name2 = name + 'c1xca'
  x = BN_ReLU_Conv_group_Swish(x, out_features=num_1x1_c+inc, kernel_size=1, stride=stride, name=name2)  

  output_residual_path = x[:,:,:,:num_1x1_c]
  output_dense_path = x[:,:,:,num_1x1_c:]

  residual_path =  tf.add(input_residual_path, output_residual_path)
  dense_path = tf.concat((input_dense_path, output_dense_path), 3)

  # if has_proj:
  #   concat_convs = tf.concat((c1x1_w, c1x1_c), 3)
  #   return concat_convs
  
  return [residual_path, dense_path]

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

