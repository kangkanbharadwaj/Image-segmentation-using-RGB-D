from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

from layers_slim import *
from layers_residual_cardinality import *


def zero_padding(X, pad=(3, 3)):
    paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                            [pad[1], pad[1]], [0, 0]])
    return tf.pad(X, paddings, 'CONSTANT')    

# Define convolution layer
def conv2D(input_, training, filters, k_size, strides, padding, name): 
    """
    @params
        input_ (4-D Tensor): (batch_size, H, W, C)
        filters (list): number of output channels        
        name (str): name postfix
        k_size: size of the filter
        padding: to pad with '0' or not
        strides: how much to slide the kernel on the input image
    Returns:
        net: output of the Convolution operations        
    """
    net = tf.layers.conv2d(input_,filters, k_size, strides, padding, activation=None,name=name) 
    return net

# Used for a mini batch size > 1
def nn_batch_norm(X, name):
    m_, v_ = tf.nn.moments(X, axes=[0, 1, 2], keep_dims=False)
    beta_ = tf.zeros(X.shape.as_list()[3])
    gamma_ = tf.ones(X.shape.as_list()[3])
    bn = tf.nn.batch_normalization(X, mean=m_, variance=v_, offset=beta_, scale=gamma_, variance_epsilon=1e-4)
    return bn

def layer_batch_norm(X, name):
    bn = tf.layers.batch_normalization(X, axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=False, name=name)
    return bn


def _batch_norm(X, training, name):
    bn = tf.cond(training, lambda: tf.contrib.layers.batch_norm(X, decay=0.9, is_training=True, center=True, scale=True, activation_fn=None, updates_collections=None, scope=name), 
                 lambda: tf.contrib.layers.batch_norm(X, decay=0.9, is_training=False, center=True, scale=True, activation_fn=None, updates_collections=None, scope=name, reuse=True))
    return bn
    
    
def batch_norm(X, name):
    bn = slim.batch_norm(X, scope=name)
    return bn    
    
# Used for a mini batch size = 1
def layer_norm(X, name):
        ln = tf.contrib.layers.batch_norm(X, name)
        return ln

# Implementing a ResNet bottleneck identity block with shortcut path passing over 3 Conv Layers
def identity_block(X, training, f, filters, stage, block):
    """
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers

    @returns
    A - Output of identity_block
    params - Params used in identity block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'ln' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    A1 = conv2D(X, training, filters=l1_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)

    A3 = conv2D(A2_act, training, filters=l3_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2c')
    A3_bn= batch_norm(A3, name=bn_name+'2c')

    A3_add = tf.add(A3_bn, X)
    A = tf.nn.relu(A3_add)
    return A


def identity_block_basic(X, training, f, filters, stage, block):
    """
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers

    @returns
    A - Output of identity_block
    params - Params used in identity block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'ln' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    A1 = conv2D(X, training, filters=l1_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')

    A2_add = tf.add(A2_bn, X)
    A = tf.nn.relu(A2_add)
    return A

# Implementing a ResNet basic convolution block with shortcut path passing over 3 Conv Layers having different sizes
def convolutional_block(X, training, f, filters, stage, block, s=2):
    """
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    s - strides used in first layer of convolutional block

    @returns
    A - Output of convolutional_block
    params - Params used in convolutional block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'ln' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    A1 = conv2D(X, training, filters=l1_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)

    A3 = conv2D(A2_act, training, filters=l3_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')

    A_ = conv2D(X, training, filters=l3_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A3_add = tf.add(A3_bn, A_bn_)
    A = tf.nn.relu(A3_add)
    return A


# Implementing a ResNet convolutional basic block with shortcut path passing over 3 Conv Layers having different sizes
def convolutional_block_basic(X, training, f, filters, stage, block, s=2):
    """
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    s - strides used in first layer of convolutional block

    @returns
    A - Output of convolutional_block
    params - Params used in convolutional block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'ln' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    A1 = conv2D(X, training, filters=l1_f, k_size=(f, f), strides=(s, s), padding='SAME', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')

    A_ = conv2D(X, training, filters=l3_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A2_add = tf.add(A2_bn, A_bn_)
    A = tf.nn.relu(A2_add)
    return A

# Transpose convolution / De convolution to get scale up resolution
def upconv_2D(tensor, training, n_filter, name):
    """
    @params
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """
    #return slim.conv2d_transpose(tensor, n_filter, [2, 2], stride=2,  padding='valid', scope="upsample_{}".format(name))
    return tf.layers.conv2d_transpose(tensor, filters=n_filter, kernel_size=(2, 2), strides=(2, 2), padding='valid', name="upsample_{}".format(name))

# Upsample `inputA` and concat with `input_B`
def upconv_concat(inputA, input_B, training, n_filter, name):
    """
    @params
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    
    up_conv = upconv_2D(inputA, training, n_filter, name)    
    _bn = batch_norm(up_conv, name="bn_upconv_{}".format(name))
    _act = tf.nn.relu(_bn)
    _concat = tf.concat([_act, input_B], axis=-1, name="up_concat_{}".format(name))
    
    return _concat

# Convolve `inputA` and concat with previous layer
def conv_concat(inputA, training, filters, k_size, strides, padding, name):
    """
    @params
        input_A (4-D Tensor): (N, H, W, C)
        filters: Number of channels to output
        name (str): name of the concat operation
        k_size: size of the filter
        padding: to pad with '0' or not
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    _conv = conv2D(inputA, training, filters, k_size, strides, padding, name="convolution_{}".format(name))
    _bn = batch_norm(_conv, name="bn_conv_{}".format(name))
    _act = tf.nn.relu(_bn)
    _concat = tf.concat([_act, inputA], axis=-1, name="concat_{}".format(name))
    return _concat


def DPDB_encoder_UNet_decoder(image, is_training=True, N_Class=0):
    ############################################################
    ### DPDB encoder with deep supervised stacked decoder ######
    ### With three stages                                 ######
    ############################################################
    print ("Number of classes are %d" %(N_Class))
    batch_norm_params = {'is_training': is_training}
    keep_prob = 0.1
    Groups = 4
    #H = self.tgt_image.get_shape()[1].value
    #W = self.tgt_image.get_shape()[2].value
    
    #end_points_collection = sc.original_name_scope + '_end_points'
    #Variable dpn92        
    k_R=128
    #For group == 48 change
    k_sec=(3, 4, 6, 3)
    #k_sec=(4, 4, 6, 9, 11)
    inc_sec=(16,16,32,24,64)

    print (image)
    
    ##Conv1 
    Conv1 = slim.conv2d(image, 96, [7, 7], stride=1, scope='FirstConv')
    print("Conv1 ")
    print (Conv1)

    ##BLOCK 1 
    blockWidth = 224
    inc = inc_sec[0]
    R = int((k_R*blockWidth)/256) 
    #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
    name='conv1'
    x = DPDB_Block_V2(Conv1, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

    for idx in range(k_sec[0]):
        name2 = name + str(idx)
        #print(name2)
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        print("Block: ", idx)
    
    print(x)
    DB1_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 32, [1, 1], scope='skip1')        

    x = TransitionDown(tf.concat((x[0], x[1]), 3), 240, keep_prob, name='convDown1')
    
    ##BLOCK 2 
    blockWidth = 448
    inc = inc_sec[1]
    R = int((k_R*blockWidth)/256) 
    #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
    name='conv2'
    x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

    for idx in range(k_sec[1]):
        name2 = name + str(idx)
        #print(name2)
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        print("Block: ", idx)

    print("Conv2 ")
    print(x)
    DB2_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 64, [1, 1], scope='skip2')        

    x = TransitionDown(tf.concat((x[0], x[1]), 3), 384, keep_prob, name='convDown2')
    
    ##BLOCK 3 
    blockWidth = 896
    inc = inc_sec[2]
    R = int((k_R*blockWidth)/256) 
    #currentC3 = DualPathInterBlock(currentC2, inc, k_sec[1],  blockWidth, R, name='conv3', residual_type='down')
    name='conv3'
    x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

    for idx in range(k_sec[2]):
        name2 = name + str(idx)
        #print(name2)
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        print("Block: ", idx)

    print("Conv3 ")
    print(x)
    #DB3_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , blockWidth/2, [1, 1], scope='skip3')        
    DB3_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 92, [1, 1], scope='skip3')        

    x = TransitionDown(tf.concat((x[0], x[1]), 3), 832, keep_prob, name='convDown3')
    

    ##BLOCK 4 
    blockWidth = 1792
    inc = inc_sec[3]
    R = int((k_R*blockWidth)/256) 
    #currentC4 = DualPathInterBlock(currentC3, inc, k_sec[2],  blockWidth, R, name='conv4', residual_type='down')
    name='conv4'
    x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

    for idx in range(k_sec[3]):
        name2 = name + str(idx)
        #print(name2)
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        print("Block: ", idx)
        
    print("Conv4 ")
    print(x)  
    
    DB4_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 128, [1, 1], scope='skip4')        
        
    print ("\nDecoder on the go!!!")
    
    # up stage 1     
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    #cnct4_1 = upconv_concat(A_6_ib6, A_5_ib5, 1024, name=4)
    #concat4_2 = conv_concat(cnct4_1, 1024, 3, strides=(1, 1), padding='SAME', name='4a')
    #concat4_3 = conv_concat(concat4_2, 1024, 3, strides=(1, 1), padding='SAME', name='4b')
    
    #print ("Tensor shape for the 4th stage: " + str(concat4_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat4_3], print_info=True)))
    
    # up stage 2    
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    
    cnct3_1 = upconv_concat(DB4_skip_connection, DB3_skip_connection, is_training, 896, name=3)
    concat3_2 = conv_concat(cnct3_1, is_training, 896, 3, strides=(1, 1), padding='SAME', name='3a')
    concat3_3 = conv_concat(concat3_2, is_training, 896, 3, strides=(1, 1), padding='SAME', name='3b')
    
    print ("Tensor shape for the up convolution 1 stage: " + str(concat3_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat3_3], print_info=True)))

    # up stage 3
    #print ("\nEntering up_convolve stage 2")
    
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    cnct2_1 = upconv_concat(concat3_3, DB2_skip_connection, is_training, 448, name=2)
    concat2_2 = conv_concat(cnct2_1, is_training, 448, 3, strides=(1, 1), padding='SAME', name='2a')
    concat2_3 = conv_concat(concat2_2, is_training, 448, 3, strides=(1, 1), padding='SAME', name='2b')
    
    print ("Tensor shape for the up convolution 2 stage: " + str(concat2_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat2_3], print_info=True)))
    
    # up stage 4
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    cnct1_1 = upconv_concat(concat2_3, DB1_skip_connection, is_training, 224, name=1)
    concat1_2 = conv_concat(cnct1_1, is_training, 224, 3, strides=(1, 1), padding='SAME', name='1a')
    concat1_3 = conv_concat(concat1_2, is_training, 224, 3, strides=(1, 1), padding='SAME', name='1b')
    
    print ("Tensor shape for the up convolution 3 stage: " + str(concat1_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat1_3], print_info=True)))
    
    # Flat convolution
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    score = tf.layers.conv2d(concat1_3, N_Class, (1, 1), name='final', activation=None, padding='SAME')
    
    print ("Tensor shape for the flat convolution stage: " + str(score.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([score], print_info=True)))

    return score


if __name__ == '__main__':
    score = ResNet50()