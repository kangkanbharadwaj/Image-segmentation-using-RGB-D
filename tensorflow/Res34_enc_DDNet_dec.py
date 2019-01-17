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
def conv2D(input_, is_training, filters, k_size, strides, padding, name): 
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


def _batch_norm(X, is_training, name):
    bn = tf.cond(is_training, lambda: tf.contrib.layers.batch_norm(X, decay=0.9, is_is_training=True, center=True, scale=True, activation_fn=None, updates_collections=None, scope=name), 
                 lambda: tf.contrib.layers.batch_norm(X, decay=0.9, is_is_training=False, center=True, scale=True, activation_fn=None, updates_collections=None, scope=name, reuse=True))
    return bn
    
    
def batch_norm(X, name):
    bn = slim.batch_norm(X, scope=name)
    return bn    
    
# Used for a mini batch size = 1
def layer_norm(X, name):
        ln = tf.contrib.layers.batch_norm(X, name)
        return ln

# Implementing a ResNet bottleneck identity block with shortcut path passing over 3 Conv Layers
def identity_block(X, is_training, f, filters, stage, block):
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

    A1 = conv2D(X, is_training, filters=l1_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, is_training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)

    A3 = conv2D(A2_act, is_training, filters=l3_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2c')
    A3_bn= batch_norm(A3, name=bn_name+'2c')

    A3_add = tf.add(A3_bn, X)
    A = tf.nn.relu(A3_add)
    return A


def identity_block_basic(X, is_training, f, filters, stage, block):
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

    A1 = conv2D(X, is_training, filters=l1_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, is_training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')

    A2_add = tf.add(A2_bn, X)
    A = tf.nn.relu(A2_add)
    return A

# Implementing a ResNet basic convolution block with shortcut path passing over 3 Conv Layers having different sizes
def convolutional_block(X, is_training, f, filters, stage, block, s=2):
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

    A1 = conv2D(X, is_training, filters=l1_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, is_training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)

    A3 = conv2D(A2_act, is_training, filters=l3_f, k_size=(1, 1), strides=(1, 1), padding='VALID', name=conv_name+'2c')
    A3_bn=batch_norm(A3, name=bn_name+'2c')

    A_ = conv2D(X, is_training, filters=l3_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A3_add = tf.add(A3_bn, A_bn_)
    A = tf.nn.relu(A3_add)
    return A


# Implementing a ResNet convolutional basic block with shortcut path passing over 3 Conv Layers having different sizes
def convolutional_block_basic(X, is_training, f, filters, stage, block, s=2):
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

    A1 = conv2D(X, is_training, filters=l1_f, k_size=(f, f), strides=(s, s), padding='SAME', name=conv_name+'2a')
    A1_bn = batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)

    A2 = conv2D(A1_act, is_training, filters=l2_f, k_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name+'2b')
    A2_bn = batch_norm(A2, name=bn_name+'2b')

    A_ = conv2D(X, is_training, filters=l3_f, k_size=(1, 1), strides=(s, s), padding='VALID', name=conv_name+'1')
    A_bn_ = batch_norm(A_, name=bn_name+'1')

    A2_add = tf.add(A2_bn, A_bn_)
    A = tf.nn.relu(A2_add)
    return A

# Transpose convolution / De convolution to get scale up resolution
def upconv_2D(tensor, is_training, n_filter, name):
    """
    @params
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(tensor, filters=n_filter, kernel_size=(2, 2), strides=(2, 2), padding='valid', name="upsample_{}".format(name))

# Upsample `inputA` and concat with `input_B`
def upconv_concat(inputA, input_B, is_training, n_filter, name):
    """
    @params
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    
    up_conv = upconv_2D(inputA, is_training, n_filter, name)    
    _bn = batch_norm(up_conv, name="bn_upconv_{}".format(name))
    _act = tf.nn.relu(_bn)
    _concat = tf.concat([_act, input_B], axis=-1, name="up_concat_{}".format(name))
    
    return _concat

# Convolve `inputA` and concat with previous layer
def conv_concat(inputA, is_training, filters, k_size, strides, padding, name):
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
    _conv = conv2D(inputA, is_training, filters, k_size, strides, padding, name="convolution_{}".format(name))
    _bn = batch_norm(_conv, name="bn_conv_{}".format(name))
    _act = tf.nn.relu(_bn)
    _concat = tf.concat([_act, inputA], axis=-1, name="concat_{}".format(name))
    return _concat


def Res34_encoder_DDNet_decoder(image, is_training, N_Class=0):  
    
    batch_norm_params = {'is_training': is_training}
    keep_prob = 0.1
    Groups = 1

    print ("\nResNet loaded")
    print ("\nEncoder on the go!!!")
    print ("\nShape of the input image is: " + str(image.get_shape().as_list()))
    
    # Convolution stage       
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
    
    A_1 = conv2D(image, is_training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_1')
    A_1_bn = batch_norm(A_1, name='ln_conv1')
    A_1_act = tf.nn.relu(A_1_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_1_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_1_act], print_info=True)))
    
    A_2 = conv2D(A_1_act, is_training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_2')
    A_2_bn = batch_norm(A_2, name='ln_conv2')
    A_2_act = tf.nn.relu(A_2_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_2_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_2_act], print_info=True)))
    
    A_3 = conv2D(A_2_act, is_training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_3')
    A_3_bn = batch_norm(A_3, name='ln_conv3')
    A_3_act = tf.nn.relu(A_3_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_3_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_3_act], print_info=True)))
    
    A_2_cb = convolutional_block_basic(A_3_act, is_training, f=3, filters=[128, 128, 128], stage=2, block='a', s=1)
    A_2_ib1 = identity_block_basic(A_2_cb, is_training, f=3, filters=[128, 128, 128], stage=2, block='b')
    A_2_ib2 = identity_block_basic(A_2_ib1, is_training, f=3, filters=[128, 128, 128], stage=2, block='c')    
    
    print ("Tensor shape for the residual stage 1: " + str(A_2_ib2.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_2_ib2], print_info=True)))

    # Residual Stage 2        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"): 
    
    A_3_cb = convolutional_block_basic(A_2_ib2, is_training, 3, [256, 256, 256], stage=3, block='a', s=2)
    A_3_ib1 = identity_block_basic(A_3_cb, is_training, 3, [256, 256, 256], stage=3, block='b')
    A_3_ib2 = identity_block_basic(A_3_ib1, is_training, 3, [256, 256, 256], stage=3, block='c')
    A_3_ib3 = identity_block_basic(A_3_ib2, is_training, 3, [256, 256, 256], stage=3, block='d')    
    
    print ("Tensor shape for the residual stage 2: " + str(A_3_ib3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_3_ib3], print_info=True)))
    
    # Residual Stage 3        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"): 
    
    A_4_cb = convolutional_block_basic(A_3_ib3, is_training, 3, [512, 512, 512], stage=4, block='a', s=2)
    A_4_ib1 = identity_block_basic(A_4_cb, is_training, 3, [512, 512, 512], stage=4, block='b')
    A_4_ib2 = identity_block_basic(A_4_ib1, is_training, 3, [512, 512, 512], stage=4, block='c')
    A_4_ib3 = identity_block_basic(A_4_ib2, is_training, 3, [512, 512, 512], stage=4, block='d')
    A_4_ib4 = identity_block_basic(A_4_ib3, is_training, 3, [512, 512, 512], stage=4, block='e')
    A_4_ib5 = identity_block_basic(A_4_ib4, is_training, 3, [512, 512, 512], stage=4, block='f')    
    
    print ("Tensor shape for the residual stage 3: " + str(A_4_ib5.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_4_ib5], print_info=True)))
    
    # Residual Stage 4        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
    
    A_5_cb = convolutional_block_basic(A_4_ib5, is_training, 3, [1024,1024, 1024], stage=5, block='a', s=2)
    A_5_ib1 = identity_block_basic(A_5_cb, is_training, 3, [1024,1024, 1024], stage=5, block='b')
    A_5_ib2 = identity_block_basic(A_5_ib1, is_training, 3, [1024,1024, 1024], stage=5, block='c')
    #A_5_ib3 = identity_block(A_5_ib2, 3, [512, 512,2048], stage=5, block='d')
    #A_5_ib4 = identity_block(A_5_ib3, 3, [512, 512, 1024], stage=5, block='e')
    #A_5_ib5 = identity_block(A_5_ib4, 3, [512, 512, 1024], stage=5, block='f')
        
    print ("Tensor shape for the residual stage 4: " + str(A_5_ib2.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_5_ib2], print_info=True)))
    
    # Up scaling image to original resolution        
    print ("\nDecoder on the go!!!")
    
    x = tf.layers.max_pooling2d(A_5_ib2, (2, 2), strides=(2, 2), name="pooling")
    
    
    ######################################################################################
    ######################################### DECODER 1 #############################################

    current_up2 = TransitionUp_elu(x, 240, name='Upconv2')
    #For Camvid
    #pattern = [[0, 0], [1, 0], [0, 0], [0, 0]]
    #current_up2 = tf.pad(current_up2, pattern)
    #For ForestV2
    # pattern = [[0, 0], [0, 0], [1, 0], [0, 0]]
    # current_up2 = tf.pad(current_up2, pattern)
    #For Forest
    #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    #current_up2 = tf.pad(current_up2, pattern)
    skipConnection2 = Concat_layers(current_up2, A_5_ib2)
    # #Features 656 - output half or 328
    current_up2 = Denseblock(skipConnection2, 10, 328, 16, is_training, keep_prob, name='conv8')

    print("current_up2 ")
    print(current_up2)

    current_up3 = TransitionUp_elu(current_up2, 192, name='Upconv3')
    #For Forest
    #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    #current_up3 = tf.pad(current_up3, pattern)

    skipConnection3 = Concat_layers(current_up3, A_4_ib5)
    current_up3 = Denseblock(skipConnection3, 7, 256, 16, is_training, keep_prob, name='conv9')

    print("current_up3 ")
    print(current_up3)

    current_up4 = TransitionUp_elu(current_up3, 128, name='Upconv4')
    skipConnection4 = Concat_layers(current_up4, A_3_ib3)
    current_up4 = Denseblock(skipConnection4, 5, 192, 16, is_training, keep_prob, name='conv10')

    print("current_up4 ")
    print(current_up4)

    current_up5 = TransitionUp_elu(current_up4, 128, name='Upconv5', Final=False)
    #current_up5 = slim.conv2d(current_up4, 128, [1, 1], scope='decoder1_transform')
    #skipConnection5 = Concat_layers(current_up5, A_2_ib2)
    #current_up5 = Denseblock(skipConnection5, 4, 128, 16, is_training, keep_prob, name='conv11') 
    print("current_up5 ")
    print(current_up5)
    #For finetunned 
    End_maps_decoder1 = slim.conv2d(current_up5, N_Class, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
    #End_maps_decoder1 = slim.conv2d(current_up5, 20, [1, 1], scope='First_Final_decoder') #(batchsize, width, height, N_classes)
    #For forest 
    #End_maps_decoder1 = slim.conv2d(current_up5, 20, [1, 1], scope='Final_decoder_F') #(batchsize, width, height, N_classes)
    Reshaped_map_decoder1 = tf.reshape(End_maps_decoder1, (-1, N_Class))

    print("End map size Decoder 1: ")
    print(Reshaped_map_decoder1)
            
    ######################################### Encoder 1 #############################################

    current2 = Denseblock(current_up5, 5, 208, 16, is_training, keep_prob, name='encoder1_conv1') # 7 * 16 = 112 + 208 = 320
    #DB3_skip = slim.conv2d(current, 64, [1, 1], scope='encoder1_skip1')
    current2 = TransitionDown(current2, 192, keep_prob, name='encoder1_convDown1')
    current_up4_skip_connection = slim.conv2d(current_up4 , 192, [1, 1], scope='skip4_decoder')
    current2 = tf.add(current2, current_up4_skip_connection)

    print("Encoder 1 block 1  ")
    print(current2)

    current2 = Denseblock(current2, 7, 304, 16, is_training, keep_prob, name='encoder1_conv2') # 10 * 16 = 160 + 320 = 480
    #DB4_skip = slim.conv2d(current, 92, [1, 1], scope='encoder1_skip2')
    current2 = TransitionDown(current2, 256, keep_prob, name='encoder1_convDown2')
    current_up3_skip_connection = slim.conv2d(current_up3 , 256, [1, 1], scope='skip3_decoder')

    current2 = tf.add(current2, current_up3_skip_connection)

    print("Encoder 1 block 2  ")
    print(current2)

    current2 = Denseblock(current2, 10, 416, 16, is_training, keep_prob, name='encoder1_conv3') # 12 * 16 = 192 + 480 = 672
    #DB5_skip = slim.conv2d(current, 128, [1, 1], scope='encoder1_skip3')        
    current2 = TransitionDown(current2, 328, keep_prob, name='encoder1_convDown3')
    current_up2_skip_connection = slim.conv2d(current_up2 , 328, [1, 1], scope='skip2_decoder')

    current2 = tf.add(current2, current_up2_skip_connection)

    print("Encoder 1 block 3  ")
    print(current2)
    
        ######################################################################################
    ######################################### DECODER 2 #############################################

    #current_up2_decoder2 = TransitionUp_elu(current5, 240, name='Upconv2_decoder2')
    #For Camvid
    #pattern = [[0, 0], [1, 0], [0, 0], [0, 0]]
    #current_up2 = tf.pad(current_up2, pattern)
    #For ForestV2
    # pattern = [[0, 0], [0, 0], [1, 0], [0, 0]]
    # current_up2 = tf.pad(current_up2, pattern)
    #For Forest
    #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    #current_up2_decoder2 = tf.pad(current_up2_decoder2, pattern)
    skipConnection2_decoder2 = Concat_layers(current2, A_5_ib2)
    # #Features 656 - output half or 328
    current_up2_decoder2 = Denseblock(skipConnection2_decoder2, 10, 328, 16, is_training, keep_prob, name='conv8_decoder2')

    print("current_up2_decoder2 ")
    print(current_up2_decoder2)

    current_up3_decoder2 = TransitionUp_elu(current_up2_decoder2, 192, name='Upconv3_decoder2')
    #For Forest
    #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    #current_up3_decoder2 = tf.pad(current_up3_decoder2, pattern)

    skipConnection3_decoder2 = Concat_layers(current_up3_decoder2, A_4_ib5)
    current_up3_decoder2 = Denseblock(skipConnection3_decoder2, 7, 256, 16, is_training, keep_prob, name='conv9_decoder2')

    print("current_up3_decoder2 ")
    print(current_up3_decoder2)

    current_up4_decoder2 = TransitionUp_elu(current_up3_decoder2, 128, name='Upconv4_decoder2')
    skipConnection4_decoder2 = Concat_layers(current_up4_decoder2, A_3_ib3)
    current_up4_decoder2 = Denseblock(skipConnection4_decoder2, 5, 192, 16, is_training, keep_prob, name='conv10_decoder2')

    print("current_up4_decoder2 ")
    print(current_up4_decoder2)

    current_up5_decoder2 = TransitionUp_elu(current_up4_decoder2, 128, name='Upconv5_decoder2', Final=False)
    #current_up5_decoder2 = slim.conv2d(current_up4_decoder2, 128, [1, 1], scope='decoder2_transform')
    #skipConnection5 = Concat_layers(current_up5, A_2_ib2)
    #current_up5 = Denseblock(skipConnection5, 4, 128, 16, is_training, keep_prob, name='conv11') 
    print("current_up5_decoder2 ")
    print(current_up5_decoder2)

    #Residual Stacked connection
    #between Decoder 1 and Decoder 2
    current_up5_decoder2 = tf.add(current_up5_decoder2, current_up5)
    print("First residual connection")

    End_maps_decoder2 = slim.conv2d(current_up5_decoder2, N_Class, [1, 1], scope='Final_decoder_decoder2') #(batchsize, width, height, N_classes)
    #End_maps_decoder2 = slim.conv2d(current_up5_decoder2, 20, [1, 1], scope='First_Final_decoder_decoder2') #(batchsize, width, height, N_classes)
    #For forest and maybe no weight balance
    #End_maps_decoder2 = slim.conv2d(current_up5_decoder2, 20, [1, 1], scope='Final_decoder_decoder2_F') #(batchsize, width, height, N_classes)
    
    Reshaped_map_decoder2 = tf.reshape(End_maps_decoder2, (-1, N_Class))

    print("End map size _decoder2: ")
    print(Reshaped_map_decoder2)
            
    ######################################### Encoder 2 #############################################

    current3 = Denseblock(current_up5_decoder2, 5, 208, 16, is_training, keep_prob, name='encoder2_conv1') # 7 * 16 = 112 + 208 = 320
    #DB3_skip = slim.conv2d(current, 64, [1, 1], scope='encoder1_skip1')
    #current3 = slim.conv2d(current3, 192, [1, 1], scope='encoder2_T1')
    current3 = TransitionDown(current3, 192, keep_prob, name='encoder2_convDown1')
    current_up4_skip_connection_decoder2 = slim.conv2d(current_up4_decoder2 , 192, [1, 1], scope='skip4_decoder2')
    current3 = tf.add(current3, current_up4_skip_connection_decoder2)

    print("Encoder 2 block 1  ")
    print(current3)

    current3 = Denseblock(current3, 7, 304, 16, is_training, keep_prob, name='encoder2_conv2') # 10 * 16 = 160 + 320 = 480
    #DB4_skip = slim.conv2d(current, 92, [1, 1], scope='encoder1_skip2')
    current3 = TransitionDown(current3, 256, keep_prob, name='encoder2_convDown2')
    current_up3_skip_connection_decoder2 = slim.conv2d(current_up3_decoder2, 256, [1, 1], scope='skip3_decoder2')

    current3 = tf.add(current3, current_up3_skip_connection_decoder2)

    print("Encoder 2 block 2  ")
    print(current3)

    current3 = Denseblock(current3, 10, 416, 16, is_training, keep_prob, name='encoder2_conv3') # 12 * 16 = 192 + 480 = 672
    #DB5_skip = slim.conv2d(current, 128, [1, 1], scope='encoder1_skip3')        
    current3 = TransitionDown(current3, 328, keep_prob, name='encoder2_convDown3')
    current_up2_skip_connection_decoder2 = slim.conv2d(current_up2_decoder2, 328, [1, 1], scope='skip2_decoder2')

    current3 = tf.add(current3, current_up2_skip_connection_decoder2)

    #current3 = TransitionDown(current3, 328, keep_prob, name='encoder2_convDown4')

    print("Encoder 2 block 3  ")
    print(current3)
    
    ######################################### Decoder 3 #############################################

    current_up3_2 = TransitionUp_elu(current3, 128, name='Decoder3_Upconv3')
    #For Forest
    #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
    #current_up3_2 = tf.pad(current_up3_2, pattern)
    
    skipConnection3_2 = Concat_layers(current_up3_2, A_4_ib5)
    current_up3_2 = Denseblock(skipConnection3_2, 6, 128, 16, is_training, keep_prob, name='Decoder3_conv9')

    print("Decoder 3 block 1 ")
    print(current_up3_2)

    current_up4_2 = TransitionUp_elu(current_up3_2, 64, name='Decoder3_Upconv4')
    skipConnection4_2 = Concat_layers(current_up4_2, A_3_ib3)
    current_up4_2 = Denseblock(skipConnection4_2, 4, 92, 16, is_training, keep_prob, name='Decoder3_conv10')

    print("Decoder 3 block 2 ")
    print(current_up4_2)

    current_up5_2 = TransitionUp_elu(current_up4_2, 64, name='Decoder3_Upconv5', Final=False)
    skipConnection5_2 = Concat_layers(current_up5_2, A_2_ib2)
    current_up5_2 = Denseblock(skipConnection5_2, 3, 64, 16, is_training, keep_prob, name='Decoder3_conv11')    
    
    print("Decoder 3 block 3 ")
    print(current_up5_2)

    #Residual Stacked connection
    #between Decoder 1 and Decoder 2
    aux_residual_stack_2 = slim.conv2d(current_up5_decoder2, 240, [1, 1], scope='aux_stackerd_residual') #(batchsize, width, height, N_classes)
    current_up5_2 = tf.add(current_up5_2, aux_residual_stack_2)
    print("Second residual connection")

    #pattern = [[0, 0], [6, 6], [6, 6], [0, 0]]
    #current_up5 = tf.pad(current_up5, pattern)
    End_maps = slim.conv2d(current_up5_2, N_Class, [1, 1], scope='last_F') #(batchsize, width, height, N_classes)
    #End_maps = slim.conv2d(current_up5_2, 20, [1, 1], scope='last') #(batchsize, width, height, N_classes)
    #Reshaped_map = tf.reshape(End_maps, (-1, 20))
    print("End map size ")
    print(End_maps)        
    return Reshaped_map_decoder1, Reshaped_map_decoder2, End_maps