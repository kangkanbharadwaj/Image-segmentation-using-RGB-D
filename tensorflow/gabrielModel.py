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


def D_Net(image, is_training=True):

    batch_norm_params = {'is_training': is_training}
    keep_prob = 0.1
    H = image.get_shape()[1].value
    W = image.get_shape()[2].value
    with tf.variable_scope('D_Net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        #print("input", tgt_image)
        
        ##NET 
        current = slim.conv2d(image, 48, [3, 3], scope='FirstConv')
        print ("\n", str(current.get_shape().as_list()))

        current = Denseblock(current, 4, 112, 16, is_training,keep_prob, name='conv1') # 4*16 = 64 + 48 = 112
        print ("\n", str(current.get_shape().as_list()))
        current = TransitionDown(current, 112, keep_prob, name='convDown1')
        print ("\n", str(current.get_shape().as_list()))

        current = Denseblock(current, 5, 192, 16, is_training, keep_prob, name='conv2')
        print ("\n", str(current.get_shape().as_list()))# 5*16 = 80 + 112 = 192
        current = TransitionDown(current, 192, keep_prob, name='convDown2')
        print ("\n", str(current.get_shape().as_list()))

        current = Denseblock(current, 7, 304, 16, is_training, keep_prob, name='conv3') # 7 * 16 = 112 + 192 = 304
        print ("\n", str(current.get_shape().as_list()))
        current = TransitionDown(current, 304, keep_prob, name='convDown3')
        print ("\n", str(current.get_shape().as_list()))

        current = Denseblock(current, 10, 464, 16, is_training, keep_prob, name='conv4') # 10 * 16 = 160 + 304 = 464
        print ("\n", str(current.get_shape().as_list()))
        current = TransitionDown(current, 464, keep_prob, name='convDown4')
        print ("\n", str(current.get_shape().as_list()))

        current = Denseblock(current, 12, 656, 16, is_training, keep_prob, name='conv5') # 12 * 16 = 192 + 464 = 656
        print ("\n", str(current.get_shape().as_list()))
        current = TransitionDown(current, 656, keep_prob, name='convDown5')
        print ("\n", str(current.get_shape().as_list()))

        ###BottoNeck layer 
        current = Denseblock(current, 15, 896, 16, is_training, keep_prob, name='conv6') # 15 * 16 = 192 + 656 = 896
        print ("\n", str(current.get_shape().as_list()))


        #Upsampling part 
        current_up1 = TransitionUp(current, 1088, name='Upconv1')
        print ("\n", str(current_up1.get_shape().as_list()))
        current_up1 = Denseblock(current_up1, 12, 1088, 16, is_training, keep_prob, name='conv7')
        print ("\n", str(current_up1.get_shape().as_list()))

        current_up2 = TransitionUp(current_up1, 512, name='Upconv2')
        print ("\n", str(current_up2.get_shape().as_list()))
        current_up2 = Denseblock(current_up2, 10, 512, 16, is_training, keep_prob, name='conv8')
        print ("\n", str(current_up2.get_shape().as_list()))

        current_up3 = TransitionUp(current_up2, 384, name='Upconv3')
        print ("\n", str(current_up3.get_shape().as_list()))
        current_up3 = Denseblock(current_up3, 7, 384, 16, is_training, keep_prob, name='conv9')
        print ("\n", str(current_up3.get_shape().as_list()))

        current_up4 = TransitionUp(current_up3, 256, name='Upconv4')
        print ("\n", str(current_up4.get_shape().as_list()))
        current_up4 = Denseblock(current_up4, 5, 256, 16, is_training, keep_prob, name='conv10')
        print ("\n", str(current_up4.get_shape().as_list()))

        current_up5 = TransitionUp(current_up4, 128, name='Upconv5', Final=False)
        print ("\n", str(current_up5.get_shape().as_list()))
        current_up5 = Denseblock(current_up5, 4, 128, 16, is_training, keep_prob, name='conv11')    
        print ("\n", str(current_up5.get_shape().as_list()))
        
        #pattern = [[0, 0], [6, 6], [6, 6], [0, 0]]
        #current_up5 = tf.pad(current_up5, pattern)
        print ("\n", str(current_up5.get_shape().as_list()))
        End_maps = slim.conv2d(current_up5, 20, [1, 1], scope='last') #(batchsize, width, height, N_classes)
        
        #reshape to be [batch_size*width*height, N_classes]
        #Reshaped_map = tf.reshape(End_maps, [batch_size*width*height, N_classes])
        #Reshaped_map = tf.reshape(End_maps, (-1, 20))
        #print("End map size ")
        #print("Reshaped_map", str(End_maps.get_shape().as_list()))
        return End_maps, image


def DPDB_Net(image, is_training=True):
    #####################################
    ### New dual path block Network ######
    #####################################

    batch_norm_params = {'is_training': is_training}
    keep_prob = 0.1
    #H = tgt_image.get_shape()[1].value
    #W = tgt_image.get_shape()[2].value
    with tf.variable_scope('dpn92') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        #Variable dpn92        
        k_R=96
        k_sec=(3, 4, 5, 6, 8)
        inc_sec=(16,16,32,24,64)

        #print("input", tgt_image)
        
        ##Conv1 
        Conv1 = slim.conv2d(image, 96, [7, 7], stride=1, scope='FirstConv')
        #BN_conv1 = slim.batch_norm(Conv1,activation_fn=None)
        #RELU_conv1 = tf.nn.relu(BN_conv1)
        #Pooling_conv1 = slim.max_pool2d(RELU_conv1, [3, 3], stride=2, scope='PoolConv1', padding='SAME')
        #print("Conv1 ")
        #print(Conv1)

        ##BLOCK 1 
        blockWidth = 64
        inc = inc_sec[0]
        R = 96 #int((k_R*blockWidth)/256) 
        #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
        name='conv1'
        x = DPDB_Block(Conv1, R, R, blockWidth, inc, name, _type='create', dropout=keep_prob)

        for idx in range(k_sec[0]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block(x, R, R, blockWidth, inc, name2, _type='dense', dropout=keep_prob)

        #print("Conv1 ")
        #print(x)
        DB1_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 64, [1, 1], scope='skip1')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 224, keep_prob, name='convDown1')

        ##BLOCK 2 
        blockWidth = 192
        inc =  inc_sec[1]
        R =  128 #int((k_R*blockWidth)/256) 
        #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
        name='conv2'
        x = DPDB_Block(x, R, R, blockWidth, inc, name, _type='create', dropout=keep_prob)

        for idx in range(k_sec[1]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block(x, R, R, blockWidth, inc, name2, _type='dense', dropout=keep_prob)

        #print("Conv2 ")
        #print(x)
        DB2_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 112, [1, 1], scope='skip2')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 368, keep_prob, name='convDown2')


        ##BLOCK 3 
        blockWidth = 384
        inc =  inc_sec[2]
        R = 256 #int((k_R*blockWidth)/256) 
        #currentC3 = DualPathInterBlock(currentC2, inc, k_sec[1],  blockWidth, R, name='conv3', residual_type='down')
        name='conv3'
        x = DPDB_Block(x, R, R, blockWidth, inc, name, _type='create', dropout=keep_prob)

        for idx in range(k_sec[2]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block(x, R, R, blockWidth, inc, name2, _type='dense', dropout=keep_prob)

        #print("Conv3 ")
        #print(x)
        DB3_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 200, [1, 1], scope='skip3')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 672, keep_prob, name='convDown3')

        ##BLOCK 4 
        blockWidth = 512
        inc = inc_sec[3]
        R =  384#int((k_R*blockWidth)/256) 
        #currentC4 = DualPathInterBlock(currentC3, inc, k_sec[2],  blockWidth, R, name='conv4', residual_type='down')
        name='conv4'
        x = DPDB_Block(x, R, R, blockWidth, inc, name, _type='create', dropout=keep_prob)

        for idx in range(k_sec[3]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block(x, R, R, blockWidth, inc, name2, _type='dense', dropout=keep_prob)
        #print("Conv4 ")
        #print(x)
        #x_aux=tf.concat((x[0], x[1]), 3) 
        DB4_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 240, [1, 1], scope='skip4')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 800, keep_prob, name='convDown4')

        
        ##BLOCK 5
        blockWidth = 768
        inc = inc_sec[4]
        R = 512 #int((k_R*blockWidth)/256) 
        #currentC5 = DualPathInterBlock(currentC4, inc, k_sec[3],  blockWidth, R, name='conv5', residual_type='down')
        name='conv5'
        x = DPDB_Block(x, R, R, blockWidth, inc, name, _type='create', dropout=keep_prob)

        for idx in range(k_sec[4]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block(x, R, R, blockWidth, inc, name2, _type='dense', dropout=keep_prob)
        
        #print("Conv5 ")
        #print(x)

        x = tf.concat((x[0], x[1]), 3)
        print("End contraction side")
        #print(x) 

        DB5_skip_connection = slim.conv2d(x, 500, [1, 1], scope='skip5')      

        x = TransitionDown(x, 500, keep_prob, name='convDown5')
        print("End contraction side DOWN ")
        #print(x) 

        ##################################################################################################
        ###############################################################################################
        ##################################################################################################
        ##################################################################################################
        ###############################################################################################
        ##################################################################################################
        #Upsampling part 
        current_up1 = TransitionUp_elu(x, 300, name='Upconv1')
        #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        #current_up1 = tf.pad(current_up1, pattern)
        skipConnection1 = Concat_layers(current_up1, DB5_skip_connection)
        current_up1 = Denseblock(skipConnection1, 3, 348, 16, is_training, keep_prob, name='conv7')

        current_up2 = TransitionUp_elu(current_up1, 200, name='Upconv2')
        #pattern = [[0, 0], [1, 0], [0, 0], [0, 0]]
        #current_up2 = tf.pad(current_up2, pattern)
        skipConnection2 = Concat_layers(current_up2, DB4_skip_connection)
        current_up2 = Denseblock(skipConnection2, 10, 350, 16, is_training, keep_prob, name='conv8')

        current_up3 = TransitionUp_elu(current_up2, 128, name='Upconv3')
        #pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        #current_up3 = tf.pad(current_up3, pattern)
        skipConnection3 = Concat_layers(current_up3, DB3_skip_connection)
        current_up3 = Denseblock(skipConnection3, 7, 268, 16, is_training, keep_prob, name='conv9')

        current_up4 = TransitionUp_elu(current_up3, 92, name='Upconv4')
        skipConnection4 = Concat_layers(current_up4, DB2_skip_connection)
        current_up4 = Denseblock(skipConnection4, 4, 156, 16, is_training, keep_prob, name='conv10')

        current_up5 = TransitionUp_elu(current_up4, 64, name='Upconv5', Final=False)
        skipConnection5 = Concat_layers(current_up5, DB1_skip_connection)
        current_up5 = Denseblock(skipConnection5, 3, 128, 16, is_training, keep_prob, name='conv11')    
        
        #pattern = [[0, 0], [6, 6], [6, 6], [0, 0]]
        #current_up5 = tf.pad(current_up5, pattern)
        End_maps = slim.conv2d(current_up5, 20, [1, 1], scope='last') #(batchsize, width, height, N_classes)
        
        #Reshaped_map = tf.reshape(End_maps, (-1, 20))
        #print("End map size ")
        #print(Reshaped_map)
        
        return End_maps, image