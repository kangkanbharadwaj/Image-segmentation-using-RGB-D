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


def DPDB_encoder_stacking_decoder_DeepSupervision_300_threeStages_Cardinality_ResidualStack(image, is_training=True, N_Class=0):
    ############################################################
    ### DPDB encoder with deep supervised stacked decoder ######
    ### With three stages                                 ######
    ############################################################
    print ("Number of classes are %d" %(N_Class))
    batch_norm_params = {'is_training': is_training}
    keep_prob = 0.1
    Groups = 1
    #H = self.tgt_image.get_shape()[1].value
    #W = self.tgt_image.get_shape()[2].value
    with tf.variable_scope('dpn92') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        #Variable dpn92        
        k_R=128
        #For group == 48 change
        k_sec=(3, 4, 6, 9, 11)
        #k_sec=(4, 4, 6, 9, 11)
        inc_sec=(16,16,32,24,64)

        print (image)
        
        ##Conv1 
        Conv1 = slim.conv2d(image, 96, [7, 7], stride=1, scope='FirstConv')
        print("Conv1 ")
        print (Conv1)

        ##BLOCK 1 
        blockWidth = 128
        inc = inc_sec[0]
        R = int((k_R*blockWidth)/256) 
        #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
        name='conv1'
        x = DPDB_Block_V2(Conv1, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

        for idx in xrange(k_sec[0]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        
        print(x)
        DB1_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 32, [1, 1], scope='skip1')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 240, keep_prob, name='convDown1')
        
        ##BLOCK 2 
        blockWidth = 256
        inc = inc_sec[1]
        R = int((k_R*blockWidth)/256) 
        #currentC2 = DualPathInterBlock(Pooling_conv1, inc, k_sec[0],  blockWidth, R, name='conv2', residual_type='create')
        name='conv2'
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

        for idx in xrange(k_sec[1]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)

        print("Conv2 ")
        print(x)
        DB2_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 64, [1, 1], scope='skip2')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 384, keep_prob, name='convDown2')
        
        ##BLOCK 3 
        blockWidth = 512
        inc = inc_sec[2]
        R = int((k_R*blockWidth)/256) 
        #currentC3 = DualPathInterBlock(currentC2, inc, k_sec[1],  blockWidth, R, name='conv3', residual_type='down')
        name='conv3'
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

        for idx in xrange(k_sec[2]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)

        print("Conv3 ")
        print(x)
        #DB3_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , blockWidth/2, [1, 1], scope='skip3')        
        DB3_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 92, [1, 1], scope='skip3')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 832, keep_prob, name='convDown3')
        

        ##BLOCK 4 
        blockWidth = 1024
        inc = inc_sec[3]
        R = int((k_R*blockWidth)/256) 
        #currentC4 = DualPathInterBlock(currentC3, inc, k_sec[2],  blockWidth, R, name='conv4', residual_type='down')
        name='conv4'
        x = DPDB_Block_V2(x, R, R, blockWidth, inc, name, _type='create',Groups=Groups)

        for idx in xrange(k_sec[3]):
            name2 = name + str(idx)
            #print(name2)
            x = DPDB_Block_V2(x, R, R, blockWidth, inc, name2, _type='dense',Groups=Groups)
        print("Conv4 ")
        print(x)
        #x_aux=tf.concat((x[0], x[1]), 3) 
        #DB4_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , blockWidth/2, [1, 1], scope='skip4')
        DB4_skip_connection = slim.conv2d(tf.concat((x[0], x[1]), 3) , 128, [1, 1], scope='skip4')        

        x = TransitionDown(tf.concat((x[0], x[1]), 3), 1336, keep_prob, name='convDown4')
        
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
        pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        current_up2 = tf.pad(current_up2, pattern)
        skipConnection2 = Concat_layers(current_up2, DB4_skip_connection)
        # #Features 656 - output half or 328
        current_up2 = Denseblock(skipConnection2, 10, 328, 16, is_training, keep_prob, name='conv8')

        print("current_up2 ")
        print(current_up2)

        current_up3 = TransitionUp_elu(current_up2, 192, name='Upconv3')
        #For Forest
        pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        current_up3 = tf.pad(current_up3, pattern)

        skipConnection3 = Concat_layers(current_up3, DB3_skip_connection)
        current_up3 = Denseblock(skipConnection3, 7, 256, 16, is_training, keep_prob, name='conv9')

        print("current_up3 ")
        print(current_up3)

        current_up4 = TransitionUp_elu(current_up3, 128, name='Upconv4')
        skipConnection4 = Concat_layers(current_up4, DB2_skip_connection)
        current_up4 = Denseblock(skipConnection4, 5, 192, 16, is_training, keep_prob, name='conv10')

        print("current_up4 ")
        print(current_up4)

        current_up5 = TransitionUp_elu(current_up4, 128, name='Upconv5', Final=False)
        #current_up5 = slim.conv2d(current_up4, 128, [1, 1], scope='decoder1_transform')
        #skipConnection5 = Concat_layers(current_up5, DB1_skip_connection)
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
        skipConnection2_decoder2 = Concat_layers(current2, DB4_skip_connection)
        # #Features 656 - output half or 328
        current_up2_decoder2 = Denseblock(skipConnection2_decoder2, 10, 328, 16, is_training, keep_prob, name='conv8_decoder2')

        print("current_up2_decoder2 ")
        print(current_up2_decoder2)

        current_up3_decoder2 = TransitionUp_elu(current_up2_decoder2, 192, name='Upconv3_decoder2')
        #For Forest
        pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        current_up3_decoder2 = tf.pad(current_up3_decoder2, pattern)

        skipConnection3_decoder2 = Concat_layers(current_up3_decoder2, DB3_skip_connection)
        current_up3_decoder2 = Denseblock(skipConnection3_decoder2, 7, 256, 16, is_training, keep_prob, name='conv9_decoder2')

        print("current_up3_decoder2 ")
        print(current_up3_decoder2)

        current_up4_decoder2 = TransitionUp_elu(current_up3_decoder2, 128, name='Upconv4_decoder2')
        skipConnection4_decoder2 = Concat_layers(current_up4_decoder2, DB2_skip_connection)
        current_up4_decoder2 = Denseblock(skipConnection4_decoder2, 5, 192, 16, is_training, keep_prob, name='conv10_decoder2')

        print("current_up4_decoder2 ")
        print(current_up4_decoder2)

        current_up5_decoder2 = TransitionUp_elu(current_up4_decoder2, 128, name='Upconv5_decoder2', Final=False)
        #current_up5_decoder2 = slim.conv2d(current_up4_decoder2, 128, [1, 1], scope='decoder2_transform')
        #skipConnection5 = Concat_layers(current_up5, DB1_skip_connection)
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
        pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
        current_up3_2 = tf.pad(current_up3_2, pattern)
        
        skipConnection3_2 = Concat_layers(current_up3_2, DB3_skip_connection)
        current_up3_2 = Denseblock(skipConnection3_2, 6, 128, 16, is_training, keep_prob, name='Decoder3_conv9')

        print("Decoder 3 block 1 ")
        print(current_up3_2)

        current_up4_2 = TransitionUp_elu(current_up3_2, 64, name='Decoder3_Upconv4')
        skipConnection4_2 = Concat_layers(current_up4_2, DB2_skip_connection)
        current_up4_2 = Denseblock(skipConnection4_2, 4, 92, 16, is_training, keep_prob, name='Decoder3_conv10')

        print("Decoder 3 block 2 ")
        print(current_up4_2)

        current_up5_2 = TransitionUp_elu(current_up4_2, 64, name='Decoder3_Upconv5', Final=False)
        skipConnection5_2 = Concat_layers(current_up5_2, DB1_skip_connection)
        current_up5_2 = Denseblock(skipConnection5_2, 3, 64, 16, is_training, keep_prob, name='Decoder3_conv11')    
        
        print("Decoder 3 block 3 ")
        print(current_up5_2)

        #Residual Stacked connection
        #between Decoder 1 and Decoder 2
        aux_residual_stack_2 = slim.conv2d(current_up5_decoder2, 144, [1, 1], scope='aux_stackerd_residual') #(batchsize, width, height, N_classes)
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
