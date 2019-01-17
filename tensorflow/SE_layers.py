import tensorflow as tf
from tflearn.layers.conv import global_avg_pool 
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

#from layers_residual_cardinality import BN_ReLU_Conv_group

reduction_ratio = 16
cardinality = 32 # of splits
depth = 4 # of channels

def Concatenation(layers):
  return tf.concat(layers, 3)

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
	return tf.nn.relu(x)

def Global_Average_Pooling(x, name):
	return global_avg_pool(x, name=name)

def Sigmoid(x):
	return tf.nn.sigmoid(x)

def Fully_connected_Layer(x, output_units, layer_name):
	with tf.name_scope(layer_name):
		return tf.layers.dense(inputs=x, use_bias=False, units=output_units)

def split(input, out_dim, stride, name):
	with tf.name_scope(name) :
		layers_split = list()
		for i in range(cardinality) :
			splits = transform_layer(input, stride=stride, name=name + '_splitN_' + str(i))
			layers_split.append(splits)
		return Concatenation(layers_split)


def transform_layer(x, stride, name):
	with tf.name_scope(name) :
		x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=name+'_conv1')
		#x = Batch_Normalization(x, training=self.training, scope=name+'_batch1')
		x = slim.batch_norm(x,activation_fn=None)
		x = Relu(x)

		x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=name+'_conv2')
		#x = Batch_Normalization(x, training=self.training, scope=name+'_batch2')
		x = slim.batch_norm(x,activation_fn=None)
		x = Relu(x)
	return x

def transition(x, out_dim, name):
	with tf.name_scope(name):
	    x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=name+'_conv1')
	    x = slim.batch_norm(x,activation_fn=None)
	    #x = Batch_Normalization(x, training=self.training, scope=name+'_batch1')
	    # x = Relu(x)

	    return x


def squeeze_excitation_layer(input_v, out_dim, ratio, name):
	with tf.name_scope(name):
		#Squeeze function - used to incorporate global spatial information
		# into a channel descriptor
		name2 = name + 'squeeze'
		squeeze = Global_Average_Pooling(input_v, name2)

		#excitation layer - in order to make use of the squeeze information
		# excitation aims to fully capture channel-wise depedencies 
		name2 = name + 'fully_1'
		Ex = Fully_connected_Layer(squeeze, output_units=(out_dim/ratio), layer_name=name2)
		Ex = tf.nn.relu(Ex)
		name2 = name + 'fully_2'
		Ex = Fully_connected_Layer(Ex, output_units=out_dim, layer_name=name2)

		#Rephape for scale function
		Ex = tf.reshape(Ex, [-1, 1, 1, out_dim])

		scaled_output = input_v * Ex

		return scaled_output

