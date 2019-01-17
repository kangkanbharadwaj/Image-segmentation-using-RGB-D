import tensorflow as tf
from tensorflow.contrib import slim
from layers_slim import *

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


def ResNet50(image, training, N_Class=0):  

    print ("\nResNet loaded")
    print ("\nEncoder on the go!!!")
    print ("\nShape of the input image is: " + str(image.get_shape().as_list()))
    
    # Convolution stage       
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
    
    A_1 = conv2D(image, training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_1')
    A_1_bn = batch_norm(A_1, name='ln_conv1')
    A_1_act = tf.nn.relu(A_1_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_1_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_1_act], print_info=True)))
    
    A_2 = conv2D(A_1_act, training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_2')
    A_2_bn = batch_norm(A_2, name='ln_conv2')
    A_2_act = tf.nn.relu(A_2_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_2_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_2_act], print_info=True)))
    
    A_3 = conv2D(A_2_act, training, filters=64, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_3')
    A_3_bn = batch_norm(A_3, name='ln_conv3')
    A_3_act = tf.nn.relu(A_3_bn)
    
    print ("Tensor shape for the Convolution stage: " + str(A_3_act.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_3_act], print_info=True)))
    
    # Pooling stage
    #print ("\n Entering Pooling stage") 
    
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
        #pool = tf.layers.max_pooling2d(A_1_act, (2, 2), strides=(2, 2), name="pooling")
    
    #print ("\n Tensor shape for the Pooling stage: " + str(pool.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([pool], print_info=True)))

    # Residual Stage 1
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):        
    
    A_2_cb = convolutional_block_basic(A_3_act, training, f=3, filters=[128, 128, 128], stage=2, block='a', s=1)
    A_2_ib1 = identity_block_basic(A_2_cb, training, f=3, filters=[128, 128, 128], stage=2, block='b')
    A_2_ib2 = identity_block_basic(A_2_ib1, training, f=3, filters=[128, 128, 128], stage=2, block='c')    
    
    print ("Tensor shape for the residual stage 1: " + str(A_2_ib2.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_2_ib2], print_info=True)))

    # Residual Stage 2        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"): 
    
    A_3_cb = convolutional_block_basic(A_2_ib2, training, 3, [256, 256, 256], stage=3, block='a', s=2)
    A_3_ib1 = identity_block_basic(A_3_cb, training, 3, [256, 256, 256], stage=3, block='b')
    A_3_ib2 = identity_block_basic(A_3_ib1, training, 3, [256, 256, 256], stage=3, block='c')
    A_3_ib3 = identity_block_basic(A_3_ib2, training, 3, [256, 256, 256], stage=3, block='d')    
    
    print ("Tensor shape for the residual stage 2: " + str(A_3_ib3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_3_ib3], print_info=True)))
    
    # Residual Stage 3        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"): 
    
    A_4_cb = convolutional_block_basic(A_3_ib3, training, 3, [512, 512, 512], stage=4, block='a', s=2)
    A_4_ib1 = identity_block_basic(A_4_cb, training, 3, [512, 512, 512], stage=4, block='b')
    A_4_ib2 = identity_block_basic(A_4_ib1, training, 3, [512, 512, 512], stage=4, block='c')
    A_4_ib3 = identity_block_basic(A_4_ib2, training, 3, [512, 512, 512], stage=4, block='d')
    A_4_ib4 = identity_block_basic(A_4_ib3, training, 3, [512, 512, 512], stage=4, block='e')
    A_4_ib5 = identity_block_basic(A_4_ib4, training, 3, [512, 512, 512], stage=4, block='f')    
    
    print ("Tensor shape for the residual stage 3: " + str(A_4_ib5.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_4_ib5], print_info=True)))
    
    # Residual Stage 4        
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
    
    A_5_cb = convolutional_block_basic(A_4_ib5, training, 3, [1024,1024, 1024], stage=5, block='a', s=2)
    A_5_ib1 = identity_block_basic(A_5_cb, training, 3, [1024,1024, 1024], stage=5, block='b')
    A_5_ib2 = identity_block_basic(A_5_ib1, training, 3, [1024,1024, 1024], stage=5, block='c')
    #A_5_ib3 = identity_block(A_5_ib2, 3, [512, 512,2048], stage=5, block='d')
    #A_5_ib4 = identity_block(A_5_ib3, 3, [512, 512, 1024], stage=5, block='e')
    #A_5_ib5 = identity_block(A_5_ib4, 3, [512, 512, 1024], stage=5, block='f')
        
    print ("Tensor shape for the residual stage 4: " + str(A_5_ib2.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_5_ib2], print_info=True)))
    
    # Stage 5         
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"):
    
    #A_6_cb = convolutional_block(A_5_ib5, 3, [1024, 1024, 2048], stage=6, block='a', s=2)
    #A_6_ib1 = identity_block(A_6_cb, 3, [1024, 1024, 2048], stage=6, block='b')
    #A_6_ib2 = identity_block(A_6_ib1, 3, [1024, 1024, 2048], stage=6, block='c')
    #A_6_ib3 = identity_block(A_6_ib2, 3, [1024, 1024, 2048], stage=6, block='d')
    #A_6_ib4 = identity_block(A_6_ib3, 3, [1024, 1024, 2048], stage=6, block='e')
    #A_6_ib5 = identity_block(A_6_ib4, 3, [1024, 1024, 2048], stage=6, block='f')
    #A_6_ib6 = identity_block(A_6_ib5, 3, [1024, 1024, 2048], stage=6, block='g')
        
    #print ("Tensor shape for the residual stage 5: " + str(A_6_ib6.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([A_6_ib6], print_info=True)))
    
    # Up scaling image to original resolution        
    print ("\nDecoder on the go!!!")
    
    # up stage 1     
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    #cnct4_1 = upconv_concat(A_6_ib6, A_5_ib5, 1024, name=4)
    #concat4_2 = conv_concat(cnct4_1, 1024, 3, strides=(1, 1), padding='SAME', name='4a')
    #concat4_3 = conv_concat(concat4_2, 1024, 3, strides=(1, 1), padding='SAME', name='4b')
    
    #print ("Tensor shape for the 4th stage: " + str(concat4_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat4_3], print_info=True)))
    
    # up stage 2    
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    cnct3_1 = upconv_concat(A_5_ib2, A_4_ib5, training, 512, name=3)
    concat3_2 = conv_concat(cnct3_1, training, 512, 3, strides=(1, 1), padding='SAME', name='3a')
    concat3_3 = conv_concat(concat3_2, training, 512, 3, strides=(1, 1), padding='SAME', name='3b')
    
    print ("Tensor shape for the up convolution 1 stage: " + str(concat3_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat3_3], print_info=True)))

    # up stage 3
    #print ("\nEntering up_convolve stage 2")
    
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    cnct2_1 = upconv_concat(concat3_3, A_3_ib3, training, 256, name=2)
    concat2_2 = conv_concat(cnct2_1, training, 256, 3, strides=(1, 1), padding='SAME', name='2a')
    concat2_3 = conv_concat(concat2_2, training, 256, 3, strides=(1, 1), padding='SAME', name='2b')
    
    print ("Tensor shape for the up convolution 2 stage: " + str(concat2_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat2_3], print_info=True)))
    
    # up stage 4
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    cnct1_1 = upconv_concat(concat2_3, A_2_ib2, training, 128, name=1)
    concat1_2 = conv_concat(cnct1_1, training, 128, 3, strides=(1, 1), padding='SAME', name='1a')
    concat1_3 = conv_concat(concat1_2, training, 128, 3, strides=(1, 1), padding='SAME', name='1b')
    
    print ("Tensor shape for the up convolution 3 stage: " + str(concat1_3.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([concat1_3], print_info=True)))
    
    # Flat convolution
    #with tf.device("/job:localhost/replica:0/task:0/device:GPU:1"): 
    
    score = tf.layers.conv2d(concat1_3, N_Class, (1, 1), name='final', activation=None, padding='SAME')
    
    print ("Tensor shape for the flat convolution stage: " + str(score.get_shape().as_list()) +"\t\t" + "Tensor size:" + str(slim.model_analyzer.analyze_vars([score], print_info=True)))

    return score


if __name__ == '__main__':
    score = ResNet50()