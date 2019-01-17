import tensorflow as tf

# Compute the cross entropy loss 
def cross_entropy_loss(prediction, labels, num_classes, label_balance):
    
    """Calculate the loss from the prediction and the labels.

    Args:
      prediction: tensor, float - [batch_size*width*height, num_classes].
       
      labels: Labels tensor, int32 - [batch_size*width*height, num_classes]
          The ground truth of your data.
      label_balance: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):

        epsilon = tf.constant(value=2e-4)                
        labels = tf.to_float(labels)        
        softmax = tf.nn.softmax(prediction) + epsilon
        
        if label_balance is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), label_balance), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss