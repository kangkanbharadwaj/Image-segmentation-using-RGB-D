import tensorflow as tf
import os, random
from glob import glob
import re
import numpy as np
import cv2
import itertools

BATCH_SIZE = 1
BUFFER_SIZE = 1
IMG_H = 304
IMG_W = 304
    
def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


################################################################################## Use this function to train for RGB data ###############################################################
#### Function to decode the images and read them as tensors for the network  #####
def decodeRGB(image, label):         

    r_image_string = tf.read_file(image)    
    r_image_decoded = tf.image.decode_jpeg(r_image_string,channels=3)      
    r_image_decoded = tf.reshape(r_image_decoded,[IMG_H,IMG_W,3]) 
    r_image_decoded = tf.image.convert_image_dtype(r_image_decoded,tf.float32) 
    r_image_decoded = tf.divide(r_image_decoded,[255])

    s_image_string = tf.read_file(label)  
    s_image_decoded = tf.image.decode_png(s_image_string,channels=1)
    s_image_decoded = tf.reshape(s_image_decoded,[IMG_H,IMG_W,1])
    
    return r_image_decoded, s_image_decoded


#### Function to map the image directories to be read as tensors  #####
def parseRGB(mode=None ,trainDir=None, ValDir=None, testDir=None):
   
    if mode == 'train':        
        print ("\nInput queue is in training mode")
        image_files = sorted(glob(trainDir+"/*.jpg"), key=stringSplitByNumbers)
        label_files = sorted(glob(trainDir+"/*.png"), key=stringSplitByNumbers)            
        image_filenames = tf.constant(image_files)
        label_filenames = tf.constant(label_files)  
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, label_filenames))              
        buffer_size = len(image_files)
        print ("The buffer size for shuffling is: %d" %(buffer_size))
        dataset = dataset.map(decodeRGB).apply(tf.contrib.data.shuffle_and_repeat(buffer_size = buffer_size)).batch(1)        
        #dataset = dataset.prefetch(4)
        return dataset
    elif mode == 'validate':
        print ("\nInput queue is in validating mode")
        image_files = sorted(glob(ValDir+"/*.jpg"), key=stringSplitByNumbers)
        label_files = sorted(glob(ValDir+"/*.png"), key=stringSplitByNumbers)   
        image_filenames = tf.constant(image_files)
        label_filenames = tf.constant(label_files)        
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, label_filenames))
        dataset = dataset.map(decodeRGB).batch(1)
        return dataset
    elif mode == 'test':
        print ("\nInput queue is in testing mode")
        image_files = sorted(glob(testDir+"/*.jpg"), key=stringSplitByNumbers)
        label_files = sorted(glob(testDir+"/*.png"), key=stringSplitByNumbers)   
        image_filenames = tf.constant(image_files)
        label_filenames = tf.constant(label_files)        
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, label_filenames))
        dataset = dataset.map(decodeRGB).batch(1)
        return dataset
    
    
###########################################################################################################################################################################################
###########################################################################################################################################################################################

#### Function that provides extensive data augmentation (if depth is available then feed it as a function argument and uncomment the necessary parts of code)  #####
def data_Augmentation(train_image,train_label,crop,CROP_H,CROP_W):

    # Squeeze first dimension
    train_image = tf.squeeze(train_image,axis=0)
    train_label = tf.squeeze(train_label,axis=0)
    
    # Get height and width
    dim = train_image.get_shape().as_list()
    IMGH = dim[0]
    IMGW = dim[1]
    
    # Select the augmentation mode
    mode = tf.random_uniform([],0,8, dtype = tf.int32)    
            
    if crop == 'True':
        ##### Crop random patches from original image of size 256X256 #####      
        offset_height = tf.random_uniform([],0,(IMGH-CROP_H), dtype = tf.int32)
        offset_width = tf.random_uniform([],0,(IMGW-CROP_W), dtype = tf.int32)
        crop_img = tf.image.crop_to_bounding_box(train_image,offset_height = offset_height, offset_width = offset_width, target_height = 256, target_width = 256)
        crop_lbl = tf.image.crop_to_bounding_box(train_label,offset_height = offset_height, offset_width = offset_width, target_height = 256, target_width = 256)
    else:
        crop_img = train_image
        crop_lbl = train_label
            
    ##### No augmentation    #####        
    def noAugmentation(train_image,train_label):
        aug_Image = train_image
        aug_Label = train_label
        #aug_Depth = train_depth
        return aug_Image,aug_Label        
    
    ##### Flip image horizontally    #####        
    def flip_left_right(train_image,train_label):
        aug_Image = tf.image.flip_left_right(train_image)
        aug_Label = tf.image.flip_left_right(train_label)        
        return aug_Image,aug_Label        
    
    ##### Flip image vertically    #####    
    def flip_up_down(train_image,train_label):
        aug_Image = tf.image.flip_up_down(train_image)
        aug_Label = tf.image.flip_up_down(train_label)             
        return aug_Image,aug_Label        
    
    ##### Random translation of image    #####        
    def translation(train_image,train_label):
        translations = np.array([random.randint(1,50), random.randint(1,50)])
        aug_Image = tf.contrib.image.translate(train_image,translations = translations,interpolation='NEAREST')
        aug_Label = tf.contrib.image.translate(train_label,translations = translations,interpolation='NEAREST')        
        return aug_Image,aug_Label        
    
    ##### Roate image in 90, 180 or 270 degrees    #####    
    def rotate(train_image,train_label):
        k = tf.random_uniform([],1,4, dtype = tf.int32)
        aug_Image = tf.image.rot90(train_image, k)
        aug_Label = tf.image.rot90(train_label, k)        
        return aug_Image,aug_Label
                
    ##### Play with the brightness (only for the RGB image -- brightness values range from 0.3 to 0.9)    #####    
    def adjust_brightness(train_image,train_label):
        delta = tf.random_uniform([],0.3,0.9, dtype = tf.float32)
        aug_Image = tf.image.adjust_brightness(train_image, delta=delta)        #img = tf.map_fn(lambda x: tf.image.adjust_brightness(x, delta=delta), img)    
        aug_Label = train_label        
        return aug_Image,aug_Label
        
    #### Change contrast of image in random fashion from 0.3 to 0.9   #####    
    def contrast(train_image,train_label):
        contrast_factor = tf.random_uniform([],0.3,0.9, dtype = tf.float32)
        aug_Image = tf.image.adjust_contrast(train_image, contrast_factor=contrast_factor)
        aug_Label = train_label        
        return aug_Image,aug_Label        
    
    #### Change saturation of image in random fashion from 0.3 to 0.9   #####    
    def random_saturation(train_image,train_label):
        saturation_factor = tf.random_uniform([],0.5,2.0, dtype = tf.float32)
        aug_Image = tf.image.adjust_saturation(train_image,saturation_factor=saturation_factor)
        aug_Label = train_label        
        return aug_Image,aug_Label
        
    ##### Add noise to the RGB image with 0.0 mean and 1.0 stddev    #####    
    def add_GausNoise(train_image,train_label):
        mean = tf.random_uniform([],0.0,1.0, dtype = tf.float32)    
        stddev = tf.random_uniform([],0.0,5.0, dtype = tf.float32)       
        noise = tf.random_normal([256,256,3],mean=mean, stddev=stddev, dtype = tf.float32)
        aug_Image = tf.add(train_image, noise)    
        aug_Image = tf.clip_by_value(aug_Image, 0.0, 1.0)
        aug_Label = train_label        
        return aug_Image,aug_Label
        
    # switch case to pick an augmentation choice
    def f0(): return noAugmentation(crop_img,crop_lbl)
    def f1(): return flip_left_right(crop_img,crop_lbl)
    def f2(): return flip_up_down(crop_img,crop_lbl)
    def f3(): return translation(crop_img,crop_lbl)
    def f4(): return rotate(crop_img,crop_lbl)
    def f5(): return adjust_brightness(crop_img,crop_lbl)
    def f6(): return contrast(crop_img,crop_lbl)
    def f7(): return random_saturation(crop_img,crop_lbl)
    def f8(): return add_GausNoise(crop_img,crop_lbl)   ### not included as a choice yet
    
    # Augment data on the fly
    aug_Image, aug_Label = tf.case({tf.equal(mode,0): f0, tf.equal(mode,1): f1, tf.equal(mode,2): f2, tf.equal(mode,3): f3, tf.equal(mode,4): f4, tf.equal(mode,5): f5, tf.equal(mode,6): f6, tf.equal(mode,7): f7}, exclusive=True)   
    
    # expand the dimensions to 4
    aug_Image = tf.expand_dims(aug_Image,axis=0)
    aug_Label = tf.expand_dims(aug_Label,axis=0)
    
    return aug_Image, aug_Label  