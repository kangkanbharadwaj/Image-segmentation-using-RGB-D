import tensorflow as tf
import numpy as np
import time, os, random, re, cv2
from resnet50 import *
from DDNet import *
from UNet import *
from input_pipeline import *
from class_weighing import *
from post_processing import *
from metrics import *
from loss import *
from glob import glob


def test(Model=None,imgh=0,imgw=0,btsize=0,predDir=None,modDir=None,RGBDir=None,tgtDir=None,ckptDir=None,testDir=None,txtDir=None,n_class=0,model_to_test=0):
    
    print ("\n##############################  Testing on test data using the model %d " %(model_to_test))
    tf.reset_default_graph()

    # Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # allocates only the required amount of memory
    config.log_device_placement = False # check which part of code is executed in which core
    config.allow_soft_placement=True 
    
    # Create the directories if it doesn't exist 
    if not os.path.exists(predDir):
        os.makedirs(predDir)
    if not os.path.exists(modDir):
        os.makedirs(modDir)
    if not os.path.exists(RGBDir):
        os.makedirs(RGBDir)
    if not os.path.exists(tgtDir):
        os.makedirs(tgtDir)
        
    net_input = tf.placeholder(tf.float32,shape=[btsize,imgh,imgw,3])
    training = tf.constant(False,tf.bool,shape=[])

    # Load the network to get the predicted output
    if Model == 'ResNet':
        pred_score = ResNet50(net_input,training,n_class)   
    if Model == 'UNet':
        pred_score = make_unet(net_input,training,n_class)  
    elif Model == 'DDNET':
        Reshaped_map_decoder1, Reshaped_map_decoder2, pred_score = DPDB_encoder_stacking_decoder_DeepSupervision_300_threeStages_Cardinality_ResidualStack(net_input, training, n_class)
        
    # Load RGB/RGBD data 
    print("Loading the test data ...")
    testSet = sorted(glob(testDir+"/*.jpg"), key=stringSplitByNumbers)
    test_input_queue = parseRGB(mode='test',testDir=testDir)
    iterator = test_input_queue.make_initializable_iterator() 
    test_image, test_label = iterator.get_next()

    print('Loading model checkpoint weights ...')
    saver=tf.train.Saver()
    
    px_acc_l = []
    iou_l = []
    px_acc_testSet = 0.0
    iou_testSet = 0.0
    mean_cls_acc = 0.0
    tot_class_accuracies = []

    with tf.Session(config=config) as sess:
        try:            
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator.initializer)  
            
            # Open the text file to write test metrics                
            iu_File = open(txtDir+"/validation_test_results.txt", "a")
            iu_File.write("############################################ Performance measured on test set ###############################################")
            iu_File.write("\n\n")
            
            saver = tf.train.import_meta_graph(ckptDir+"/seg_model-%d.meta" %(model_to_test))
            saver.restore(sess, ckptDir+"/seg_model-%d" %(model_to_test))
            #saver.restore(sess, tf.train.latest_checkpoint('/home/bharadwaj/ImageSegmentation/data/snapshots/'))
            print ("\nTest dataset size for test directory is %d\n" %((len(os.listdir(testDir))/2)))
           
            for i in range((len(os.listdir(testDir))/2)):             
                testImage,testLabel = sess.run([test_image, test_label])                                                                             
                pred_label = sess.run(pred_score,feed_dict={net_input:testImage,training:False})
                pred_label = np.squeeze(pred_label,axis=0)
                pred_label = np.argmax(pred_label, axis=2)
                cv2.imwrite(predDir+'/'+str(testSet[i].split('/')[len(testSet[i].split('/'))-1].split('.')[0]+'.png') ,pred_label)
                pixel_accuracy, class_accuracies, iou  = evaluate_segmentation_per_image(pred=pred_label, label=testLabel, num_classes=n_class)
                print ("Testing on image %s \t Accuracy: %.2f \t IOU: %.2f" %(testSet[i].split('/')[len(testSet[i].split('/'))-1],pixel_accuracy*100.0,iou*100.0))                
                px_acc_l.append(pixel_accuracy)
                iou_l.append(iou)                 
                tot_class_accuracies.append(class_accuracies)   
            
            px_acc_testSet = np.mean(px_acc_l)*100.0
            iou_testSet = np.mean(iou_l)*100.0
            mean_cls_acc = np.mean([np.asarray(tot_class_accuracies[i]) for i in range(len(tot_class_accuracies))], axis=0)
            mean_cls_acc[:] = [ x * 100.0 for x in mean_cls_acc ]
                        
            print ("\n------------------------------------------------------------------- Saving the metrics in the file --------------------------------------------------------------\n") 
            iu_File.write("pixel accuracy for model %d is = %f" %(i,px_acc_testSet))
            iu_File.write("\n")
            iu_File.write("iou for model %d is = %f" %(i,iou_testSet))
            iu_File.write("\n")
            iu_File.write("Per class accuracies for model %d are = %s" %(i,mean_cls_acc))
            iu_File.write("\n\n")                   
            
            print ("\n-------------------------------------------------------------------------- Pixel Accuracy on test set: %.3f ------------------------------------------------------------------" %(px_acc_testSet))
            print ("\n-------------------------------------------------------------------------- IOU on test set: %.3f -----------------------------------------------------------------------" %(iou_testSet)) 
            print ("\n-------------------------------------------------------------------------- Independent class Accuracies on test set: \n %s" %(mean_cls_acc))
             
            
            # Replace the ignored label with appopriate label value from ground truth mask as the network was not trained to learn on it
            replace_ignoreLabel(testDir, predDir, modDir)
            print ("\n ------------------------------------------------------------------------- Done!!! -------------------------------------------------------------------------------------")
            # Color the images
            colorMasks(modDir, RGBDir)            
            print ("\n-------------------------------------------------------------------------- Post processing done: gray to RGB ------------------------------------------------------------")
            print ("\n-------------------------------------------------------------------------- Segmentation over... check your results!!! ------------------------------------------------------------")
            iu_File.close()
            
        finally:            
            sess.close()

if __name__ == '__main__':
    test()