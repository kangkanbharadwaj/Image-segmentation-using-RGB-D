import tensorflow as tf
import numpy as np
import time, os, random, re, cv2
from resnet50 import *
from DDNet import *
from input_pipeline import *
from class_weighing import *
from post_processing import *
from metrics import *
from loss import *
from glob import glob

################################################################################################ Invoke the main function to start training ############################################################################################################# 

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

def train_validate(Model=None,N_Class=0,imgh=0,imgw=0,croph=0,cropw=0,c_blc=None,c_blc_T=None,opt=None,LR=0,decay=0.0,Momentum=0.95,epsilon=0.0,beta1=0.0,epoch=100,
                   bfSize=10,btSize=1,trainDir=None,ValDir=None,trainLogDir=None,ValLogDir=None,ckptDir=None,TxtDir=None,MX_ITR=0,MX2KEEP=0,brkpt=0,augmentation=None,crop=None):    
    
    tf.reset_default_graph()
    
    # Initialize the test model
    model_to_test = 0   
    
    # Create the directories if does not exist
    if not os.path.exists(ckptDir):
        os.makedirs(ckptDir)
    if not os.path.exists(trainLogDir):
        os.makedirs(trainLogDir)
    if not os.path.exists(ValLogDir):
        os.makedirs(ValLogDir)
    
    print ("\n------------------------------------------------------------------ Image ImageSegmentation using TF -------------------------------------------------------------------")
    print ("\n------------------------------------------------------------------ Class description-----------------------------------------------------------------------------------\n")
    print ("Class 0: Asphalt\nClass 1: Concrete pavement/Mosaic pavement/Natural stone pavement/Loose natural surface/ Gutter, curb stone\nClass 2: Hole cover\nClass 3: Tree trunk\nClass 4: Broadleaf crown/Conifer crown\nClass 5: Bush\nClass 6: Street sign\nClass 7: Streetlight\nClass 8: Traffic light\nClass 9: Guardrail\nClass 10: Encasement of equipment\nClass 11: Advertisement panel\nClass 12: Building\nClass 13: Enclosure\nClass 14: Metal pole\nClass 15: Sky\nClass 16: Vehicle\nClass 17: People\nClass 18:  Bollard\nClass 19: Undefined")
    
    print ("\n\nCollecting parsed arguments for the model...")
    print ("Model to work with: %s\nNumber of classes are: %d\nImage height: %d---Image width: %d---Crop height: %d---Crop width: %d\nClass balance required: %s\nClass balance type: %s\nOptimizer: %s---Learning Rate: %f---Decay: %f---Momentum: %f---Epsilon: %f---Beta1: %f\nEpoch: %d\nBuffer size: %d Batch size: %d\nMax iterations: %d\nMaximum checkpoints to keep: %d\nBreakpoints at every %d iterations\nTrain directory: %s\nValidation Directory: %s\nCheckpoint directory: %s\nTextfile Directory: %s\nTrain log directory: %s\nValidation log directory: %s\nAugmentation Required: %s\nCropping required: %s" %(Model,N_Class,imgh,imgw,croph,cropw,c_blc,c_blc_T,opt,LR,decay,Momentum,epsilon,beta1,epoch,bfSize,btSize,                                                                                                                                                                                                                                                                MX_ITR,MX2KEEP,brkpt,trainDir,ValDir,ckptDir,TxtDir,trainLogDir,ValLogDir,augmentation,crop))
    
    print ("\n------------------------------------------------------------------ Initialize train data ---------------------------------------------------------------------------------")
    
    # Initialize empty placeholders to be fed during the session train operation
    if crop == 'True':
        X = tf.placeholder(tf.float32, shape=[btSize, croph, cropw, 3], name="X")    
        y = tf.placeholder(tf.uint8, shape=[btSize, croph, cropw, 1], name="y")         
    else:
        X = tf.placeholder(tf.float32, shape=[btSize, imgh, imgw, 3], name="X")    
        y = tf.placeholder(tf.uint8, shape=[btSize, imgh, imgw, 1], name="y")   
    
    training = tf.placeholder(tf.bool, shape=[], name="mode")
    v_loss = tf.placeholder(tf.float32, shape=[], name="v_loss")
    v_iou = tf.placeholder(tf.float32, shape=[], name="v_iou")
    v_acc = tf.placeholder(tf.float32, shape=[], name="v_acc")
    
    print ("\n----------------------------------------------------------------------- Loading the model --------------------------------------------------------------------------------")
    
    # Load the network to get the predicted output
    if Model == 'ResNet':
        pred_score = ResNet50(X,training,N_Class)   
    elif Model == 'DDNET':
        Reshaped_map_decoder1, Reshaped_map_decoder2, pred_score = DPDB_encoder_stacking_decoder_DeepSupervision_300_threeStages_Cardinality_ResidualStack(X, training,N_Class)
    elif Model == 'UNet':
        pred_score = ResNet50(X,training,N_Class)   
    
    print ("\n----------------------------------------------------- Reshaping the logits and masks (Masks are one hot encoded) ---------------------------------------------------------")
    
    # Label is one hot encoded and reshaped for cross-entropy computation loss later on
    segMap = tf.reshape(pred_score, (-1, N_Class))
    one_hot_encoded_label = tf.one_hot(tf.squeeze(y, axis=-1), N_Class, axis=-1)    
    Reshaped_labels = tf.reshape(one_hot_encoded_label, shape=(-1, N_Class))
        
    print ("\nShapes of the one_hot_encoded_label and flat label are: " + str(one_hot_encoded_label.get_shape().as_list())+"\t"+str(Reshaped_labels.get_shape().as_list()))
    print ("Shapes of the logit and flattened logits are: " + str(pred_score.get_shape().as_list())+"\t"+str(segMap.get_shape().as_list()))
    
            
    print ("\n------------------------------------------------------------ Computing the cross entropy ----------------------------------------------------------------------------------")       
    
    #with tf.device("/cpu:0"):
    
    # Initilaze an empty class weights list
    class_weights = []
    
    # Compute the cross entropy loss ignoring undefined object label 
    if c_blc == 'True':
        trainMasks = sorted(glob(trainDir+'/*.png'), key=stringSplitByNumbers)
        if c_blc_T == 'weigh_equal_importance':   
            print ("\nIgnore the last label and weigh rest of the labels equally")
            class_weights = [1.0] * (N_Class-1)
            class_weights.insert((N_Class-1), 0.0) 
            print ("The class weights are: %s" %(class_weights))
            class_weights = tf.reshape(tf.convert_to_tensor(class_weights, dtype=tf.float32), [N_Class])             
        elif c_blc_T == 'cls_weighing': 
            print ("\nENet class weighing is selected")
            class_weights = ENet_weighing(trainMasks,N_Class)
            print ("The class weights are: %s" %(class_weights))
            class_weights = tf.reshape(tf.convert_to_tensor(class_weights, dtype=tf.float32), [N_Class]) 
        else:
            print ("\nENet Frequency median is selected")
            class_weights = median_frequency_balancing(trainMasks,N_Class) 
            print ("The class weights are: %s" %(class_weights))
            class_weights = tf.reshape(tf.convert_to_tensor(class_weights, dtype=tf.float32), [N_Class]) 
    else:
         class_weights = None   
    
    
    print ("Class Weights shape as a tensor: " + str(class_weights.get_shape().as_list()))
    
    if Model == 'ResNet':
        print ("Computing loss from ResNet")
        Totalloss = cross_entropy_loss(segMap, Reshaped_labels, N_Class,class_weights)
    elif Model == 'DDNET':
        print ("Computing loss from DDNET")
        Loss_decoder1 = cross_entropy_loss(Reshaped_map_decoder1, Reshaped_labels, N_Class,class_weights)
        Loss_decoder2 = cross_entropy_loss(Reshaped_map_decoder2, Reshaped_labels, N_Class,class_weights)
        Loss_final_decoder = cross_entropy_loss(segMap, Reshaped_labels, N_Class,class_weights)
        Totalloss = Loss_decoder1 + Loss_decoder2 + Loss_final_decoder 
    elif Model == 'UNet':
        print ("Computing loss from UNet")
        Totalloss = cross_entropy_loss(segMap, Reshaped_labels, N_Class,class_weights)
        
    # Create the global step to keep track of iterations
    global_step = tf.train.get_or_create_global_step()    
    
    # optimizer.compute_gradients and optimizer.apply_gradients is equivalent to running train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)    
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)    # Use this if updation is not handled in place during batch norm i.e if updates_collections=tf.GraphKeys.UPDATE_OPS
    #with tf.control_dependencies(update_ops):  
    if opt == 'Adam':
        print ("\n------------------------------------------------------------------- ADAM OPTIMIZER ON THE GO -------------------------------------------------------------------------------")
        lr = tf.train.exponential_decay(LR, global_step, epoch, decay, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(Totalloss, global_step=global_step)
    elif opt == 'RMSProp':    
        lr = tf.train.exponential_decay(LR, global_step, epoch, decay, staircase=True)
        print ("\n------------------------------------------------------------------- RMSPROP OPTIMIZER ON THE GO -------------------------------------------------------------------------------")
        optimizer = tf.train.RMSPropOptimizer(lr, decay, Momentum, epsilon).minimize(Totalloss, global_step=global_step)
    elif opt == 'SGD':        
        lr = tf.train.exponential_decay(LR, global_step, epoch, decay, staircase=True)
        print ("\n------------------------------------------------------------------- SGD OPTIMIZER ON THE GO -------------------------------------------------------------------------------")
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=Momentum).minimize(Totalloss, global_step=global_step)
    
    #Performance Measures
    print ("\n------------------------------------------------------------------- Computing respective Accuracies and IOUs ----------------------------------------------------------------")
    
    # Accuracy       
    softmax = tf.nn.softmax(segMap)
    correct_prediction = tf.equal(tf.argmax(segMap,1), tf.argmax(Reshaped_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(tf.argmax(softmax,1), tf.argmax(Reshaped_labels,1))
    accuracy = accuracy * 100.0  

    # intersection over union
    IOU, IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=tf.cast(tf.argmax(softmax,1), tf.int32), labels=tf.cast(tf.argmax(Reshaped_labels,1),tf.int32), num_classes=N_Class)
    IOU = IOU * 100.0
    logits=tf.reshape(segMap, [-1])
    trn_labels=tf.reshape(Reshaped_labels, [-1])
    #logits=tf.argmax(segMap,1)
    #trn_labels=tf.argmax(Reshaped_labels,1)
    #logits=tf.cast(logits, tf.float32)
    #trn_labels=tf.cast(trn_labels, tf.float32)
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    val_iou=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    val_iou=val_iou*100.0
    
    
    print ("\n------------------------------------------------------------------------ Feed the RGB data ----------------------------------------------------------------------------------")
    
    # Load RGB/RGBD data 
    
    # Fetch one RGB train element at a time and augment it
    train_input_queue = parseRGB(mode='train',trainDir=trainDir)
    train_iterator = train_input_queue.make_initializable_iterator() 
    _train_image, _train_label = train_iterator.get_next()
    if augmentation == 'True':
        train_image, train_label = data_Augmentation(_train_image,_train_label,crop,croph,cropw)
    
    # Fetch one RGB validate element at a time and augment it
    val_input_queue = parseRGB('validate',ValDir=ValDir)
    val_iterator = val_input_queue.make_initializable_iterator() 
    val_image, val_label = val_iterator.get_next()

    print ("\n--------------------------------------------------------------- Read a single batch successfully -----------------------------------------------------------------------------") 
    
    # Log the training in the respective directory
    print ("\n-------------------------------------------------------- Logging: training and validation logs --------------------------------------------------------------------------------")
    log_time = time.asctime( time.localtime(time.time()))
    log_time = log_time.replace(" ", "_")
    train_logdir = os.path.join(trainLogDir, log_time)
    val_logdir = os.path.join(ValLogDir, log_time)    
    
    # Counter for total number of iterations performed so far.
    init_iterations = 0    
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    tot_loss = 0.0
    tot_acc = 0.0
    tot_iou = 0.0
    
    mean_acc_valSet = []
    mean_iou_valSet = []
    mean_CELoss_valSet = {}
    overfit = 0
    
    # Accumulate summaries and merge them to display in tensorboard        
    print ("\n---------------------------------------------- Accumulate training summaries and merge them to display in tensorboard ------------------------------------------------------------------")    
    im_h = tf.summary.histogram("Softmax activation on predicted image", softmax)    
    im_tr = tf.summary.image("Image to train", _train_image, max_outputs=1)
    mk_tr = tf.summary.image("Label to train", _train_label, max_outputs=1)     
    if augmentation == 'True':
        im_tr_aug = tf.summary.image("Augmented image to train", train_image, max_outputs=1)        
        mk_tr_aug = tf.summary.image("Augmented label to train", train_label, max_outputs=1)
        if crop == 'True':
            mk_p = tf.summary.image("Predicted mask", tf.cast(tf.reshape((tf.argmax(softmax,1)), [1,croph,cropw,1]), tf.uint8), max_outputs=1) 
        else:
            mk_p = tf.summary.image("Predicted mask", tf.cast(tf.reshape((tf.argmax(softmax,1)), [1,imgh,imgw,1]), tf.uint8), max_outputs=1) 
    else:
        mk_p = tf.summary.image("Predicted mask", tf.cast(tf.reshape((tf.argmax(softmax,1)), [1,imgh,imgw,1]), tf.uint8), max_outputs=1) 
    t_l = tf.summary.scalar('Cross entropy loss', Totalloss) 
    t_a = tf.summary.scalar('Accuracy', accuracy)
    t_I = tf.summary.scalar('Intersection Over Union', IOU)
    #summary_op = tf.summary.merge_all()
    
    print ("\n---------------------------------------------- Accumulate validation summaries and merge them to display in tensorboard ------------------------------------------------------------------")    
    
    v_im = tf.summary.image("Validation Image", val_image, max_outputs=1)
    v_mk = tf.summary.image("Validation Label", val_label, max_outputs=1)
    if crop == 'True':
        v_p = tf.summary.image("Validated mask", tf.cast(tf.reshape((tf.argmax(softmax,1)), [1,croph,cropw,1]), tf.uint8), max_outputs=1) 
    else:
        v_p = tf.summary.image("Validated mask", tf.cast(tf.reshape((tf.argmax(softmax,1)), [1,imgh,imgw,1]), tf.uint8), max_outputs=1) 
    v_l = tf.summary.scalar('Validation Cross entropy loss', v_loss) 
    v_a = tf.summary.scalar('Validation Accuracy', v_acc)
    v_I = tf.summary.scalar('Validation Intersection Over Union', v_iou)
    #val_op = tf.summary.merge_all()
        
    
    # Create the directories if doesn't exist yet
    print ("\n----------------------------------------------------------------- Creating directories if does not exist ----------------------------------------------------------------------")
    if not os.path.exists(train_logdir):
        os.makedirs(train_logdir)
    if not os.path.exists(val_logdir):
        os.makedirs(val_logdir)
        
    # Create savers and tensorboard 
    print ("\n----------------------------------------------------------------- Provision for saving models and creating Tensorboard ---------------------------------------------------------")
    saver = tf.train.Saver(max_to_keep=MX2KEEP)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    
    # Configure the GPU options according to your requirement
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # allocates only the required amount of memory
    config.log_device_placement = False # check which part of code is executed in which core
    config.allow_soft_placement=False    # allow automatic placement of necessary operation on the respective cores
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8   # how much percentage of memory do you want to allocate per GPU
    
    with tf.Session(config=config) as sess:
        try:
            # Time noted when actual train processess
            _start = time.time()
            
            #log performance training### Create the summary writer -- to write all the logs into a specified file. This file can be later read by tensorboard.
            train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph) 
            val_summary_writer = tf.summary.FileWriter(val_logdir)      
                        
            #resume training exactly where you left from 
            global_step = tf.train.get_global_step(sess.graph)
            
            # Initialize all the variables
            sess.run(init_global)
            sess.run(init_local)       
            
            # Initialize the trainable iterator            
            sess.run(train_iterator.initializer)            
            sess.run(val_iterator.initializer) 
             
                                        
            # Get a text file to save important validation metrics for easy readability                
            iu_File = open(TxtDir+"/validation_test_results.txt", "a")
            iu_File.write("############################################ Performance measured on validation set ###############################################")
            iu_File.write("\n\n")
                
            # Restore checkpoint to continue training if checkpoint exist            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckptDir+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                print("\n----------------------------------------------------- Restoring from latest checkpoint ------------------------------------------------------")                
                saver.restore(sess, '/home/bharadwaj/ImageSegmentation/data/snapshots/seg_model-245000')  
                #saver.restore(sess, ckpt.model_checkpoint_path)  
                init_iterations = global_step.eval() 
            
            # Initialize the validation set
            valset = sorted(glob(ValDir+'/*.jpg'), key=stringSplitByNumbers)                        
            
            # Start training or resume training
            if init_iterations == 0:
                print("\n============================================================================= Training started from scratch ================================================================================")
                print("\n============================================================================= Loss display in every 100 iterations =========================================================================\n")
            else :
                print("\n============================================================================= Resuming training from iteration %d ==========================================================================" %(init_iterations))
                print("\n============================================================================= Loss display in every 100 iterations =========================================================================\n")
            
            for i in range(init_iterations, MX_ITR):
                
                # Measure time elapsed
                start = time.time() 
                
                # Initialize local metrics
                c_loss=0.0
                acc=0.0
                iou=0.0
                                
                if augmentation == 'True':
                    # Get dynamic augmented images and labels
                    trainimg, trainlbl = sess.run([train_image, train_label])
                    # Get all details to be displayed -- summarized on tensorboard
                    c_loss,train_step,acc,iou,_,imtraug,mktraug,imh,imtr,mktr,mkp,tl,ta,tI,global_step_value = sess.run([Totalloss,optimizer,accuracy,IOU,IOU_update,im_tr_aug,mk_tr_aug,im_h,im_tr,mk_tr,mk_p,t_l,t_a,t_I,global_step], 
                                                                                        feed_dict={X:trainimg,y:trainlbl,training:True}) 
                    # Summarize important details
                    train_summary_writer.add_summary(imh,global_step_value)
                    train_summary_writer.add_summary(imtr,global_step_value)                
                    train_summary_writer.add_summary(mktr,global_step_value)
                    train_summary_writer.add_summary(imtraug,global_step_value)
                    train_summary_writer.add_summary(mktraug,global_step_value)
                    train_summary_writer.add_summary(mkp,global_step_value)                
                    train_summary_writer.add_summary(tl,global_step_value)                
                    train_summary_writer.add_summary(ta,global_step_value)
                    train_summary_writer.add_summary(tI,global_step_value) 
                else:
                    # Get dynamic augmented images and labels
                    trainimg, trainlbl = sess.run([_train_image, _train_label])
                    # Get all details to be displayed -- summarized on tensorboard
                    c_loss,train_step,acc,iou,_,imh,imtr,mktr,mkp,tl,ta,tI,global_step_value = sess.run([Totalloss,optimizer,accuracy,IOU,IOU_update,im_h,im_tr,mk_tr,mk_p,t_l,t_a,t_I,global_step], 
                                                                                        feed_dict={X:trainimg,y:trainlbl,training:True}) 
                    # Summarize important details
                    train_summary_writer.add_summary(imh,global_step_value)
                    train_summary_writer.add_summary(imtr,global_step_value)                
                    train_summary_writer.add_summary(mktr,global_step_value)
                    train_summary_writer.add_summary(mkp,global_step_value)                
                    train_summary_writer.add_summary(tl,global_step_value)                
                    train_summary_writer.add_summary(ta,global_step_value)
                    train_summary_writer.add_summary(tI,global_step_value) 
                    
                #print (a)
                total_loss = total_loss + c_loss
                total_acc = total_acc + acc 
                total_iou = total_iou + iou 
                tot_loss = tot_loss + c_loss
                tot_acc = tot_acc + acc
                tot_iou = tot_iou + iou               
                current_time = str(time.asctime( time.localtime(time.time())))
                end = time.time()
                time_Elapsed = abs(end - start)
                
                # Display the loss details in every 100 iterations
                if i > 0 and i % 100 == 0:
                    mean_loss = tot_loss/100.0
                    mean_acc = tot_acc/100.0
                    mean_iou = tot_iou/100.0
                    print("Iteration: %d \t\t Time: %s \t Per image processing time: %.2f secs \t\t Cross_entropy_loss: %.3f \t Accuracy: %.2f \t IOU: %.2f" %(i,current_time,time_Elapsed,mean_loss,mean_acc,mean_iou))
                    tot_loss = 0.0
                    tot_acc = 0.0
                    tot_iou = 0.0

                # Save the snapshots/checkpoints in every brkpt iterations
                if i > 0 and i % brkpt == 0:
                    print ("\n---------------------- Mean cross_entropy_loss and accuracy (over %d iterations) at %dth step are ----------------------------" %(brkpt,i))
                    mean_loss = float(total_loss/brkpt)
                    mean_acc = float(total_acc/brkpt)
                    mean_iou = float(total_iou/brkpt)
                    print ("\n--------------------- Loss: %.3f \t\t Acc: %.2f \t\t IOU: %.2f  -----------------------------------------------" %(mean_loss,mean_acc,mean_iou))
                                      
                    total_loss = 0.0
                    total_acc = 0.0
                    total_iou = 0.0                 
                    
                    # Save the first model checkpoint
                    
                    print ("\nSaving model %d in snapshots directory" %(i))                                        
                    saver.save(sess, ckptDir + '/seg_model',global_step=i)
                    _end = time.time()
                    elapsed_time = int(_end - _start)                    
                    print ("\nTraining since : {:02d}:{:02d}:{:02d} hours".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60))
                                       
                    acc_valSet = []
                    iou_valSet = [] 
                    loss_valSet = [] 
                    acc_V = 0.0
                    iou_V = 0.0
                    loss_V = 0.0
                                        
                    # Call validate to test model performance
                    print ("\n================================================================== Validating on validation set at %dth step ===================================================================\n"%(i))
                    
                    # Validate on small patches if trained on small patches
                    if crop == 'True':                    
                        print ("\nValidating on patches of 256X256")
                        print ("\nValidation directory has %d data\n" %((len(os.listdir(ValDir))/2)))
                        for n in range (len(valset)):                        
                            valimg, vallbl = sess.run([val_image, val_label])
                            
                            # Collect height and width
                            height = valimg.shape[1]
                            width = valimg.shape[2] 
                            
                            # Squeeze the first dimensions
                            valimg = np.squeeze(valimg,axis=0)
                            vallbl = np.squeeze(vallbl,axis=0)
                            
                            patch=1
                            p=0
                            while p+256 <= height:
                                j=0
                                while j+256 <= width:                                                                
                                    # Crop 256X256 patch
                                    valImg = valimg[p:p+256, j:j+256] 
                                    valLabel = vallbl[p:p+256, j:j+256]                                 
                                    
                                    # Expand the dimensions back to 4
                                    valImg = np.expand_dims(valImg,axis=0)
                                    valLabel = np.expand_dims(valLabel,axis=0)
                                                                                                    
                                    # validate model on patch
                                    valloss,valacc,valiou,v_image,v_mask,val_pred = sess.run([Totalloss,accuracy,val_iou,v_im,v_mk,v_p], feed_dict={X:valImg, y:valLabel, training: False})    
                                    print ("Validating on image %s \t patch: %d \t CE Loss: %.3f \t Accuracy: %.2f \t IOU: %.2f" %(valset[n].split('/')[len(valset[n].split('/'))-1],patch,valloss,valacc,valiou))                                                                               
                                                                    
                                    # Summarize important details
                                    val_summary_writer.add_summary(v_image,global_step_value)
                                    val_summary_writer.add_summary(v_mask,global_step_value)
                                    val_summary_writer.add_summary(val_pred,global_step_value)
                                    
                                    # Accumulate metrics to determine average over the entire set
                                    acc_valSet.append(valacc)
                                    iou_valSet.append(valiou)
                                    loss_valSet.append(valloss)                                                                              
                                    j = j + 256
                                    patch =patch + 1
                                p = p + 256
                            print ("\n")
                    else:
                        print ("\nValidating on original image resolution\n")
                        #print ("\nValidation directory has %d data\n" %((len(os.listdir(ValDir))/2)))
                        print ("\nValidation directory has %d data\n" %(len(valset)))
                        for n in range (len(valset)):                        
                            valimg, vallbl = sess.run([val_image, val_label])                                                                                                   
                            
                            # validate model on image
                            valloss,valacc,valiou,v_image,v_mask,val_pred = sess.run([Totalloss,accuracy,val_iou,v_im,v_mk,v_p], feed_dict={X:valimg, y:vallbl, training: False})    
                            print ("Validating on image %s \t  CE Loss: %.3f \t Accuracy: %.2f \t IOU: %.2f" %(valset[n].split('/')[len(valset[n].split('/'))-1],valloss,valacc,valiou))                                                                               
                                                            
                            # Summarize important details
                            val_summary_writer.add_summary(v_image,global_step_value)
                            val_summary_writer.add_summary(v_mask,global_step_value)
                            val_summary_writer.add_summary(val_pred,global_step_value) 
                            
                            # Accumulate metrics to determine average over the entire set
                            acc_valSet.append(valacc)
                            iou_valSet.append(valiou)
                            loss_valSet.append(valloss)                                                                                                                               
                                                                                                                                       
                    acc_V = np.mean(acc_valSet)
                    iou_V = np.mean(iou_valSet)
                    loss_V = np.mean(loss_valSet)
                    
                    # Summarize important details and add it to the directory
                    x,y,z = sess.run([v_l,v_a,v_I],feed_dict={v_loss:loss_V,v_iou:iou_V,v_acc:acc_V})
                    val_summary_writer.add_summary(x,global_step_value)
                    val_summary_writer.add_summary(y,global_step_value)
                    val_summary_writer.add_summary(z,global_step_value)
                    
                    print ("\n---------------------------------------------------------------------------- validation accuracy: %.3f ----------------------------------------------------------------------" %(acc_V))
                    print ("---------------------------------------------------------------------------- validation IOU: %.3f ---------------------------------------------------------------------------" %(iou_V))
                    print ("---------------------------------------------------------------------------- Cross Entropy loss: %.3f -------------------------------------------------------------------------" %(loss_V)) 
                                        
                    # Note down performance in text file                    
                    print ("\n---------------------------------------------------------------------------- Saving the metrics in the file ------------------------------------------------------------------\n") 
                    iu_File.write("accuracy for model %d is = %f" %(i,acc_V))
                    iu_File.write("\n")
                    iu_File.write("iou for model %d is = %f" %(i,iou_V))
                    iu_File.write("\n")
                    iu_File.write("ce_loss for model %d is = %f" %(i,loss_V))
                    iu_File.write("\n\n")                      

                    # Check when to stop training and go for testing
                    if not mean_CELoss_valSet:
                        print ("\nEmpty checkpoint directory, keeping the first model ###### training contd...\n")                        
                        mean_CELoss_valSet[i] = loss_V
                    else:
                        if all(v > loss_V for k,v in mean_CELoss_valSet.items()):
                            mean_CELoss_valSet[i] = loss_V
                            print ("\n--------------------------------------------------------------------- Continuing training -----------------------------------------------------------------------------\n")                        
                            overfit = 0                    
                        else:
                            overfit = overfit +1 
                            print ("\n------------------------------------------------------------------------- Validation dropping at %d step -------------------------------------------------------------------\n"%(overfit))                                               
                    if overfit == 50:
                        model_to_test =  (min(mean_CELoss_valSet.items(), key=lambda x: x[1])[0])
                        print ("\n##############################  The testing model is %d with meam cross entropy %.3f" %(model_to_test,mean_CELoss_valSet.get(model_to_test)))
                        print ("\n##############################  Stopping training as no more improvement in segmentation is possible")
                        print ("\n##############################  Proceed for testing with the model with best validation score")
                        iu_File.close()
                        break
                    

        finally:            
            sess.close()
    
    return model_to_test

############################################################################################################# Train and validation over ###############################################################################################################
#######################################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################################

            
if __name__ == '__main__':
    train_validate()