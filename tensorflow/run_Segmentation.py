import argparse
from train_validate import *
from test import *

def argparse_setup():

    parser = argparse.ArgumentParser(prog='IMAGE SEGMENTATION', usage='%(prog)s [options]', description='A set of parameters to run Image segmentation', epilog='Please refer o the readme.txt for better understandong of how to run the program')
    
    parser.add_argument("--MODEL", type=str, help="Which model do you want to use?", nargs='?', default=ResNet50)
    parser.add_argument("--N_CLASS", type=int, help="Number of classes for segmentation",  nargs='?', default=20)
    parser.add_argument("--IMG_HEIGHT", type=int, help="Height of image", nargs='?', default= 0)
    parser.add_argument("--IMG_WIDTH", type=int, help="Width of image", nargs='?', default=0)
    parser.add_argument("--CROP_HEIGHT", type=int, help="Height of the cropped patch",  nargs='?', default=256)
    parser.add_argument("--CROP_WIDTH", type=int, help="Width of the cropped patch", nargs='?', default=256)
    parser.add_argument("--CLS_BALANCE", type=str, help="Do you want to balance classes and ignore label?", nargs='?', choices=['True', 'False'], default='True')
    parser.add_argument("--AUGMENTATION", type=str, help="Augmentation enabled/disabled", nargs='?', choices=['True', 'False'], default='False')
    parser.add_argument("--CROP", type=str, help="Croppping required (hardcoded size is 256X256)", nargs='?', choices=['True', 'False'], default='False')
    parser.add_argument("--CLS_BALANCE_TYPE", type=str, help="What kind of class balancing do you need?", nargs='?', choices=['cls_weighing', 'median_frequency', 'weigh_equal_importance'], default='weigh_equal_importance')
    parser.add_argument("--OPTIMIZER", type=str, help="Optimizer to use", nargs='?', choices=['Adam', 'RMSProp', 'SGD'], default='Adam')
    parser.add_argument("--LR", type=float, help="Learning rate", nargs='?', default=0.0001)
    parser.add_argument("--DECAY", type=float, help="Decay the initial learning rate in every prefererd number of steps", nargs='?', default=0.90)
    parser.add_argument("--MOMENTUM", type=float, help="Momentum of decay", nargs='?', default=0.90)
    parser.add_argument("--EPSILON", type=float, help="A small constant for numerical stability", nargs='?', default=0.1)
    parser.add_argument("--BETA1", type=float, help="The exponential decay rate for the 1st moment estimates", nargs='?', default=0.9)
    parser.add_argument("--EPOCH", type=int, help="Number of iterations to complete one dataset", nargs='?', default=4894)
    parser.add_argument("--BUFFER_SIZE", type=int, help="Buffer size for shuffling", nargs='?', default=5000)
    parser.add_argument("--BATCH_SIZE", type=int, help="Mini Batch size for train data", nargs='?', default=1)
    parser.add_argument("--TRAINDIR", type=str, help="Directory that holds train data")
    parser.add_argument("--VALDIR", type=str, help="Directory that holds validation data")
    parser.add_argument("--TESTDIR", type=str, help="Directory that holds test data")      
    parser.add_argument("--TARGETDIR", type=str, help="Directory that holds stitched patches from single resolution image") 
    parser.add_argument("--PREDDIR", type=str, help="Directory that holds predicted test data")
    parser.add_argument("--MODPREDDIR", type=str, help="Directory that holds predicted test data replaced with unlearnt label in the right locations")
    parser.add_argument("--RGBDIR", type=str, help="Directory that holds colored masks from test set")
    parser.add_argument("--TRAIN_LOGDIR", type=str, help="Directory that holds train logs")
    parser.add_argument("--VAL_LOGDIR", type=str, help="Directory that holds validation logs")
    parser.add_argument("--SNAPSHOTS_DIR", type=str, help="Directory that holds saved checkpoints/models/snapshots")
    parser.add_argument("--TEXTFILE_DIR", type=str, help="Directory to save a text file that holds details on validation set")
    parser.add_argument("--MAX_ITERATIONS", type=int, help="Train the model until the max iteration", default='2000000')
    parser.add_argument("--MAX_TO_KEEP", type=int, help="Number of saved models to be kept in the diretory (The old modesl beyond this number will get deleted)", default='1000')
    parser.add_argument("--BREAK_POINT", type=int, help="Save a model in each perferred number of iterations", nargs='?', default='5000')
  
    return parser

def main():
    #Fetch the arguments    
    parser = argparse_setup()
    args = parser.parse_args()     
    
    # Train and validate the model
    model_to_test =    train_validate(args.MODEL,args.N_CLASS,args.IMG_HEIGHT,args.IMG_WIDTH,args.CROP_HEIGHT,args.CROP_WIDTH,args.CLS_BALANCE,args.CLS_BALANCE_TYPE,
                                      args.OPTIMIZER,args.LR,args.DECAY,args.MOMENTUM,args.EPSILON,args.BETA1,args.EPOCH,args.BUFFER_SIZE,args.BATCH_SIZE,args.TRAINDIR,args.VALDIR,
                                      args.TRAIN_LOGDIR,args.VAL_LOGDIR,args.SNAPSHOTS_DIR,args.TEXTFILE_DIR,args.MAX_ITERATIONS,args.MAX_TO_KEEP,args.BREAK_POINT,args.AUGMENTATION,args.CROP)
    # Test the best model
    test(args.MODEL,args.IMG_HEIGHT,args.IMG_WIDTH,args.BATCH_SIZE,args.PREDDIR,args.MODPREDDIR,args.RGBDIR,args.TARGETDIR,args.SNAPSHOTS_DIR,
         args.TESTDIR,args.TEXTFILE_DIR,args.N_CLASS,model_to_test)
    
    
if __name__ == '__main__':
    main()
