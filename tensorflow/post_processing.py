import cv2
import glob
import numpy as np
from PIL import Image
import re

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

# Concatenate images to form a single image
def patch_2_FullImg(predDir=None,tgtDir=None):
    print ("\n ---------------------------------------------------------------------- Stitching patches to make the original image ---------------------------------------------------------------------------")
    subDirs = glob.glob(predDir+"*/")    
    for dirs in subDirs:       
        finImgName = dirs.split("/")[7]        
        patches = sorted(glob.glob(dirs+"*"), key=stringSplitByNumbers)       
        count = 1
        i = 0
        j = 0

        finImg = Image.new('RGB', (512, 512))
        for imgs in sorted(patches, key=stringSplitByNumbers):     
            img = Image.open(imgs)             
            if count<3:        
                finImg.paste(img,(i,j))
                i = i+256     
                count = count+1
            if count>2:
                j = j+256
                i = 0
                count = 1

        finImg.save(tgtDir+str(finImgName)+'.png') 


# Replace ignored labels in prediction by corresponding class values from original masks
def replace_ignoreLabel(orgDir=None,predDir=None,modDir=None):
    print ("\n ------------------------------------------------------------------------- Working on replacing unlearnt labels ---------------------------------------------------------------------------")
    orig_Label = glob.glob(orgDir+"/*.png")
    for masks in orig_Label:   
        mask_Name = masks.split("/")[len(masks.split("/"))-1]
        orig_mask = cv2.imread(masks)    
        pred_mask = cv2.imread(predDir+"/%s" %(mask_Name))    
        height = orig_mask.shape[0]
        width = orig_mask.shape[1]    
        for i in range(0,height):
            for j in range (0,width):             
                if orig_mask[i,j][0]==19 and orig_mask[i,j][1]==19 and orig_mask[i,j][2]==19:
                    pred_mask[i,j] = [19, 19, 19]  
        cv2.imwrite(modDir+'/%s' %(mask_Name),pred_mask)
        

# Convert label PNGs to RGB PNGs
def colorMasks(tgtDir=None,RGBDir=None):
    print ("\n ------------------------------------------------------------------------- Coloring the classes for better visualization --------------------------------------------------------------")
    pngDir = glob.glob(tgtDir+"/*.png")
    for labels in pngDir:
        label = cv2.imread(labels)
        labelName = labels.split('/')[len(labels.split('/'))-1].split('.')[0]
        height = label.shape[0]
        width = label.shape[1]

        for i in range(0,height):
            for j in range (0,width): 
                if label[i,j][0]==0 and label[i,j][1]==0 and label[i,j][2]==0:
                    label[i,j] = [150, 150, 150]       
                if label[i,j][0]==1 and label[i,j][1]==1 and label[i,j][2]==1:
                    label[i][j] = [100,100,0] 
                if label[i,j][0]==2 and label[i,j][1]==2 and label[i,j][2]==2:
                    label[i][j] = [100,0,100] 
                if label[i,j][0]==3 and label[i,j][1]==3 and label[i,j][2]==3:
                    label[i][j] = [14,70,127]
                if label [i,j][0]==4 and label[i,j][1]==4 and label[i,j][2]==4:
                    label[i][j] = [50,255,50]
                if label[i,j][0]==5 and label[i,j][1]==5 and label[i,j][2]==5:
                    label[i][j] = [0,120,0]
                if label[i,j][0]==6 and label[i,j][1]==6 and label[i,j][2]==6:
                    label[i][j] = [19,123,218]
                if label[i,j][0]==7 and label[i,j][1]==7 and label[i,j][2]==7:
                    label[i][j] = [87,85,21]  
                if label[i,j][0]==8 and label[i,j][1]==8 and label[i,j][2]==8:
                    label[i][j] = [126,126,201]
                if label[i,j][0]==9 and label[i,j][1]==9 and label[i,j][2]==9:
                    label[i][j] = [31,71,163] 
                if label[i,j][0]==10 and label[i,j][1]==10 and label[i,j][2]==10:
                    label[i][j] = [201,126,126]
                if label[i,j][0]==11 and label[i,j][1]==11 and label[i,j][2]==11:
                    label[i][j] = [71,31,163]
                if label[i,j][0]==12 and label[i,j][1]==12 and label[i,j][2]==12:
                    label[i][j] = [255,34,37]
                if label[i,j][0]==13 and label[i,j][1]==13 and label[i,j][2]==13:
                    label[i][j] = [71,47,63] 
                if label[i,j][0]==14 and label[i,j][1]==14 and label[i,j][2]==14:
                    label[i][j] = [67,67,67] 
                if label[i,j][0]==15 and label[i,j][1]==15 and label[i,j][2]==15:
                    label[i][j] = [255,255,0]
                if label[i,j][0]==16 and label[i,j][1]==16 and label[i,j][2]==16:
                    label[i][j] = [255,0,255]
                if label[i,j][0]==17 and label[i,j][1]==17 and label[i,j][2]==17:
                    label[i][j] = [0,255,255]
                if label[i,j][0]==18 and label[i,j][1]==18 and label[i,j][2]==18:
                    label[i][j] = [142,193,255]
                if label[i,j][0]==19 and label[i,j][1]==19 and label[i,j][2]==19:
                    label[i][j] = [104,78,91]
                

        cv2.imwrite(RGBDir+'/%s.png' %(labelName),label)
    
    