import glob
import re
import os

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

models = glob.glob("/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/snapshots/*.caffemodel")

for caffeModels in sorted(models, key=stringSplitByNumbers):    
    modelName = caffeModels.split('/')[7]  
    #print modelName
    cmd = "/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/train.sh %s" %(modelName)
    os.system(cmd) 
