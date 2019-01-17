import OpenEXR
import Imath
from PIL import Image
import numpy as np
import os
import glob

filenames = glob.glob("/home/bharadwaj/CaffeUNet/dataset4/Berlin_data/tmp/*.exr")

for names in filenames:    
    fn = names.split("/")[7].split(".")[0]
    infile = OpenEXR.InputFile(names)       
    dw = infile.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    print("size is ")
    print(sz)

    print("header:\n")
    head=infile.header();
    for h in head:
        print("{}: {}".format(h,head[h]))
        
    data=infile.channel("Y")
    img=np.frombuffer(data, dtype=np.float32)

    nonzeromin=np.min(img[img>0.01])

    print("non-zero minimum in the file is {}".format(nonzeromin))

    img=(img-nonzeromin)
    scalefactor=255.0/(1.0-nonzeromin)
    img=img*scalefactor
    img[img<0]=0
    img=img.astype(np.uint8)

    img=img.reshape((sz[1],sz[0]))

    imgPIL = Image.fromarray(img, 'L')    
    imgPIL.save("/home/bharadwaj/CaffeUNet/dataset4/Berlin_data/tmp/%s.png" %(fn))
