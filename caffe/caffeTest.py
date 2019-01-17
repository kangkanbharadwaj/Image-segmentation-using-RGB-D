import glob
import sys
import re
import cv2
import numpy as np

sys.path.append('/root/src/caffe/python')
import caffe

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


caffe.set_mode_gpu()
iteration = 415000

imgDir = glob.glob("/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/val/*.jpg")
net = caffe.Net('/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/resnet34.prototxt', '/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/snapshots/snapshot_iter_' + str(iteration) + '.caffemodel', caffe.TEST)

for images in sorted(imgDir, key=stringSplitByNumbers):
    
    img = cv2.imread (images)    
    imageName = images.split('/')[7].split('.')[0]
    #depth = cv2.imread("/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/val/%s.exr" %(imageName), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)    
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
    #stackImg = np.dstack((norm_image, depth))
    in_image = norm_image.transpose((2,0,1))    
    in_image = np.expand_dims(in_image, axis=0)
    net.blobs['data'].data[...] = in_image
    net.forward()         
    net_out = net.blobs['score'].data       
    out_im = np.swapaxes(net_out, 1,3)  
    out_im = np.swapaxes(out_im, 1,2)  
    out_im = np.squeeze(out_im, axis=0) 
    out_im = out_im.argmax(axis=2)         
    img = np.uint8(out_im)        
    cv2.imwrite("/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/segmentResults/masks/%s.png" %(imageName), img)    
