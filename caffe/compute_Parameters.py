import glob
import sys
import re
import cv2
import numpy as np
from numpy import prod, sum

sys.path.append('/root/src/caffe/python')
import caffe

caffe.set_mode_gpu()
iteration = 370000
net = caffe.Net('/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/resnet34.prototxt', '/home/bharadwaj/ImageSegmentation/data/streetObjects2Data/model/snapshot_iter_' + str(iteration) + '.caffemodel', caffe.TEST)
print ("\n\n Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()])))