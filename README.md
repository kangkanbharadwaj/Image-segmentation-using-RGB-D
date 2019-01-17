# Image-segmentation-using-RGB-D
Learning depth-based semantic segmentation of street scenes

## Abstract

This work addresses multi-class semantic segmentation of street scenes by exploring depth information with RGB data. Our dataset comprises of street images from Berlin taken from four different camera angles and scanned using a laser scanner and later processed to create the depth images from 3D point clouds by projection. Our work also proposes an architecture model comprising of a Residual Network as an encoder and a UNet decoder for the Berlin set that learns good quality feature representation. We achieve a mean accuracy of 58.35%, mean pixel accuracy of 94.36% and mean IOU (Intersection over Union) of 51.91% on the test set. We further analyze the benefits that the model ex- hibits on certain classes when trained including depth to the RGB data with that of the model based only on RGB information. An alternative approach of feeding the depth information using a separate encoder was carried out to study the performance variation in segmentation and if it can bring any significant hike to it’s quality. And finally we draw a performance contrast of our network to one of the state-of-the-art models on our dataset.

## Introduction

<img style="border: 1px solid grey" src="images/1.png" alt="image segmentation vs semantic segmentation" width="500" height="400"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img style="border: 1px solid grey" src="images/2.png" alt="image segmentation using depth" width="500" height="350"/>


## Motivation

<img style="border: 1px solid grey" src="images/3.png" alt="image segmentation vs semantic segmentation" width="700" height="500"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


## Goals

1.  Achieve quality segmentation using RGB-D
2.  Comparison study of RGB-D to RGB segmentation
3.  Explore alternative approach to feed depth
4.  Compare our model to state-of-the-art <br />
<br />

## Approach (Data acquisition)

<img style="border: 1px solid grey" src="images/4.png" alt="image segmentation vs semantic segmentation" width="700" height="500"/> 
<br />
<br />

### List of labels to work with

<img style="border: 1px solid grey" src="images/5.png" alt="image segmentation vs semantic segmentation" width="700" height="500"/> 
<br />
<br />

## Approach (Architecture ResNet-34 fused with UNet decoder)

<img style="border: 1px solid grey" src="images/my_resnet.jpg" alt="image segmentation vs semantic segmentation" width="600" height="800"/>
<br />
<br />

## Experiments and Results

### ResNet-34 vs ResNet-50 vs ResNet-101

<img style="border: 1px solid grey" src="images/6.png" alt="image segmentation vs semantic segmentation" width="600" height="400"/>
<br />

<img style="border: 1px solid grey" src="images/7.png" alt="image segmentation vs semantic segmentation" width="900" height="500"/>
<img style="border: 1px solid grey" src="images/8.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<br />

### ResNet-34 on testset

<img style="border: 1px solid grey" src="images/9.png" alt="image segmentation vs semantic segmentation" width="600" height="400"/>
<br />

<img style="border: 1px solid grey" src="images/10.png" alt="image segmentation vs semantic segmentation" width="700" height="700"/>
<img style="border: 1px solid grey" src="images/11.png" alt="image segmentation vs semantic segmentation" width="700" height="700"/>
<br />

### RGB-D vs RGB segmentation

<img style="border: 1px solid grey" src="images/12.png" alt="image segmentation vs semantic segmentation" width="600" height="600"/>
<br />

<img style="border: 1px solid grey" src="images/13.png" alt="image segmentation vs semantic segmentation" width="900" height="600"/>
<img style="border: 1px solid grey" src="images/14.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<img style="border: 1px solid grey" src="images/15.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<img style="border: 1px solid grey" src="images/16.png" alt="image segmentation vs semantic segmentation" width="900" height="350"/>
<br />
<br />


## Reproducing FuseNet approach on ResNet-34 fused with UNet decoder

### Early Fusion vs Late Fusion

<img style="border: 1px solid grey" src="renset_fusion.jpg" alt="image segmentation vs semantic segmentation" width="600" height="400"/>
<br />

<img style="border: 1px solid grey" src="images/17.png" alt="image segmentation vs semantic segmentation" width="600" height="600"/>
<br />

<img style="border: 1px solid grey" src="images/18.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<img style="border: 1px solid grey" src="images/19.png" alt="image segmentation vs semantic segmentation" width="900" height="350"/>
<img style="border: 1px solid grey" src="images/20.png" alt="image segmentation vs semantic segmentation" width="900" height="350"/>
<br />

### Early Fusion (smaller model) vs Late Fusion

<img style="border: 1px solid grey" src="images/21.png" alt="image segmentation vs semantic segmentation" width="600" height="600"/>
<br />

<img style="border: 1px solid grey" src="images/22.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<br />


### Early Fusion vs Late Fusion1 vs Late Fusion2


<img style="border: 1px solid grey" src="images/23.png" alt="image segmentation vs semantic segmentation" width="600" height="600"/>
<br />

<img style="border: 1px solid grey" src="images/24.png" alt="image segmentation vs semantic segmentation" width="900" height="500"/>
<img style="border: 1px solid grey" src="images/25.png" alt="image segmentation vs semantic segmentation" width="900" height="500"/>
<img style="border: 1px solid grey" src="images/26.png" alt="image segmentation vs semantic segmentation" width="900" height="350"/>
<img style="border: 1px solid grey" src="images/27.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<br />
<br />

## ResNet-34 vs DDNet

### Incorporating dense connections in UNet decoder

<img style="border: 1px solid grey" src="images/dense_decoder.jpg" alt="image segmentation vs semantic segmentation" width="450" height="500"/>
<br />

### Results


<img style="border: 1px solid grey" src="images/29.png" alt="image segmentation vs semantic segmentation" width="400" height="120"/>
<br />

<img style="border: 1px solid grey" src="images/30.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<img style="border: 1px solid grey" src="images/31.png" alt="image segmentation vs semantic segmentation" width="900" height="200"/>
<br />


<img style="border: 1px solid grey" src="images/32.png" alt="image segmentation vs semantic segmentation" width="400" height="120"/>
<br />

<img style="border: 1px solid grey" src="images/33.png" alt="image segmentation vs semantic segmentation" width="900" height="550"/>
<img style="border: 1px solid grey" src="images/34.png" alt="image segmentation vs semantic segmentation" width="900" height="200"/>
<br />


## Swapping encoders and decoders of both architectures

### DPDB-UNet vs Res-DDNet


<img style="border: 1px solid grey" src="images/35.png" alt="image segmentation vs semantic segmentation" width="400" height="120"/>
<br />

<img style="border: 1px solid grey" src="images/36.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<img style="border: 1px solid grey" src="images/37.png" alt="image segmentation vs semantic segmentation" width="900" height="400"/>
<img style="border: 1px solid grey" src="images/38.png" alt="image segmentation vs semantic segmentation" width="900" height="400"/>
<img style="border: 1px solid grey" src="images/39.png" alt="image segmentation vs semantic segmentation" width="900" height="400"/>
<br />
<br />


## DPDB-UNet variations

### Results



<img style="border: 1px solid grey" src="images/40.png" alt="image segmentation vs semantic segmentation" width="400" height="120"/>
<br />

<img style="border: 1px solid grey" src="images/41.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<img style="border: 1px solid grey" src="images/42.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<img style="border: 1px solid grey" src="images/43.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<img style="border: 1px solid grey" src="images/44.png" alt="image segmentation vs semantic segmentation" width="900" height="450"/>
<br />
<br />

## Conclusions

1.  Proposed architecture model learns good quality feature representation
2.  Depth can deliver performance hike
3.  Late fusion is counter-productive
4.  On Berlin set, ResNet-34 performs better than DDNet




