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
4.  Compare our model to state-of-the-art  


## Approach (Data acquisition)

<img style="border: 1px solid grey" src="images/4.png" alt="image segmentation vs semantic segmentation" width="700" height="500"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs
