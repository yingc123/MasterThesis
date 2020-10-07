# Binary Mask Generation in 3D+T Microscopy Images using Convolutional Neural Networks and Temporal Smoothness Constraints

We proposed a method combining **convolution neural network** and **spatiotemporal smoothing techniques** used for semantic segmentation of the 4D *Drosophila* and *Arabidopsis* embryos datasets. The method, 3D U-net + STP, segments the background from the foreground regions for a temporal sequence of volumetric images, it can automatically analyze large scale datasets and replace the currently manually performed masking steps. Furthermore, the generated masks of 3D U-net +STP are smoothly changing in both the spatial and temporal domains.

![3D U-net + STP (spatiotemporal postprocessing)](https://github.com/yingc123/MasterThesis/blob/master/3dunet_smoothing.png)

## Data preprocessing

#### Downsampling
The size of the datasets is quite large, especially for 4D datasets. Thus, we downscale the spatial dimension to balance the segmentation accuracy and computational complexity. We set the resolution of the cross-section of the dataset to be 256 × 256, while the depth (Z-axis) and the time axis remain unchanged.

#### Image enhancement
Furthermore, due to the transparency of the tissues and limited illumination requirement of the specimen, the images from microscopy might be low contrast. Some
[image processing techniques](https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html) can be employed to improve or adjust the contrast of the whole 3D images, like histogram equalization, contrast stretching.

## 2D [U-net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) as baseline
Code is partly adapted from [Git](https://github.com/zhixuhao/unet)
<p align="center">
  <img width="460" height="300" src="https://github.com/yingc123/MasterThesis/blob/master/2D/u-net-architecture.png">
</p>

Data are augmented in several ways, rotation, mirroring, shift, shear, and zoom, thus few training samples are needed to teach the U-net and produce robust result.
The [weighted cross-entropy](https://link.springer.com/article/10.1007/s10462-020-09854-1) and [Dice coefficient](https://link.springer.com/article/10.1007/s10462-020-09854-1) are used as loss function and metric, the weighted crossentropy is critical for the unbalanced images learning, the Dice coefficient is used to gauge the similarity of two samples.

## [3D U-net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)
Code is partly developed on the basis of the preliminary work by Dennis Eschweiler ([paper link](https://ieeexplore.ieee.org/document/8759242))
<p align="center">
  <img width="460" height="300" src="https://github.com/yingc123/MasterThesis/blob/master/3D/3dunet.png">
</p>

## Smoothing
Smoothing idea is based on the concept Surface of Interest ([paper link](https://www.nature.com/articles/nmeth.3648/))

## Datasets 

#### *Arabidopsis Theliana* & *Drosophila Melanogaster*
There are two kinds of 4D embryo datasets used in this thesis. One is *Arabidopsis thaliana* (thale cress), it is densely filled inside and thus only has the outer boundary. The other one is *Drosophila melanogaster* (fruit fly), it is hollow inside and has both inner and outer boundaries.

Following is the *Arabidopsis* embryos datasets in different time frames.
![*Arabidopsis* embyro growth](https://github.com/yingc123/MasterThesis/blob/master/Datasets/arabi_growth.png)

These images represent the segmentation types.

*Arabidopsis*
![*Arabidopsis* embyro segmentation](https://github.com/yingc123/MasterThesis/blob/master/Datasets/arabi_process.png)

*Drosophila*
![*Drosophila* embyro segmentation](https://github.com/yingc123/MasterThesis/blob/master/Datasets/dro_process.png)

#### Ground Truth generation for *Drosophila Melanogaster*
Manual Masking for cropped 3D Images: [Matlab](https://github.com/stegmaierj/CellShapeAnalysis/tree/master/MaskGeneration)

Semi-Automatic Generation of Tight Binary Masks and Non-Convex Isosurfaces for Quantitative Analysis of 3D Biological Samples ([paper link](https://arxiv.org/abs/2001.11469))
