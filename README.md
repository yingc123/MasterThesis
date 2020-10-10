# Binary Mask Generation in 3D+T Microscopy Images using Convolutional Neural Networks and Temporal Smoothness Constraints

We proposed a method combining **convolution neural network** and **spatiotemporal smoothing techniques** used for semantic segmentation of the 4D *Drosophila* and *Arabidopsis* embryos datasets. The method, 3D U-net + STP, segments the background from the foreground regions for a temporal sequence of volumetric images, it can automatically analyze large scale datasets and replace the currently manually performed masking steps. Furthermore, the generated masks of 3D U-net +STP are smoothly changing in both the spatial and temporal domains.

![3D U-net + STP (spatiotemporal postprocessing)](https://github.com/yingc123/MasterThesis/blob/master/3dunet_smoothing.png)

## Results
For a qualitative assessment, we employed the 3D U-net +STP on all three Drosophila samples. Fig. 5.15 to 5.17 shows the segmentation results for our three samples.

![segmentation results 1](https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/1.PNG)

![segmentation results 2](https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/2.PNG)

![segmentation results 3](https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/3.PNG)

## Methods
### Data preprocessing

#### Downsampling
The size of the datasets is quite large, especially for 4D datasets. Thus, we downscale the spatial dimension to balance the segmentation accuracy and computational complexity. We set the resolution of the cross-section of the dataset to be 256 × 256, while the depth (Z-axis) and the time axis remain unchanged.

#### Image enhancement
Furthermore, due to the transparency of the tissues and limited illumination requirement of the specimen, the images from microscopy might be low contrast. Some
[image processing techniques](https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html) can be employed to improve or adjust the contrast of the whole 3D images, like histogram equalization, contrast stretching.
![Cross section of images](https://github.com/yingc123/MasterThesis/blob/master/Datasets/DataPreprocessing.PNG)


### 2D [U-net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) as baseline
Code is partly adapted from [Git](https://github.com/zhixuhao/unet)
<p align="center">
  <img width="460" height="300" src="https://github.com/yingc123/MasterThesis/blob/master/2D/u-net-architecture.png">
</p>

Data are augmented in several ways, rotation, mirroring, shift, shear, and zoom, thus few training samples are needed to teach the U-net and produce robust result.
The [weighted cross-entropy](https://link.springer.com/article/10.1007/s10462-020-09854-1) and [Dice coefficient](https://link.springer.com/article/10.1007/s10462-020-09854-1) are used as loss function and metric, the weighted crossentropy is critical for the unbalanced images learning, the Dice coefficient is used to gauge the similarity of two samples.

### 3D U-net + STP
#### [3D U-net](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)
Code is partly developed on the basis of the preliminary work by Dennis Eschweiler ([paper link](https://ieeexplore.ieee.org/document/8759242))
<p align="center">
  <img width="460" height="300" src="https://github.com/yingc123/MasterThesis/blob/master/3D/3dunet.png">
</p>

The following parameters are also identical to 3D U-net: the size of the convolution operation, the max-pooling operation, and the deconvolution operation, the activation function after each layer. The difference is that we use zero-padding to ensure the output has the same size as input, just like map pixels to pixels.

The input to the network is a sequence of grayscale images and the output of whole network architecture is the same size corresponding predicted mask. To increase the predicted accuracy and reduce computational complexity, **batch normalization** and seamless tiling strategy are applied so that it can work on arbitrary large volumes.

Loss function is **weighted cross-entropy loss**, it is necessary for unbalanced-classes image, e.g. the boundary mask. The weights are calculated based on the percentage of the two labels in the whole patch, i.e. the weight of foreground is the ratio of the total number of pixels to foreground pixels and the same to the weight of background. Besides, a rectified linear unit (**ReLu**) is applied in each convolutional layer, this limits the occurrence of negative values.

#### Denoising
The second subsystem is named denoising, because sometimes the background pixels are misidentified as boundary or foreground labels or just the opposite, boundary are considered as background. Due to the prior knowledge about the datasets, we can design some algorithms to reduce these inaccurate classifications.
![Noise removal](https://github.com/yingc123/MasterThesis/blob/master/smoothing/denoising_1.png)

Standardize the input of the denoising module. Morphological transformations use [XPIWIT](https://academic.oup.com/bioinformatics/article/32/2/315/1744077).

The idea of denoising algorithm is to remove incorrect and unimportant pixels, and leave only the pixels that have enough information to represent whole surface.

![Filtering process](https://github.com/yingc123/MasterThesis/blob/master/smoothing/denoising_2.png)

Filtering these points and only leave 2 points, one for the inner boundary and the other one for the outer boundary, store the coordinate value of points with the order of increasing angle value; Finally combining all stored points for each direction and connect all slices together, we can get the denoised boundary images.

#### Smoothing
Based on the idea, Surface of Interest ([paper link](https://www.nature.com/articles/nmeth.3648/)), that we can map the ordered point cloud lying on a surface to a surface map according to their value in cylindrical coordinate system (See followed picture). It reduces data from 3D to 2D that decrease data size and processing time. 
<p align="center">
  <img width="700" height="500" src="https://github.com/yingc123/MasterThesis/blob/master/smoothing/denoising_4.png">
</p>

**The intensity value of the 2D surface map is the distance of the surface pixels to the central line, for each slice is the distance of boundary pixels to the center.** In other words, we map the surface pixels to a cylinder coordinate system and unroll the surface to a 2D map, the value of each pixels represents the distance of corresponding point to the central line.

Each surface of a 3D volumetric data can be generated a surface map. Therefore, a volumetric image of *Drosophila* produces two surface maps, respectively **inner surface map and outer surface map**. Furthermore, different time frames correspond to different inner and outer surface maps. These surface maps can be concatenated together with increasing time frames, details in next figure.
<p align="center">
  <img width="484" height="360" src="https://github.com/yingc123/MasterThesis/blob/master/smoothing/smoothing_1.png">
</p>

## Evaluation
#### U-net vs. 3D U-net

#### 3D U-net vs. 3D U-net + STP

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
