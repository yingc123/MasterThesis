# Binary Mask Generation in 3D+T Microscopy Images using Convolutional Neural Networks and Temporal Smoothness Constraints

We proposed a method combining convolution neural network and spatiotemporal smoothing techniques used for semantic segmentation of the 4D Drosophila and Arabidopsis embryos datasets. The method, 3D U-net +STP, segments the background from the foreground regions for a temporal sequence of volumetric images, it can automatically analyze large scale datasets and replace the currently manually performed masking steps. Furthermore, the generated masks of 3D U-net +STP are smoothly changing in both the spatial and temporal domains.

![3D U-net + STP (spatiotemporal postprocessing)](https://github.com/yingc123/MasterThesis/blob/master/3dunet_smoothing.png)


## Preprocessing (Ground Truth generation for *Drosophila Melanogaster*)
Manual Masking for cropped 3D Images: https://github.com/stegmaierj/CellShapeAnalysis/tree/master/MaskGeneration
[Semi-Automatic Generation of Tight Binary Masks and Non-Convex Isosurfaces for Quantitative Analysis of 3D Biological Samples](https://arxiv.org/abs/2001.11469)

## 2D U-net as baseline
Unet implementation in Tensorflow+keras: https://github.com/zhixuhao/unet
