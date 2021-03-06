# Binary Mask Generation in 3D+T Microscopy Images using Convolutional Neural Networks and Temporal Smoothness Constraints

We proposed a method combining **convolution neural network** and **spatiotemporal smoothing techniques** used for semantic segmentation of the 4D *Drosophila* and *Arabidopsis* embryos datasets. The method, 3D U-net + STP, segments the background from the foreground regions for a temporal sequence of volumetric images, it can automatically analyze large scale datasets and replace the currently manually performed masking steps. Furthermore, the generated masks of 3D U-net +STP are smoothly changing in both the spatial and temporal domains.

![3D U-net + STP (spatiotemporal postprocessing)](https://github.com/yingc123/MasterThesis/blob/master/3dunet_smoothing.png)

## Results
For a qualitative assessment, we employed the 3D U-net +STP on all three Drosophila samples. Fig. 5.15 to 5.17 shows the segmentation results for our three samples.

<p align="center">
  <img width="763" height="877" src="https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/1.PNG">
</p>

<p align="center">
  <img width="770" height="874" src="https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/2.PNG">
</p>

<p align="center">
  <img width="764" height="878" src="https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/3.PNG">
</p>


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

The idea of denoising algorithm is to remove incorrect and unimportant pixels, and leave only the pixels that have enough information to represent whole surface. To generate the radial lines, we used the [Bresenham’s line algorithm](https://de.wikipedia.org/wiki/Bresenham-Algorithmus).

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

The surface map shows dense information about the location of the surface pixels for all time frames (Fig. 4.16 separate to 2D slices along time axis). The surface
map facilitates the observation of the variations of the surface pixels, simple to find its location in original volumetric data. More importantly, adjacent points (both in
spatial and temporal domain) are stored close to each other, smoothing can then be applied by using the surface map data.

<p align="center">
  <img width="575" height="817" src="https://github.com/yingc123/MasterThesis/blob/master/Results%26Evaluation/SurfaceMap.PNG">
</p>

In image processing, the aim of smoothing is to remove noise or other fine-scale structures/rapid changes. Two techniques are employed for smoothing process in our thesis, convolution (image filtering) and linear regression.

**Image filtering for spatial smoothing**: The surface map represents the location of surface pixels, applying smoothing (Sec. 2.5) to the surface map is similar as smooth the 3D surface. Limiting the location of each point according to its neighborhood points. This step is to ensure the generating surface is a smooth surface, not a rugged one.

**Linear regression for temporal smoothing**: During the cellularization of blastoderm of Drosophila, the cell membranes gradually grow inward. There is a clear trend of growth in the data, with almost the same amount of growth over the same time interval, so linear regression is chosen to fit the model along time axis. A series of data points along the time axis are processed each time, and newly predicted data points are stored. After employing linear regression in the temporal domain, the smoothness between volumetric data of continuous time frames are assured.

Here shows results for both outer surface maps and inner surface maps, left is before smoothing and right corresponds after smoothing.

<p align="center">
  <img width="581" height="860" src="https://github.com/yingc123/MasterThesis/blob/master/smoothing/SurfaceMap1.PNG">
</p>

<p align="center">
  <img width="585" height="853" src="https://github.com/yingc123/MasterThesis/blob/master/smoothing/SurfaceMap2.PNG">
</p>

## Evaluation
#### Preprocessing is essential
Using different preprocessing methods with 3D U-net, we can found the result is better than original images predicted result.
Evaluation between different preprocessed images for *Drosophila melanogaster*, the evaluation metric is BF Score.

<div align="center">

| Foreground Mask | Dataset1     | Dataset2     | Dataset3     |
| ---------- | :-----------:  | :-----------: |:-----------: |
| Original Image    | 0.98040    | 0.93348    |  0.94025    |
| Image Applied Histogram Equlization    | 0.98280    | 0.93143   | **0.94899**    |
| Image Applied Contrast Stretching    | **0.98327**    | **0.93415**   |  0.94678   |

</div>

Evaluation between different preprocessed image for *Arabidopsis thaliana*, the evaluation metrics are recall, precision and BF Score.

<div align="center">

| Boundary Thick Mask | Recall     | Precision     | BF Score     |
| ---------- | :-----------:  | :-----------: |:-----------: |
| Original Image    | 0.94745    | 0.79866    |  0.86632    |
| Image Applied Histogram Equlization    | **0.95897**    | 0.76315   | 0.8496    |
| Image Applied Contrast Stretching    | 0.94403    | **0.81586**   |  **0.87483**   |

</div>

#### U-net vs. 3D U-net

Evaluation between 2D U-net and 3D U-net, evaluated in BF Score.

<div align="center">

|            | *Drosophila*1     | *Drosophila*2      | *Drosophila*3      |  *Arabidopsis*      |
| ---------- | :-----------:  | :-----------: |:-----------: |:-----------: |
| 2D U-net    | 0.98176    | **0.94778**    |  **0.96649**   | 0.85564   |
| 3D U-net    | **0.98327**    | 0.93145   | 0.94678   | **0.87483**   |

</div>

The input datasets are identical for both 2D U-net and 3D U-net, the result of 2D U-net can then be merged together to be volumetric images and evaluated with the 3D ground truth.

The evaluation results are shown in last table, BF Score is used as the accuracy measure to compare ground truth volume to the predicted 3D volume. The result of 2D U-net is slightly better than 3D U-net in *Drosophila* dataset2 and dataset3. However, it is difficult for 2D U-net to generate smoothed results in space if the dataset in an unregular shape, like for *Arabidopsis*, the 3D U-net outperforms than 2D U-net.

#### 3D U-net vs. 3D U-net + STP

For a qualitative assessment, we employed the 3D U-net +STP on all three *Drosophila* samples. The reults part shows the segmentation results for our three samples. The
L2 norm of the first-order derivative over space and time is used to evaluate the spatiotemporal smoothness of predicted masks.



<p align="center">
  <img src="http://chart.googleapis.com/chart?cht=tx&chl=Dist=\sum_{i=0}^{T-2}|p_{pred}^{i%2B1}-p_{pred}^i|^2%2B\sum_{i=0}^{S-2}|p_{pred}^{i%2B1}-p_{pred}^i|^2%2B\sum_{i=0}^{P-2}|p_{pred}^{i%2B1}-p_{pred}^i|^2" style="border:none;">
</p>

*P* means pixels in the surface map. *T*, *S* and *P* are respectively total numbers of time frames, slices and points. The L2 norm is to measure the similarity between frames. The smaller the L2 norm is, the smoother surface masks generated are. 

<div align="center">
  
<table border=0 cellpadding=0 cellspacing=0 width=683 style='border-collapse:
 collapse;table-layout:fixed;width:516pt'>
 <col width=115 span=2 style='mso-width-source:userset;mso-width-alt:4026;
 width:87pt'>
 <col width=151 span=2 style='mso-width-source:userset;mso-width-alt:5282;
 width:114pt'>
 <col width=151 style='mso-width-source:userset;mso-width-alt:5282;width:114pt'>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl65 width=115 style='height:14.5pt;width:87pt'>&nbsp;</td>
  <td class=xl65 width=115 style='border-left:none;width:87pt'>&nbsp;</td>
  <td class=xl65 width=151 style='border-left:none;width:114pt'>w/o
  Smoothing<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-left:none;width:114pt'>Regression+Smoothing<span
  style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-left:none;width:114pt'>Smoothing+Regression</td>
 </tr>
 <tr height=67 style='height:50.0pt'>
  <td height=67 class=xl65 width=115 style='height:50.0pt;border-top:none;
  width:87pt'>Dataset1</td>
  <td class=xl65 width=115 style='border-top:none;border-left:none;width:87pt'>Time
  Axis <br>
    Slice Axis <br>
    Point Axis <br>
    Total<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>281.66752
  <br>
    75.28298 <br>
    195.93661 <br>
    552.88711<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>294.36342
  <br>
    44.44158 <br>
    87.30024 <br>
    426.10524<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>294.37389<br>
    44.4438<br>
    87.30207<br>
    426.11976</td>
 </tr>
 <tr height=67 style='height:50.0pt'>
  <td height=67 class=xl65 width=115 style='height:50.0pt;border-top:none;
  width:87pt'>Dataset2</td>
  <td class=xl65 width=115 style='border-top:none;border-left:none;width:87pt'>Time
  Axis <br>
    Slice Axis <br>
    Point Axis <br>
    Total<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>274.19039
  <br>
    69.74792 <br>
    202.80693 <br>
    546.74524<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>216.425
  <br>
    29.54986 <br>
    48.55943 <br>
    294.53429<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>216.42063<br>
    29.55021<br>
    48.55816<br>
    294.529</td>
 </tr>
 <tr height=67 style='height:50.0pt'>
  <td height=67 class=xl65 width=115 style='height:50.0pt;border-top:none;
  width:87pt'>Dataset3</td>
  <td class=xl65 width=115 style='border-top:none;border-left:none;width:87pt'>Time
  Axis <br>
    Slice Axis <br>
    Point Axis <br>
    Total<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>305.60828
  <br>
    57.52826 <br>
    211.37414 <br>
    574.51068<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>281.73747
  <br>
    32.92505 <br>
    112.81207 <br>
    427.47459<span style='mso-spacerun:yes'> </span></td>
  <td class=xl65 width=151 style='border-top:none;border-left:none;width:114pt'>281.74963<br>
    32.92517<br>
    112.81266<br>
    427.48746</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=115 style='width:87pt'></td>
  <td width=115 style='width:87pt'></td>
  <td width=151 style='width:114pt'></td>
  <td width=151 style='width:114pt'></td>
  <td width=151 style='width:114pt'></td>
 </tr>
 <![endif]>
</table>

</div>

From table we can find that applying the spatiotemporal postprocessing can generate smoother surface masks comparing without STP for all three datasets. It limits the position of one point based on its surrounding points, the postprocessing algorithm makes the distance between adjacent points to be more uniform, see next table. Therefore, the generated masks from 3D U-net + STP are smoothly changed in both space and time domains.

<div align="center">

<table border=0 cellpadding=0 cellspacing=0 width=575 style='border-collapse:
 collapse;table-layout:fixed;width:430pt'>
 <col width=115 span=5 style='mso-width-source:userset;mso-width-alt:4002;
 width:86pt'>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;width:86pt'>Distance
  number<span style='mso-spacerun:yes'> </span></td>
  <td class=xl66 width=115 style='border-left:none;width:86pt'>inner dist.<span
  style='mso-spacerun:yes'> </span></td>
  <td class=xl66 width=115 style='border-left:none;width:86pt'>outer dist.<span
  style='mso-spacerun:yes'> </span></td>
  <td class=xl66 width=115 style='border-left:none;width:86pt'>inner dist.<span
  style='mso-spacerun:yes'> </span></td>
  <td class=xl66 width=115 style='border-left:none;width:86pt'>outer dist.</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl65 style='height:14.5pt'></td>
  <td colspan=2 class=xl67 width=230 style='border-right:.5pt solid black;
  width:172pt'>w/o Smoothing<span style='mso-spacerun:yes'> </span></td>
  <td colspan=2 class=xl67 width=230 style='border-right:.5pt solid black;
  width:172pt'>Spatiotemporal Smoothing</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;width:86pt'>1</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>150.43908</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>137.64019</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.46412</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.90947</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;border-top:none;
  width:86pt'>2</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>153.03513</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>125.26635</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.48918</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.89299</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;border-top:none;
  width:86pt'>3</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>150.48266</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>124.60538</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.46455</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.90254</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;border-top:none;
  width:86pt'>4</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>158.52608</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>119.58127</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.46688</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.89876</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;border-top:none;
  width:86pt'>5</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>151.43724</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>131.13379</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.53568</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.88919</td>
 </tr>
 <tr height=19 style='height:14.5pt'>
  <td height=19 class=xl66 width=115 style='height:14.5pt;border-top:none;
  width:86pt'>6</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>156.47816</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>131.3798</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>166.42547</td>
  <td class=xl66 width=115 style='border-top:none;border-left:none;width:86pt'>127.90452</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=115 style='width:86pt'></td>
  <td width=115 style='width:86pt'></td>
  <td width=115 style='width:86pt'></td>
  <td width=115 style='width:86pt'></td>
  <td width=115 style='width:86pt'></td>
 </tr>
 <![endif]>
</table>

</div>

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
