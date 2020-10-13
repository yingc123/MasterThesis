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

## Evaluation
#### Preprocessing is essential
| Foreground Mask | Dataset1     | Dataset2     | Dataset3     |
| ---------- | :-----------:  | :-----------: |:-----------: |
| Original Image    | 0.98040    | 0.93348    |  0.94025    |
| Image Applied Histogram Equlization    | 0.98280    | 0.93143   | **0.94899**    |
| Image Applied Contrast Stretching    | **0.98327**    | **0.93415**   |  0.94678   |


| Boundary Thick Mask | Recall     | Precision     | BF Score     |
| ---------- | :-----------:  | :-----------: |:-----------: |
| Original Image    | 0.94745    | 0.79866    |  0.86632    |
| Image Applied Histogram Equlization    | **0.95897**    | 0.76315   | 0.8496    |
| Image Applied Contrast Stretching    | 0.94403    | **0.81586**   |  **0.87483**   |



#### U-net vs. 3D U-net

|            | Drosophila1     | Drosophila2      | Drosophila3      |  Arabidopsis      |
| ---------- | :-----------:  | :-----------: |:-----------: |:-----------: |
| 2D U-net    | 0.98176    | **0.94778**    |  **0.96649**   | 0.85564   |
| 3D U-net    | **0.98327**    | 0.93145   | 0.94678   | **0.87483**   |


#### 3D U-net vs. 3D U-net + STP


<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:x="urn:schemas-microsoft-com:office:excel"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta name="Excel Workbook Frameset">
<meta http-equiv=Content-Type content="text/html; charset=windows-1252">
<meta name=ProgId content=Excel.Sheet>
<meta name=Generator content="Microsoft Excel 15">
<link rel=File-List href="Smoothing.files/filelist.xml">
<![if !supportTabStrip]>
<link id="shLink" href="Smoothing.files/sheet001.html">

<link id="shLink">

<script language="JavaScript">
<!--
 var c_lTabs=1;

 var c_rgszSh=new Array(c_lTabs);
 c_rgszSh[0] = "Sheet1";



 var c_rgszClr=new Array(8);
 c_rgszClr[0]="window";
 c_rgszClr[1]="buttonface";
 c_rgszClr[2]="windowframe";
 c_rgszClr[3]="windowtext";
 c_rgszClr[4]="threedlightshadow";
 c_rgszClr[5]="threedhighlight";
 c_rgszClr[6]="threeddarkshadow";
 c_rgszClr[7]="threedshadow";

 var g_iShCur;
 var g_rglTabX=new Array(c_lTabs);

function fnGetIEVer()
{
 var ua=window.navigator.userAgent
 var msie=ua.indexOf("MSIE")
 if (msie>0 && window.navigator.platform=="Win32")
  return parseInt(ua.substring(msie+5,ua.indexOf(".", msie)));
 else
  return 0;
}

function fnBuildFrameset()
{
 var szHTML="<frameset rows=\"*,18\" border=0 width=0 frameborder=no framespacing=0>"+
  "<frame src=\""+document.all.item("shLink")[0].href+"\" name=\"frSheet\" noresize>"+
  "<frameset cols=\"54,*\" border=0 width=0 frameborder=no framespacing=0>"+
  "<frame src=\"\" name=\"frScroll\" marginwidth=0 marginheight=0 scrolling=no>"+
  "<frame src=\"\" name=\"frTabs\" marginwidth=0 marginheight=0 scrolling=no>"+
  "</frameset></frameset><plaintext>";

 with (document) {
  open("text/html","replace");
  write(szHTML);
  close();
 }

 fnBuildTabStrip();
}

function fnBuildTabStrip()
{
 var szHTML=
  "<html><head><style>.clScroll {font:8pt Courier New;color:"+c_rgszClr[6]+";cursor:default;line-height:10pt;}"+
  ".clScroll2 {font:10pt Arial;color:"+c_rgszClr[6]+";cursor:default;line-height:11pt;}</style></head>"+
  "<body onclick=\"event.returnValue=false;\" ondragstart=\"event.returnValue=false;\" onselectstart=\"event.returnValue=false;\" bgcolor="+c_rgszClr[4]+" topmargin=0 leftmargin=0><table cellpadding=0 cellspacing=0 width=100%>"+
  "<tr><td colspan=6 height=1 bgcolor="+c_rgszClr[2]+"></td></tr>"+
  "<tr><td style=\"font:1pt\">&nbsp;<td>"+
  "<td valign=top id=tdScroll class=\"clScroll\" onclick=\"parent.fnFastScrollTabs(0);\" onmouseover=\"parent.fnMouseOverScroll(0);\" onmouseout=\"parent.fnMouseOutScroll(0);\"><a>&#171;</a></td>"+
  "<td valign=top id=tdScroll class=\"clScroll2\" onclick=\"parent.fnScrollTabs(0);\" ondblclick=\"parent.fnScrollTabs(0);\" onmouseover=\"parent.fnMouseOverScroll(1);\" onmouseout=\"parent.fnMouseOutScroll(1);\"><a>&lt</a></td>"+
  "<td valign=top id=tdScroll class=\"clScroll2\" onclick=\"parent.fnScrollTabs(1);\" ondblclick=\"parent.fnScrollTabs(1);\" onmouseover=\"parent.fnMouseOverScroll(2);\" onmouseout=\"parent.fnMouseOutScroll(2);\"><a>&gt</a></td>"+
  "<td valign=top id=tdScroll class=\"clScroll\" onclick=\"parent.fnFastScrollTabs(1);\" onmouseover=\"parent.fnMouseOverScroll(3);\" onmouseout=\"parent.fnMouseOutScroll(3);\"><a>&#187;</a></td>"+
  "<td style=\"font:1pt\">&nbsp;<td></tr></table></body></html>";

 with (frames['frScroll'].document) {
  open("text/html","replace");
  write(szHTML);
  close();
 }

 szHTML =
  "<html><head>"+
  "<style>A:link,A:visited,A:active {text-decoration:none;"+"color:"+c_rgszClr[3]+";}"+
  ".clTab {cursor:hand;background:"+c_rgszClr[1]+";font:9pt Arial;padding-left:3px;padding-right:3px;text-align:center;}"+
  ".clBorder {background:"+c_rgszClr[2]+";font:1pt;}"+
  "</style></head><body onload=\"parent.fnInit();\" onselectstart=\"event.returnValue=false;\" ondragstart=\"event.returnValue=false;\" bgcolor="+c_rgszClr[4]+
  " topmargin=0 leftmargin=0><table id=tbTabs cellpadding=0 cellspacing=0>";

 var iCellCount=(c_lTabs+1)*2;

 var i;
 for (i=0;i<iCellCount;i+=2)
  szHTML+="<col width=1><col>";

 var iRow;
 for (iRow=0;iRow<6;iRow++) {

  szHTML+="<tr>";

  if (iRow==5)
   szHTML+="<td colspan="+iCellCount+"></td>";
  else {
   if (iRow==0) {
    for(i=0;i<iCellCount;i++)
     szHTML+="<td height=1 class=\"clBorder\"></td>";
   } else if (iRow==1) {
    for(i=0;i<c_lTabs;i++) {
     szHTML+="<td height=1 nowrap class=\"clBorder\">&nbsp;</td>";
     szHTML+=
      "<td id=tdTab height=1 nowrap class=\"clTab\" onmouseover=\"parent.fnMouseOverTab("+i+");\" onmouseout=\"parent.fnMouseOutTab("+i+");\">"+
      "<a href=\""+document.all.item("shLink")[i].href+"\" target=\"frSheet\" id=aTab>&nbsp;"+c_rgszSh[i]+"&nbsp;</a></td>";
    }
    szHTML+="<td id=tdTab height=1 nowrap class=\"clBorder\"><a id=aTab>&nbsp;</a></td><td width=100%></td>";
   } else if (iRow==2) {
    for (i=0;i<c_lTabs;i++)
     szHTML+="<td height=1></td><td height=1 class=\"clBorder\"></td>";
    szHTML+="<td height=1></td><td height=1></td>";
   } else if (iRow==3) {
    for (i=0;i<iCellCount;i++)
     szHTML+="<td height=1></td>";
   } else if (iRow==4) {
    for (i=0;i<c_lTabs;i++)
     szHTML+="<td height=1 width=1></td><td height=1></td>";
    szHTML+="<td height=1 width=1></td><td></td>";
   }
  }
  szHTML+="</tr>";
 }

 szHTML+="</table></body></html>";
 with (frames['frTabs'].document) {
  open("text/html","replace");
  charset=document.charset;
  write(szHTML);
  close();
 }
}

function fnInit()
{
 g_rglTabX[0]=0;
 var i;
 for (i=1;i<=c_lTabs;i++)
  with (frames['frTabs'].document.all.tbTabs.rows[1].cells[fnTabToCol(i-1)])
   g_rglTabX[i]=offsetLeft+offsetWidth-6;
}

function fnTabToCol(iTab)
{
 return 2*iTab+1;
}

function fnNextTab(fDir)
{
 var iNextTab=-1;
 var i;

 with (frames['frTabs'].document.body) {
  if (fDir==0) {
   if (scrollLeft>0) {
    for (i=0;i<c_lTabs&&g_rglTabX[i]<scrollLeft;i++);
    if (i<c_lTabs)
     iNextTab=i-1;
   }
  } else {
   if (g_rglTabX[c_lTabs]+6>offsetWidth+scrollLeft) {
    for (i=0;i<c_lTabs&&g_rglTabX[i]<=scrollLeft;i++);
    if (i<c_lTabs)
     iNextTab=i;
   }
  }
 }
 return iNextTab;
}

function fnScrollTabs(fDir)
{
 var iNextTab=fnNextTab(fDir);

 if (iNextTab>=0) {
  frames['frTabs'].scroll(g_rglTabX[iNextTab],0);
  return true;
 } else
  return false;
}

function fnFastScrollTabs(fDir)
{
 if (c_lTabs>16)
  frames['frTabs'].scroll(g_rglTabX[fDir?c_lTabs-1:0],0);
 else
  if (fnScrollTabs(fDir)>0) window.setTimeout("fnFastScrollTabs("+fDir+");",5);
}

function fnSetTabProps(iTab,fActive)
{
 var iCol=fnTabToCol(iTab);
 var i;

 if (iTab>=0) {
  with (frames['frTabs'].document.all) {
   with (tbTabs) {
    for (i=0;i<=4;i++) {
     with (rows[i]) {
      if (i==0)
       cells[iCol].style.background=c_rgszClr[fActive?0:2];
      else if (i>0 && i<4) {
       if (fActive) {
        cells[iCol-1].style.background=c_rgszClr[2];
        cells[iCol].style.background=c_rgszClr[0];
        cells[iCol+1].style.background=c_rgszClr[2];
       } else {
        if (i==1) {
         cells[iCol-1].style.background=c_rgszClr[2];
         cells[iCol].style.background=c_rgszClr[1];
         cells[iCol+1].style.background=c_rgszClr[2];
        } else {
         cells[iCol-1].style.background=c_rgszClr[4];
         cells[iCol].style.background=c_rgszClr[(i==2)?2:4];
         cells[iCol+1].style.background=c_rgszClr[4];
        }
       }
      } else
       cells[iCol].style.background=c_rgszClr[fActive?2:4];
     }
    }
   }
   with (aTab[iTab].style) {
    cursor=(fActive?"default":"hand");
    color=c_rgszClr[3];
   }
  }
 }
}

function fnMouseOverScroll(iCtl)
{
 frames['frScroll'].document.all.tdScroll[iCtl].style.color=c_rgszClr[7];
}

function fnMouseOutScroll(iCtl)
{
 frames['frScroll'].document.all.tdScroll[iCtl].style.color=c_rgszClr[6];
}

function fnMouseOverTab(iTab)
{
 if (iTab!=g_iShCur) {
  var iCol=fnTabToCol(iTab);
  with (frames['frTabs'].document.all) {
   tdTab[iTab].style.background=c_rgszClr[5];
  }
 }
}

function fnMouseOutTab(iTab)
{
 if (iTab>=0) {
  var elFrom=frames['frTabs'].event.srcElement;
  var elTo=frames['frTabs'].event.toElement;

  if ((!elTo) ||
   (elFrom.tagName==elTo.tagName) ||
   (elTo.tagName=="A" && elTo.parentElement!=elFrom) ||
   (elFrom.tagName=="A" && elFrom.parentElement!=elTo)) {

   if (iTab!=g_iShCur) {
    with (frames['frTabs'].document.all) {
     tdTab[iTab].style.background=c_rgszClr[1];
    }
   }
  }
 }
}

function fnSetActiveSheet(iSh)
{
 if (iSh!=g_iShCur) {
  fnSetTabProps(g_iShCur,false);
  fnSetTabProps(iSh,true);
  g_iShCur=iSh;
 }
}

 window.g_iIEVer=fnGetIEVer();
 if (window.g_iIEVer>=4)
  fnBuildFrameset();
//-->
</script>
<![endif]><!--[if gte mso 9]><xml>
 <x:ExcelWorkbook>
  <x:ExcelWorksheets>
   <x:ExcelWorksheet>
    <x:Name>Sheet1</x:Name>
    <x:WorksheetSource HRef="Smoothing.files/sheet001.html"/>
   </x:ExcelWorksheet>
  </x:ExcelWorksheets>
  <x:Stylesheet HRef="Smoothing.files/stylesheet.css"/>
  <x:WindowHeight>8820</x:WindowHeight>
  <x:WindowWidth>19200</x:WindowWidth>
  <x:WindowTopX>32767</x:WindowTopX>
  <x:WindowTopY>32767</x:WindowTopY>
  <x:ProtectStructure>False</x:ProtectStructure>
  <x:ProtectWindows>False</x:ProtectWindows>
 </x:ExcelWorkbook>
</xml><![endif]-->
</head>

<frameset rows="*,39" border=0 width=0 frameborder=no framespacing=0>
 <frame src="Smoothing.files/sheet001.html" name="frSheet">
 <frame src="Smoothing.files/tabstrip.html" name="frTabs" marginwidth=0 marginheight=0>
 <noframes>
  <body>
   <p>&#27492;&#39029;&#38754;&#20351;&#29992;&#20102;&#26694;&#26550;&#65292;&#32780;&#24744;&#30340;&#27983;&#35272;&#22120;&#19981;&#25903;&#25345;&#26694;&#26550;&#12290;</p>
  </body>
 </noframes>
</frameset>
</html>

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
