import numpy as np
import math
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from bresenham import bresenham
from collections import defaultdict
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
import cv2
from sklearn.linear_model import LinearRegression
import argparse
from skimage.morphology import erosion, dilation, opening, closing
from sklearn.metrics import mean_squared_error


def tiff_add():
    path = '/work/scratch/ychen/segmentation/smoothing/drosophila_new_2/filter_boundary'
    tp = 9
    for x in range(tp):
        inner_im = imread(os.path.join(path, 'inner_filter{}.tif'.format(str(x))), plugin='tifffile')
        outer_im = imread(os.path.join(path, 'outer_filter{}.tif'.format(str(x))), plugin='tifffile')
        imsave(os.path.join(path, 'combine{}.tif'.format(str(x))), inner_im+outer_im, plugin='tifffile')

# image postprocessing
def erode(img):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img, kernel, iterations = 2)
    return erosion

# image postprocessing
def dilate(img):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 2)
    return dilation

# image postprocessing
def close(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

# image postprocessing
def open(img):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

# calculating the mean BF score for whole image， comparing with 2D U-net and 3D U-net
def bfscore(im_true, im_pred, value):
    print(im_true.shape[0], im_true.shape[1], im_true.shape[2])
    print(im_pred.shape)
    assert im_pred.shape == im_true.shape, 'the inut shapes is not the same'
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for idx_x in range(im_true.shape[0]):
        for idx_y in range(im_true.shape[1]):
            for idx_z in range(im_true.shape[2]):
                if im_true[idx_x, idx_y, idx_z] == 65535 and im_pred[idx_x, idx_y, idx_z] == value:
                    tp += 1
                elif im_true[idx_x, idx_y, idx_z] == 65535 and im_pred[idx_x, idx_y, idx_z] == 0:
                    fn += 1
                elif im_true[idx_x, idx_y, idx_z] == 0 and im_pred[idx_x, idx_y, idx_z] == value:
                    fp += 1
                elif im_true[idx_x, idx_y, idx_z] == 0 and im_pred[idx_x, idx_y, idx_z] == 0:
                    tn += 1
                else:
                    print('The value is incorrect, check image again')

    print(tp, fn, fp, tn)
    assert (tp+tn+fp+fn) == (im_true.shape[0]*im_true.shape[1]*im_true.shape[2]), "the whole value is not matched, check again"
    #print(tp+tn+fp+fn)
    #print(im_true.shape[0]*im_true.shape[1]*im_true.shape[2])
    recall = tp/(tp+fn)
    precision = tp / (tp + fp)
    bf_score = 2*tp/(2*tp+fp+fn)
    return bf_score, recall, precision, tp, fn, fp

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='drosophila',
                        help='which dataset')
    return parser.parse_args()
args = args_parse()

# calculate distance via surface map, along point axis
def t_distance2(i, j):
    source = ['origin', 'smoothing', 'regression']
    src = ['drosophila', 'drosophila_new_1', 'drosophila_new_2']
    path = os.path.join('/work/scratch/ychen/segmentation/smoothing', src[j])
    if source[i] == 'origin':
        inner = imread(os.path.join(path, 'inner_surface_map.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'outer_surface_map.tif'), plugin='tifffile')
    if source[i] == 'smoothing':
        inner = imread(os.path.join(path, 'regression_smoothing','inner_smoothing.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'regression_smoothing','outer_smoothing.tif'), plugin='tifffile')
    if source[i] == 'regression':
        inner = imread(os.path.join(path, 'smoothing_regression','inner_regression.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'smoothing_regression','outer_regression.tif'), plugin='tifffile')
    tp = inner.shape[0]
    slice = inner.shape[1]
    points = inner.shape[2]
    #print(tp, slice, points)
    inner_new = np.transpose(inner, (2,0,1))
    outer_new = np.transpose(outer, (2, 0, 1))
    dist_list = []
    for i in range(points):
        if i<points-1:
            inner_dist = np.round(mean_squared_error(inner_new[i], inner_new[i+1], squared=False), decimals=5)
            outer_dist = np.round(mean_squared_error(outer_new[i], outer_new[i+1], squared=False), decimals=5)
            #print(inner_dist, outer_dist)
            dist_list.append((inner_dist+outer_dist))#/(tp*slice))
        else:pass
    dist = np.round(np.average(dist_list), decimals=5)
    #print(dist)
    return dist

# calculate distance via surface map, along slice axis
def t_distance1(i, j):
    source = ['origin', 'smoothing', 'regression']
    src = ['drosophila', 'drosophila_new_1', 'drosophila_new_2']
    path = os.path.join('/work/scratch/ychen/segmentation/smoothing', src[j])
    if source[i] == 'origin':
        inner = imread(os.path.join(path, 'inner_surface_map.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'outer_surface_map.tif'), plugin='tifffile')
    if source[i] == 'smoothing':
        inner = imread(os.path.join(path, 'regression_smoothing','inner_smoothing.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'regression_smoothing','outer_smoothing.tif'), plugin='tifffile')
    if source[i] == 'regression':
        inner = imread(os.path.join(path, 'smoothing_regression','inner_regression.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'smoothing_regression','outer_regression.tif'), plugin='tifffile')
    tp = inner.shape[0]
    slice = inner.shape[1]
    points = inner.shape[2]
    #print(tp, slice, points)
    inner_new = np.transpose(inner, (1,0,2))
    outer_new = np.transpose(outer, (1, 0, 2))
    dist_list = []
    for i in range(slice):
        if i<slice-1:
            inner_dist = np.round(mean_squared_error(inner_new[i], inner_new[i+1], squared=False), decimals=5)
            outer_dist = np.round(mean_squared_error(outer_new[i], outer_new[i+1], squared=False), decimals=5)
            #print(inner_dist, outer_dist)
            dist_list.append((inner_dist+outer_dist))#/(tp*points))
        else:pass
    dist = np.round(np.average(dist_list), decimals=5)
    #print(dist)
    return dist

# calculate distance via surface map, along time axis
def t_distance(i, j):
    source = ['origin', 'smoothing', 'regression']
    src = ['drosophila', 'drosophila_new_1', 'drosophila_new_2']
    path = os.path.join('/work/scratch/ychen/segmentation/smoothing', src[j])
    if source[i] == 'origin':
        inner = imread(os.path.join(path, 'inner_surface_map.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'outer_surface_map.tif'), plugin='tifffile')
    if source[i] == 'smoothing':
        inner = imread(os.path.join(path, 'regression_smoothing','inner_smoothing.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'regression_smoothing','outer_smoothing.tif'), plugin='tifffile')
    if source[i] == 'regression':
        inner = imread(os.path.join(path, 'smoothing_regression','inner_regression.tif'), plugin='tifffile')
        outer = imread(os.path.join(path, 'smoothing_regression','outer_regression.tif'), plugin='tifffile')
    tp = inner.shape[0]
    slice = inner.shape[1]
    points = inner.shape[2]
    #print(tp, slice, points)
    dist_list = []
    for i in range(tp):
        if i<tp-1:
            inner_dist = np.round(mean_squared_error(inner[i], inner[i+1], squared=False), decimals=5)
            outer_dist = np.round(mean_squared_error(outer[i], outer[i+1], squared=False), decimals=5)
            print(inner_dist, outer_dist)
            dist_list.append((inner_dist+outer_dist))#/(slice*points))
        else:pass
    dist = np.round(np.average(dist_list), decimals=5)
    #print(dist)
    return dist

def main():

    # i = 2
    # j = 0
    # time_dist = t_distance(i, j)
    # slice_dist = t_distance1(i, j)
    # point_dist = t_distance2(i, j)
    # print(time_dist)
    # print(slice_dist)
    # print(point_dist)
    # print(time_dist + slice_dist + point_dist)

    # pred_dir = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/model/drosophila_new_2/full_mask/histogram_rescale_4_1000/whole_images'
    # true_dir = '/work/scratch/ychen/preprocessing/drosophila_new_2/small_size/rotated/mask/full_mask'
    # tp_l = []
    # fn_l = []
    # fp_l = []
    # recall_l = []
    # precision_l = []
    # bfscore_l = []
    # for i in os.listdir(pred_dir):
    #     if i.endswith('.tif'):
    #         im_pred = imread(os.path.join(pred_dir, i), plugin='tifffile')
    #         im_pred = np.squeeze(im_pred)
    #         im_true = imread(os.path.join(true_dir, i[:4]+'_Masked_rotated.tif'), plugin='tifffile')
    #         im_true = np.squeeze(im_true)
    #         bf_score, recall, precision, tp, fn, fp = bfscore(im_pred=im_pred, im_true=im_true, value=65535)
    #         tp_l.append(tp)
    #         fn_l.append(fn)
    #         fp_l.append(fp)
    #         recall_l.append(recall)
    #         precision_l.append(precision)
    #         bfscore_l.append(bf_score)
    # print(np.sum(tp_l))
    # print(np.sum(fn_l))
    # print(np.sum(fp_l))
    # print(np.average(recall_l))
    # print(np.average(precision_l))
    # print(np.average(bfscore_l))
    tiff_add()

if __name__ == '__main__':
    main()