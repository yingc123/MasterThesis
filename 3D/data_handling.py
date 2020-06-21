# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:09:51 2018

@author: eschweiler
Modification: YingChen
"""


import numpy as np
import os
import csv
import glob
import itertools
import h5py
import warnings
import multiprocessing as mp

from functools import partial
from xml.dom import minidom
from skimage import io, morphology, measure, exposure
from scipy.ndimage.morphology import distance_transform_edt

from utils import sanitycheck_patch_params, sanitycheck_path, spherical_instance_sampling, samples2delaunay, delaunay2indices, print_continuous




###############################################################################
''' PROCESSING '''
###############################################################################


def extract_from_array(matrix, start, end):

    start = np.array(start)
    end = np.array(end)

    patch_dims = len(start)

    pad_before = np.minimum([0,]*patch_dims, start)
    pad_after = np.maximum([0,]*patch_dims, end-matrix.shape[:patch_dims])

    crop_start = np.maximum([0,]*patch_dims, start)
    crop_end = np.minimum(matrix.shape[:patch_dims], end)

    slices = tuple(map(slice, crop_start, crop_end))
    matrix = matrix[slices]

    matrix = np.pad(matrix, [p for p in zip(pad_before, pad_after)]+[(0,0)]*(matrix.ndim-patch_dims), 'symmetric')

    return matrix



###############################################################################
''' TRANSFORMATIONS '''
###############################################################################


def to_onehot(data, labels=None, **kwargs):

    if labels is None:
        labels = np.unique(data)

    # if there still is a grayscale channel layer, cut it off
    if data.shape[-1] == 1:
        data = data[...,0]

    data_onehot = np.zeros(data.shape+(len(labels),), dtype=data.dtype)
    for num_label,label in enumerate(labels):
        data_onehot[...,num_label] = data==label

    return data_onehot


def cut_last_dim(data, **kwargs):
    return data[...,0]


# normalized the image data
def normalization(data, max_val=0, min_val=0, mean_val=0, std_val=0, norm_method=None, **kwargs):

    if norm_method == 'minmax_data':
        #data = data/(data.max()-data.min())-data.min()/(data.max()-data.min()) # prevents value overflows
        data = data-data.min()
        data = data/data.max()

    elif norm_method == 'minmax_dtype':
        stats = np.iinfo(data.dtype)
        data = data/(stats.max-stats.min)-stats.min/(stats.max-stats.min)

    elif norm_method is None or norm_method == 'minmax':
        data = (data-min_val)/(max_val-min_val)

    elif norm_method == 'meanstd':
        data = (data-mean_val)/std_val

    elif norm_method == 'scale':
        data = data/max_val

    else:
        raise UserWarning('Unknown normalization method "{0}".'.format(norm_method))

    #data = data.astype(np.float32)

    return data


def histogram_rescale(data, **kwargs):
    p15, p95 = np.percentile(data, (15, 95))
    img_rescale = exposure.rescale_intensity(data, in_range=(p15, p95))

    return img_rescale

def histogram_equalization(data, **kwargs):
    img_HE = exposure.equalize_hist(data)

    return img_HE

# adds gaussian noise to the image data
def add_gaussian_noise(data, gauss_prob=1.0, gauss_mean=0, gauss_sigma=0.1, gauss_scale=[0,1], max_val=None, min_val=0, **kwargs):

    if np.random.rand() <= gauss_prob:

        if max_val is None:
            max_val = data.max()

        # choose random noise scaling
        gauss_scale = np.random.choice(range(gauss_scale[0],gauss_scale[1]+1))

        # apply noise to the image
        data = np.clip(data + gauss_scale*np.random.normal(gauss_mean, gauss_sigma, data.shape), min_val, max_val)

    return data



# scale the image data by a random value
def intensity_scale(data, intens_scale_min=0.1, intens_scale_max=1.0, intens_scale_prob=1.0, max_val=None, **kwargs):

    if np.random.rand() <= intens_scale_prob:

        # draw scale from uniform distribution
        intens_scale = np.random.uniform(intens_scale_min, intens_scale_max)
        data = data * intens_scale

        # keep values within the desired range
        if not max_val is None:
            data = np.clip(data, 0, max_val)

    return data


# gamma transformation
def gamma_transform(data, gamma_min=0, gamma_max=2, gamma_prob=1.0, max_val=None, **kwargs):

    if np.random.rand() < gamma_prob:

        # draw scale from uniform distribution
        gamma = np.random.uniform(gamma_min, gamma_max)
        data = max_val*((data/max_val)**gamma)

    return data


def remove_small_objects(data, bg_values=[0], size_thresh=10, **kwargs):

    # Remove background regions
    for bg_value in bg_values:
        data[data==bg_value] = 0

    # Remove small objects
    morphology.remove_small_objects(data, min_size=size_thresh, in_place=True)

    return data



def apply_augmentation(data, patch_params):

    # rotation
    if not patch_params['rotation_count'] is None:
        data = np.rot90(data, k=patch_params['rotation_count'], axes=(0,1))

    # mirroring
    if patch_params['mirror_x']:
        data = np.flip(data, axis=0)

    if patch_params['mirror_y']:
        data = np.flip(data, axis=1)

    return data


def instances2multiclass(data, bg_values=[0], size_thresh=20, **kwargs):

    data = data.astype(np.uint16)

    #labels: 1=bg, 2=centroid, 3=membrane
    data_multiclass = np.zeros(data.shape, dtype=np.uint8)

    # get all instances
    instances = np.unique(data)
    # exclude background instance
    instances = list(set(instances)-set(bg_values))

    # save background mask and exclude background from the image
    for bg_value in bg_values:
        data_multiclass[data==bg_value] = 1
        data[data==bg_value] = 0

    # get membrane segmentation
    instance_mask = data - morphology.erosion(data)
    data_multiclass[instance_mask>0] = 3

    # get centroids for each instance
    # make sure to exclude sampling artifacts at instance borders
    instance_mask = morphology.dilation(instance_mask)
    data[instance_mask>0] = 0
    regions = measure.regionprops(data[...,0])
    for props in regions:
        if props.area > size_thresh:
            region_centroid = props.centroid
            region_centroid = tuple([int(np.round(c)) for c in region_centroid])
            data_multiclass[region_centroid] = 2
    data_multiclass = morphology.dilation(data_multiclass)


    return data_multiclass


def instances2binaryclass(data, **kwargs):
    data = data.astype(np.float16)
    data[data < 0.5] = 0
    data[data >= 0.5] = 1
    return data



def instances2distmap(data, bg_values=[0], saturation_dist=1, dist_activation='tanh', **kwargs):

    data = data.astype(np.uint16)

    # extract background value
    data_bg = np.zeros(data.shape, dtype=np.bool)
    for bg_value in bg_values:
        data_bg = np.logical_or(data_bg, data==bg_value)
    data_bg = distance_transform_edt(data_bg)

    # get membrane segmentation
    data_map = data - morphology.erosion(data)
    data_map = distance_transform_edt(data_map<=0)

    # create saturated distance map
    if 'tanh' in dist_activation:
        data_map = np.multiply((data_bg<=0),data_map) - np.multiply((data_bg>0),data_bg)
        data_map = np.tanh(np.divide(data_map,saturation_dist))
    else:
        data_map = data_map/saturation_dist

    return data_map




def instances2indices(data, bg_values=[0], **kwargs):

    data = data.astype(np.uint16)

    # get all instances
    instances = np.unique(data)
    # exclude background instances
    instances = list(set(instances)-set(bg_values))
    # exclude background from the image
    for bg_value in bg_values:
        data[data==bg_value] = 0

    regions = measure.regionprops(data[...,0])
    centroids = []
    for props in regions:
        centroids.append(props.centroid)

    return centroids




def instances2label(data, bg_values=[0], verbose=False, **kwargs):

    is_instance = np.array(data>=1).any()

    return is_instance[...,np.newaxis]




def instances2harmonicmask(data, s2h_converter, probs=None, shape=(112,112,112), cell_size=(8,8,8), dets_per_region=3, bg_values=[0], verbose=False, **kwargs):
    '''
    An output mask for 3 possible detections in 3D will have the following channels:
        [p1,p2,...,x1,y1,z1,x2,y2,z2,...,h11,h12,h13,...,h21,h22,h23,...]
    '''

    # sample each instance
    instances, r_sampling, centroids = spherical_instance_sampling(data[...,0], s2h_converter.theta_phi_sampling, bg_values=bg_values, verbose=verbose)

    # if no probabilities are given, assume certain detections
    if probs is None:
        probs = [1,]*len(instances)

    # convert the sampling to harmonics
    r_harmonics = s2h_converter.convert(r_sampling)
    num_coefficients = s2h_converter.num_coefficients

    # create mask
    mask = np.zeros(tuple([int(s/c) for s,c in zip(shape,cell_size)])+((4+num_coefficients)*dets_per_region,), dtype=np.float)
    for idx,descriptor,prob in zip(centroids,r_harmonics,probs):
        # get current cell index and intra cell offset
        cell_idx = [int(i//c) for i,c in zip(idx,cell_size)]
        voxel_offset = [int(i%c)/(c-1) for i,c in zip(idx, cell_size)]
        # each cell is allowed to contain multiple objects (each [p,x,y,z])
        for num_det in range(dets_per_region):
            if mask[tuple(cell_idx)][num_det] == 0:
                # set probability information
                mask[tuple(cell_idx)][num_det] = prob
                # set positional information
                mask[tuple(cell_idx)][dets_per_region*1+num_det*3:dets_per_region*1+(num_det+1)*3] = voxel_offset
                # set shape information
                mask[tuple(cell_idx)][dets_per_region*4+num_det*num_coefficients:dets_per_region*4+(num_det+1)*num_coefficients] = descriptor
                break
    return mask




def harmonicmask2sampling(harmonic_mask, h2s_converter, cell_size=(8,8,8), dets_per_region=3, thresh=0., convert2radii=True, positional_weighting=False, **kwargs):

    probs = []
    centroids = []
    shape_descriptors = []

    num_coefficients = h2s_converter.num_coefficients
    trace_indices = itertools.product(*[range(s) for s in harmonic_mask.shape[:-1]])

    patch_size = tuple([cs*hi for cs,hi in zip(cell_size, harmonic_mask.shape[:-1])])

    for trace_idx in trace_indices:
        # extract detection information for the current region
        pred_info = harmonic_mask[trace_idx]
        for num_det in range(dets_per_region):

            # extract shape information
            harmonic_descriptors = pred_info[dets_per_region*4+num_det*num_coefficients:dets_per_region*4+(num_det+1)*num_coefficients]
            # sanitycheck if there actually is a shape and not only a single noise point
            if np.count_nonzero(harmonic_descriptors) > 0:

                # extract positional information
                pos_info = pred_info[dets_per_region*1+num_det*3:dets_per_region*1+(num_det+1)*3]
                # reconstruct the position within the mask space
                # (position+offset)*cell_size
                # restricted by 0 and the image size (=mask_shape*cell_size)
                mask_index = [np.clip(np.ceil((t+p)*(s)-1),0,s*ms-1).astype(np.int16) for t,p,s,ms in zip(trace_idx, pos_info, cell_size, harmonic_mask.shape[:-1])]

                # extract confidence information
                prob = pred_info[num_det]
                # add positional weight (low at patch boundaries, weighted by tanh and scaled by the 8th of the patch size)
                if positional_weighting:
                    prob_weight = [np.minimum(np.tanh(mi/ps*8),np.tanh((ps-mi)/ps*8)) for mi,ps in zip(mask_index, patch_size)]
                    prob = prob * np.min(prob_weight)

                 # if a cell was detected, start localizing it
                if prob>thresh:
                    # append information
                    shape_descriptors.append(harmonic_descriptors)
                    centroids.append(tuple(mask_index))
                    probs.append(prob)

    if convert2radii:
        shape_descriptors = h2s_converter.convert(shape_descriptors)

    return centroids, probs, shape_descriptors




def descriptors2image_poolhelper(descriptor, theta=None, phi=None, shape=(112,112,112), thresh=0, verbose=False):

        centroid = descriptor[0]
        prob = descriptor[1]
        shape_descriptor = descriptor[2]

        if prob >= thresh and np.count_nonzero(shape_descriptor) > 0:

            # Create object map
            max_object_extend = int(np.ceil(shape_descriptor.max()))
            x,y,z = np.indices((2*max_object_extend+1,)*len(shape))
            idx = np.stack([x,y,z], axis=-1)
            idx[...,0] = idx[...,0]-max_object_extend
            idx[...,1] = idx[...,1]-max_object_extend
            idx[...,2] = idx[...,2]-max_object_extend

            # Get Delaunay triangulation and indices of voxels within the object
            delaunay_tri = samples2delaunay([shape_descriptor, theta, phi], cartesian=False)
            instance_indices = delaunay2indices(delaunay_tri, idx)

            # Adjust instance indices to the actual image position, considering the bounds
            instance_indices = tuple([np.array([np.maximum(0, np.minimum(shape[0]-1, i+int(np.round(centroid[0]))-max_object_extend)) for i in instance_indices[0]]),\
                                      np.array([np.maximum(0, np.minimum(shape[1]-1, i+int(np.round(centroid[1]))-max_object_extend)) for i in instance_indices[1]]),\
                                      np.array([np.maximum(0, np.minimum(shape[2]-1, i+int(np.round(centroid[2]))-max_object_extend)) for i in instance_indices[2]])])
        else:
             # Create emtpy instance
            instance_indices = ()

        return instance_indices



def descriptors2image(descriptors, theta_phi_sampling=None, shape=(112,112,112), thresh=0, verbose=False, num_kernel=4, **kwargs):
    '''
    descriptors need to be in shape [centroids, probs, radii_sampling]
    '''

    theta = [tps[0] for tps in theta_phi_sampling]
    phi = [tps[1] for tps in theta_phi_sampling]

    instance_mask = np.zeros(shape, dtype=np.uint16)

    # Parallelize instance voxelization
    with mp.Pool(processes=num_kernel) as p:
        instance_list = p.map(partial(descriptors2image_poolhelper, theta=theta, phi=phi, shape=shape, thresh=thresh, verbose=verbose), list(zip(*descriptors)))

    # Fill the final instance mask
    for instance_count,instance in enumerate(instance_list):
        if len(instance)>0:
            instance_mask[instance] = instance_count+1

    return instance_mask




# construct mask from annotation indices
def indices2mask(indices, probs=None, shape=(112,112,112), cell_size=(8,8,8), num_channels=12, axes_order=None, **kwargs):

    '''
    A mask for 3 possible detections in 3D will have the following channels:
        [p1,p2,p3,x1,y1,z1,x2,y2,z2,x3,y3,z3]
    '''
    # if no probabilities are given, assume certain detections
    if probs is None:
        probs = [1,]*len(indices)

    det_count = num_channels//4

    # create mask
    mask = np.zeros(tuple([int(s/c) for s,c in zip(shape,cell_size)])+(num_channels,), dtype=np.float)

    # change order if necessarry
    #mask = np.transpose(mask, axes_order) if not axes_order is None else mask

    for idx,prob in zip(indices, probs):
        # get current cell index and intra cell offset
        cell_idx = [int(i//c) for i,c in zip(idx,cell_size)]
        voxel_offset = [int(i%c)/(c-1) for i,c in zip(idx, cell_size)]
        # each cell is allowed to contain multiple objects (each [p,x,y,z])
        for num_det in range(det_count):
            if mask[tuple(cell_idx)][num_det] == 0:
                # set occurence probability
                mask[tuple(cell_idx)][num_det] = prob
                # set positional information
                mask[tuple(cell_idx)][det_count+num_det*3:det_count+(num_det+1)*3] = voxel_offset
                break

    # re-change order if necessarry
    #mask = np.transpose(mask, axes_order) if not axes_order is None else mask

    return mask



# reconstruct annotation indices from mask
def mask2indices(mask, cell_size=(8,8,8), thresh=0., **kwargs):
    '''
    A mask for 3 possible detections in 3D will have the following channels:
        [p1,p2,p3,x1,y1,z1,x2,y2,z2,x3,y3,z3]
    '''

    indices = []
    probs = []

    trace_indices = itertools.product(*[range(s) for s in mask.shape[:-1]])
    for trace_idx in trace_indices:
        # extract detection information for the current region
        det_info = mask[trace_idx]
        det_count = len(det_info)//4
        for num_det in range(det_count):
            # if a cell was detected, start localizing it
            if det_info[num_det]>thresh:
                # extract positional information
                pos_info = det_info[det_count+num_det*3:det_count+(num_det+1)*3]
                # reconstruct the position within the mask space
                # (position+offset)*cell_size
                # restricted by 0 and the image size (=mask_shape*cell_size)
                mask_index = [np.clip(np.ceil((t+p)*(s)-1),0,s*ms-1).astype(np.int16) for t,p,s,ms in zip(trace_idx, pos_info, cell_size, mask.shape[:-1])]
                indices.append(tuple(mask_index))
                # extract confidence information
                probs.append(det_info[num_det])
    return indices, probs



# construct mask image from point annotations
def indices2image(index_list, image_shape, prob_list=None, **kwargs):
    if prob_list is None:
        prob_list = [1.0,]*len(index_list)
    image = np.zeros(image_shape, dtype=np.uint8)
    for idx, prob in zip(index_list,prob_list):
        image[tuple([int(i) for i in idx])] = 255*prob
    image = morphology.dilation(image, selem=np.ones((3,)*len(image_shape), dtype=np.bool))
    return image



###############################################################################
''' LIST HANDLING '''
###############################################################################

# get files from a specific directory
def get_files(folder, pre_path='', extension=''):
    filelist = sorted(glob.glob(os.path.join(os.path.normpath(folder), '*.'+extension)))

    filelist = [os.path.join(os.path.normpath(pre_path), os.path.basename(f)) for f in filelist]

    return filelist



# Create file list of train, test and valdidation splits
def write_list(im_list, mask_list, save_path='', test_split=0.2, val_split=0.1, rnd_seed=None):

    save_path = os.path.normpath(save_path)

    assert len(im_list)==len(mask_list), 'Number of images and masks does not match'

    # get indices for all sets
    idx_list = np.arange(len(im_list))
    np.random.seed(rnd_seed)
    np.random.shuffle(idx_list)
    im_list = [im_list[i] for i in idx_list]
    mask_list = [mask_list[i] for i in idx_list]

    test_count = int(len(idx_list)*test_split)
    val_count = int((len(idx_list)-test_count)*val_split)
    train_count = int(len(idx_list)-val_count-test_count)

    # write csv files
    if train_count > 0:
        with open(save_path+'_train.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[0:train_count], mask_list[0:train_count]))
    if test_count > 0:
        with open(save_path+'_test.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[train_count:train_count+test_count], \
                                 mask_list[train_count:train_count+test_count]))
    if val_count > 0:
        with open(save_path+'_val.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[train_count+test_count:], mask_list[train_count+test_count:]))


# read a specific filelist
def read_list(list_path, use_local=False):
    filelist = []

    try:
        with open(list_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                row = [sanitycheck_path(r, use_local=use_local) for r in row]
                filelist.append(row)
    except:
        filelist = []

    return filelist


###############################################################################
''' LOADING '''
###############################################################################

# Load tiff file
def load_tiff(filepath, patch_params=None, **kwargs):

    patch_params = sanitycheck_patch_params(patch_params)

    ## loading
    image = io.imread(filepath)

    # create a dummy channel dimension, if dealing with grayscale images
    if patch_params['grayscale']:
        image = image[...,np.newaxis]

    # ensure to have ordering: [spatial_dims, channels]
    if not patch_params['axes_order'] is None:
        image = np.transpose(image, axes=patch_params['axes_order'])

    # save the shape of the original image
    patch_params['data_shape'] = image.shape[::-1]

    ## position handling
    # if no patch shape is given, extract the whole image
    if patch_params['shape'] is None:
        patch_params['shape'] = image.shape[:-1]

    # if no specification is given, extract patch at random location (lower left)
    if patch_params['start'] is None:
        start_max = [np.maximum(0,i-p) for i,p in zip(image.shape[:-1],patch_params['shape'])]
        patch_params['start'] = [np.asscalar(np.random.randint(0,s+1,1)) for s in start_max]

    # extract the patch
    image = extract_from_array(image, patch_params['start'], np.add(patch_params['start'], patch_params['shape']))

    return image, patch_params




# hdf5 data loading
def load_bdv_hdf5(filepath, patch_params=None, mask_prob=0, **kwargs):

    patch_params = sanitycheck_patch_params(patch_params)

    with h5py.File(filepath, 'r') as f_handle:
        data = f_handle['t00000/s00/{0}/cells'.format(0 if patch_params['scale_level'] is None else patch_params['scale_level'])]

        data_shape = data.shape
        if not patch_params['axes_order'] is None:
            data_shape = tuple([data_shape[ax] for ax in patch_params['axes_order'][:-1]])  # without color channel
        patch_params['data_shape'] = data_shape

        pad_before = (0,0,0)

        if patch_params['shape'] is None:
            # if no shape is provided, load the whole image
            patch_params['shape'] = data_shape
            patch = data[:]
            rnd_start = (0,0,0)
        else:
            # get data slice
            if not patch_params['start'] is None:
                if len(patch_params['start'])==len(data.shape):
                    rnd_start = patch_params['start']
                else:
                    raise ValueError('Dimensions of given start index and data do not match.')
            else:

                rnd_start = [np.random.randint(0, np.maximum(1,dim_shape-patch_params['shape'][num_dim])) for num_dim,dim_shape in enumerate(data_shape)]

            rnd_end = [idx+patch_params['shape'][num_dim] for num_dim,idx in enumerate(rnd_start)]

            pad_before = [-rs if rs<0 else 0 for rs in rnd_start]
            rnd_start = [rs if rs>=0 else 0 for rs in rnd_start]

            slicing = tuple(map(slice, rnd_start, rnd_end))
            if not patch_params['axes_order'] is None:
                slicing = tuple([slicing[ax] for ax in patch_params['axes_order'][:-1]])
            # extract patch
            patch = data[slicing]

        # save start parameters
        patch_params['start'] = [s-p for s,p in zip(rnd_start,pad_before)]

        # create a channel dimension, if dealing with grayscale images
        if patch_params['grayscale']:
            patch = patch[...,np.newaxis]

        # ensure to have ordering: [spatial_dims, channels]
        if not patch_params['axes_order'] is None:
            patch = np.transpose(patch, axes=patch_params['axes_order'])

        # pad to desired size, if necessary
        pad_width = [(pb,np.maximum(0, s-p)-pb) for p,s,pb in zip(patch.shape[:-1], patch_params['shape'], pad_before)]+[(0,0),] # add channel padding
        patch = np.pad(patch, pad_width, 'constant')

    return patch, patch_params



# extract annotations from XML files
def load_mamut_annotations(filepath, start=None, shape=None, mask_prob=0.5, scale_level=0, **kwargs):

    if scale_level is None:
        scale_level = 0

    with minidom.parse(filepath) as xmldoc:

        # extract image info
        image_data = xmldoc.getElementsByTagName('ImageData')[0]
        image_info = {}
        image_info['filename'] = image_data.attributes['filename'].value
        image_info['width'] = float(image_data.attributes['width'].value)
        image_info['height'] = float(image_data.attributes['height'].value)
        image_info['nslices'] = float(image_data.attributes['nslices'].value)
        image_info['nframes'] = float(image_data.attributes['nframes'].value)

        # extract voxel sizes
        xmldoc_image = minidom.parse(os.path.join(os.path.dirname(filepath),image_info['filename']))
        voxel_size = xmldoc_image.getElementsByTagName('voxelSize')
        voxel_size = voxel_size[0].getElementsByTagName('size')
        voxel_size = voxel_size[0].childNodes[0].data.split(' ')
        voxel_size = [float(v) for v in voxel_size]

        # extract annotations
        annotations = {}
        annotations['pos_x'] = []
        annotations['pos_y'] = []
        annotations['pos_z'] = []
        annotations['pos_t'] = []
        annotations['frame'] = []
        annotations['radius'] = []
        itemlist = xmldoc.getElementsByTagName('Spot')
        for item in itemlist:
            annotations['pos_x'].append(float(item.attributes['POSITION_X'].value)/voxel_size[0])
            annotations['pos_y'].append(float(item.attributes['POSITION_Y'].value)/voxel_size[1])
            annotations['pos_z'].append(float(item.attributes['POSITION_Z'].value)/voxel_size[2])
            annotations['pos_t'].append(float(item.attributes['POSITION_T'].value))
            annotations['frame'].append(float(item.attributes['FRAME'].value))
            annotations['radius'].append(float(item.attributes['RADIUS'].value)) #scaling???

        # configure the shape
        if shape is None:
            shape = [image_info['width'], image_info['height'], image_info['nslices']]
        shape = [s*(2**scale_level) for s in shape]

        if start is None:
            # choose start index from annotation coordinates
            # assure, that patches are within image boundaries
            if np.random.random()<mask_prob:
                annot_idx = np.random.randint(0, high=len(annotations['pos_x']), size=1, dtype=np.uint16)[0]
                start = [np.minimum(np.maximum(0,annotations['pos_x'][annot_idx]-float(shape[0]//2)), image_info['width']-float(shape[0])),
                         np.minimum(np.maximum(0,annotations['pos_y'][annot_idx]-float(shape[1]//2)), image_info['height']-float(shape[1])),
                         np.minimum(np.maximum(0,annotations['pos_z'][annot_idx]-float(shape[2]//2)), image_info['nslices']-float(shape[2]))]
            # else choose random start idx
            else:
                start = [np.random.choice(np.arange(0,np.maximum(1,image_info['width']-shape[0]+1))),
                         np.random.choice(np.arange(0,np.maximum(1,image_info['height']-shape[1]+1))),
                         np.random.choice(np.arange(0,np.maximum(1,image_info['nslices']-shape[2]+1)))]

        # remove annotations outside the specified patch area
        idx_choose = []
        for num_annot in range(len(annotations['pos_x'])):
            if all([start[0]<=annotations['pos_x'][num_annot]<start[0]+shape[0],
                    start[1]<=annotations['pos_y'][num_annot]<start[1]+shape[1],
                    start[2]<=annotations['pos_z'][num_annot]<start[2]+shape[2]]):
                idx_choose.append(num_annot)
        for key in annotations:
            annotations[key] = [annotations[key][i] for i in idx_choose]

        # adjust coordinates to the patch size
        annotations['pos_x'] = [int(annot_x-start[0]) for annot_x in annotations['pos_x']]
        annotations['pos_y'] = [int(annot_y-start[1]) for annot_y in annotations['pos_y']]
        annotations['pos_z'] = [int(annot_z-start[2]) for annot_z in annotations['pos_z']]

        start = [int(s) for s in start]
        shape = [int(s) for s in shape]

    return annotations, start, shape





###############################################################################
''' SAVING '''
###############################################################################

def save_image(image, save_path, axes_order=None, bit_16=True, **kwargs):

    if image.shape[-1]>1:
        image = image.astype(np.float16)
    else:
        image = image.astype(np.float32)

    # normalize the image
    if bit_16:
        image = image - image.min()
        if image.max() != 0:
            image = image / image.max()
        image = 65535*image
        image = image.astype(np.uint16)

    # reorder axes if necessary
    image = np.transpose(image, axes_order) if not axes_order is None else image

    # save the image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(save_path, image)







