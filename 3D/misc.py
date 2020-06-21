# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:54:55 2019

@author: eschweiler
Modification: YingChen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import json
import glob
import cv2
import csv

from skimage import io
from skimage.feature import peak_local_max
from skimage.measure import regionprops, compare_ssim
from skimage.morphology import erosion
from skimage.transform import resize
from scipy.ndimage import zoom, morphology, filters
from scipy.misc import imresize
from scipy.signal import medfilt
from sklearn.model_selection import StratifiedKFold
from difflib import get_close_matches

# Append the path containing all customized utility functions
parent_path = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
sys.path.insert(0, parent_path)

from data_loader import image_tiler
from data_handling import *
from utils import agglomerative_clustering, metric_collection, harmonic_non_max_suppression, get_sampling_sphere, \
    spherical_instance_sampling, samples2delaunay, delaunay2indices, harmonics2sampling, sampling2harmonics, \
    multiclass_f1_score, seg_score_accumulator


# Predicting masks on whole images
def predict_whole_image(filepath, config, model, overlap=0, application_scales=[0], class_thresh=None, **kwargs):
    tiler = image_tiler(filepath, config, overlap=overlap, application_scales=application_scales)
    pred_map = None

    start_time = time.time()
    for num_tile in range(tiler.__len__()):
        print('\r' * 27 + 'Progress {0:0>2d}% (Tile {1:0>3d}/{2:0>3d})'.format(int(num_tile / tiler.__len__() * 100),
                                                                               num_tile, tiler.__len__()), end='\r')

        # get and predict the current tile
        tile = tiler.__getitem__(num_tile)
        pred = model.predict(tile)

        # apply threshold
        if not class_thresh is None:
            pred[..., 0] = pred[..., 0] >= class_thresh

        # extract cetntral part
        for num_scale, application_scale in enumerate(application_scales):
            start_idx = [0, ] * len(config['shape'])
            end_idx = [np.int(s / (2 ** application_scale)) for s in config['shape']]
            pred_scaled = extract_from_array(pred[num_scale, ...], start_idx, end_idx)
            pred_scaled = np.kron(pred_scaled, np.ones((2 ** application_scale,) * (pred_scaled.ndim - 1) + (1,)))
            # pred_scaled = zoom(pred_scaled, (2**application_scale,)*len(config['shape'])+(1,))
            # pred_scaled = np.pad(pred_scaled, [(0,i-j) for i,j in zip(pred.shape[1:],pred_scaled.shape)], 'symmetric')
            pred[num_scale, ...] = pred_scaled
        pred = np.mean(pred, axis=0)

        # create the new mask if there is no mask yet
        if pred_map is None:
            pred_map = np.zeros((tiler.data_shape + (pred.shape[-1],)), dtype=np.float16)

            # get the start and end position
        location_start = tiler.locations[num_tile]
        location_end = [l + s for l, s in zip(location_start, config['shape'])]

        # determine overlapping constraints
        patch_start = [overlap // 2 if l > overlap // 2 else 0 for l, ps in
                       zip(location_start, config['shape'])]  # //2 to meet in the middle of the overlap
        patch_end = [ps - overlap // 2 if l < ws - overlap // 2 else ps for l, ps, ws in
                     zip(location_end, config['shape'], pred_map.shape)]
        location_start = [l + overlap // 2 if l > overlap // 2 else l for l in location_start]
        location_end = [l - overlap // 2 if l < ws - overlap // 2 else l for l, ws in zip(location_end, pred_map.shape)]

        # insert current patch
        tile_cropping = tuple(map(slice, patch_start, patch_end))
        tile_location = tuple(map(slice, location_start, location_end))
        pred_map[tile_location] = pred[tile_cropping]

    print('\r' * 27 + 'Finished processing {0} tiles after {1:.2f} seconds.'.format(tiler.__len__(),
                                                                                    time.time() - start_time))
    return pred_map


# Predicting masks on whole images
def predict_whole_detections(filepath, image_config, mask_config, general_config, model, overlap=0, use_nms=False,
                             **kwargs):
    tiler = image_tiler(filepath, image_config, overlap=overlap)

    pred_indices = []
    pred_probs = []

    # Predict all tiles
    start_time = time.time()
    for num_tile in range(tiler.__len__()):
        print('\r' * 27 + 'Progress {0:0>2d}% (Tile {1:0>3d}/{2:0>3d})'.format(int(num_tile / tiler.__len__() * 100),
                                                                               num_tile, tiler.__len__()), end='\r')
        tile = tiler.__getitem__(num_tile)
        pred_tile = model.predict(tile)
        tile_indices, tile_probs = mask2indices(pred_tile[0, ...], cell_size=mask_config['transforms']['cell_size'],
                                                thresh=general_config['prob_thresh'])

        for ind, prob in zip(tile_indices, tile_probs):
            # only add point, if its not within half the overlap region and the point is not at the image border
            if (any(i < overlap // 2 for i in ind) and all(l > 0 for l in tiler.locations[num_tile])) or \
                    (any(i > t - overlap // 2 for i, t in zip(ind, image_config['shape'])) and all(
                        l < ds - s for l, ds, s in
                        zip(tiler.locations[num_tile], tiler.data_shape, image_config['shape']))):
                pass
            else:
                ind = [i + pos for i, pos in zip(ind, tiler.locations[num_tile])]
                pred_indices.append(tuple(ind))
                pred_probs.append(prob)
    print('\r' * 27 + 'Finished processing {0} slices after {1:.2f} seconds.'.format(tiler.__len__(),
                                                                                     time.time() - start_time))

    # Perform clustering
    pred_indices, pred_probs = agglomerative_clustering(pred_indices, pred_probs,
                                                        max_dist=general_config['cluster_thresh'], use_nms=use_nms)

    # Create prediction mask
    pred_mask = indices2image(pred_indices, tiler.data_shape, pred_probs)
    if mask_config['grayscale']:
        pred_mask = pred_mask[..., np.newaxis]

    return pred_mask, pred_indices, pred_probs


# Predicting shape masks on whole images
def predict_whole_harmonics(filepath, image_config, mask_config, general_config, model, h2s_converter, overlap=0,
                            use_nms=False, positional_weighting=False, num_kernel=4, verbose=False, **kwargs):
    tiler = image_tiler(filepath, image_config, overlap=overlap)

    pred_indices = []
    pred_probs = []
    pred_shapes = []

    # Predict all tiles
    start_time = time.time()
    for num_tile in range(tiler.__len__()):
        if verbose:
            print(
                '\r' * 27 + 'Progress {0:0>2d}% (Tile {1:0>4d}/{2:0>4d})'.format(int(num_tile / tiler.__len__() * 100),
                                                                                 num_tile, tiler.__len__()), end='\r')
        tile = tiler.__getitem__(num_tile)
        pred_tile = model.predict(tile)
        tile_indices, tile_probs, tile_shapes = harmonicmask2sampling(pred_tile[0, ...], h2s_converter=h2s_converter, \
                                                                      cell_size=mask_config['transforms']['cell_size'], \
                                                                      dets_per_region=general_config['dets_per_region'], \
                                                                      thresh=general_config['prob_thresh'], \
                                                                      convert2radii=True,
                                                                      positional_weighting=positional_weighting)

        for ind, prob, shape_desc in zip(tile_indices, tile_probs, tile_shapes):
            # Either weight or exclude predictions and patch boundaries
            if not positional_weighting and (any(i < overlap // 2 for i in ind) or any(
                    i > t - overlap // 2 for i, t in zip(ind, image_config['shape']))):
                # #and all(l>0 for l in tiler.locations[num_tile])) or \and all(l<ds-s for l,ds,s in zip(tiler.locations[num_tile], tiler.data_shape, image_config['shape']))):
                pass
            else:
                # Perform pre-clustering to reduce computation and redundancy
                tile_indices, tile_probs, tile_shapes = agglomerative_clustering(tile_indices, tile_probs,
                                                                                 shape_descriptors=tile_shapes,
                                                                                 max_dist=general_config[
                                                                                     'cluster_thresh'])

                ind = [i + pos for i, pos in zip(ind, tiler.locations[num_tile])]
                pred_indices.append(tuple(ind))
                pred_probs.append(prob)
                pred_shapes.append(shape_desc)
    if verbose:
        print('\r' * 27 + 'Finished processing {0} slices after {1:.2f} seconds.'.format(tiler.__len__(),
                                                                                         time.time() - start_time))

    # Perform pre-clustering to reduce computation and redundancy
    if len(pred_probs) > 20000:
        pred_mask = np.zeros(tiler.data_shape + (1,))

    else:
        if verbose:
            print('Perform clustering...')
        pred_indices, pred_probs, pred_shapes = agglomerative_clustering(pred_indices, pred_probs,
                                                                         shape_descriptors=pred_shapes,
                                                                         max_dist=general_config['cluster_thresh'])

        # Perform NMS
        if use_nms:
            if verbose:
                print('Perform non-maximum-suppression...')
            pred_indices, pred_probs, pred_shapes = harmonic_non_max_suppression(pred_indices, pred_probs, pred_shapes,
                                                                                 overlap_thresh=general_config[
                                                                                     'nms_thresh'],
                                                                                 num_kernel=num_kernel)

        # Create prediction mask
        if verbose:
            print('Create instance mask...')
        pred_mask = descriptors2image([pred_indices, pred_probs, pred_shapes],
                                      theta_phi_sampling=h2s_converter.theta_phi_sampling, shape=tiler.data_shape,
                                      thresh=general_config['prob_thresh'], num_kernel=num_kernel, verbose=verbose)
        if mask_config['grayscale']:
            pred_mask = pred_mask[..., np.newaxis]

    return pred_mask, pred_indices, pred_probs, pred_shapes




# Evaluate sampling error
def eval_sampling_error(data_gen, theta_phi_sampling, num_instances=1000, sh_orders=np.arange(0, 18, 1) - 1,
                        num_kernel=4):
    for sh_order in sh_orders:

        # create folders
        os.makedirs('/work/scratch/eschweiler/tests', exist_ok=True)
        os.makedirs('/work/scratch/eschweiler/tests/shorder{0}'.format(sh_order), exist_ok=True)

        # create score dict
        scores = {'TP': 0, 'FP': 0, 'FN': 0}

        # get harmonic transformations
        s2h = sampling2harmonics(sh_order, theta_phi_sampling)
        h2s = harmonics2sampling(sh_order, theta_phi_sampling)
        print('Using {0} harmonic coefficient(s)'.format(s2h.num_coefficients))

        num_patch = 0
        instance_count = 0

        while instance_count < num_instances:

            # get data
            empty_mask = True
            while empty_mask:
                _, data = data_gen.__getitem__(num_patch)
                data = data[0, ..., 0]
                labels, counts = np.unique(data, return_counts=True)
                if len(labels) > 2 and all([c < 25000 for c in counts[1:]]):
                    empty_mask = False
            save_image(data[2:-2, 2:-2, 2:-2],
                       '/work/scratch/eschweiler/tests/shorder{0}/shorder{0}_num{1}_true.tif'.format(sh_order,
                                                                                                     num_patch),
                       axes_order=data_gen.mask_config['axes_order'][:-1])

            # sample each instance
            instances, r_sampling, centroids = spherical_instance_sampling(data, theta_phi_sampling, bg_values=[0],
                                                                           verbose=False)

            # use harmonic transformation
            if sh_order >= 0:
                r_harmonics = s2h.convert(r_sampling)
                r_sampling = h2s.convert(r_harmonics)

                # reconstruct the mask
            for num_instance in range(len(r_sampling)):

                print('\r' * 22 + 'Progress {0:0>2d}% ({1:0>3d}/{2:0>3d})'.format(
                    int((instance_count + 1) / num_instances * 100), instance_count + 1, num_instances, end='\r'))

                centroid = centroids[num_instance]
                instance_sampled = descriptors2image([[centroid, ], [1, ], [r_sampling[num_instance]]],
                                                     theta_phi_sampling=theta_phi_sampling, shape=data.shape, thresh=0,
                                                     num_kernel=num_kernel, verbose=False)
                instance_sampled = instance_sampled[2:-2, 2:-2, 2:-2]

                gt_label = data[int(centroid[0]), int(centroid[1]), int(centroid[2])]
                if gt_label == 0:
                    continue
                instance_data = data == gt_label
                instance_data = instance_data[2:-2, 2:-2, 2:-2].astype(np.uint16)

                # save the images
                save_image(instance_sampled,
                           '/work/scratch/eschweiler/tests/shorder{0}/shorder{0}_num{1}_sampled{2}.tif'.format(sh_order,
                                                                                                               num_patch,
                                                                                                               num_instance),
                           axes_order=data_gen.mask_config['axes_order'][:-1])

                # calculate scores
                instance_sampled = instance_sampled.astype(np.bool)
                instance_data = instance_data.astype(np.bool)

                scores['TP'] += np.sum(np.logical_and(instance_sampled, instance_data))
                scores['FP'] += np.sum(np.logical_and(instance_sampled, ~instance_data))
                scores['FN'] += np.sum(np.logical_and(~instance_sampled, instance_data))

                instance_count += 1
                if instance_count >= num_instances:
                    break

            num_patch += 1

        scores['DICE'] = (2 * scores['TP'] + 1e-10) / (2 * scores['TP'] + scores['FP'] + scores['FN'] + 1e-10)
        scores['IOU'] = (scores['TP'] + 1e-10) / (scores['TP'] + scores['FP'] + scores['FN'] + 1e-10)
        with open('/work/scratch/eschweiler/tests/shorder{0}_results.csv'.format(sh_order), 'w') as f:
            for key in scores.keys():
                f.write("%s,%s\n" % (key, np.round(scores[key], decimals=2)))


def plot_sampling_error(sh_orders=np.arange(0, 18, 1) - 1):
    results = []

    for sh_order in sh_orders:
        # create folders
        filepath = os.path.join('/work/scratch/eschweiler/tests/FluoSim1000',
                                'shorder' + str(sh_order) + '_results.csv')
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            result_dict = dict((row[0], float(row[1])) for row in reader)
            # results.append(result_dict['DICE'])
            results.append(result_dict['TP'] / (result_dict['TP'] + result_dict['FP'] + result_dict['FN']))

    plt.figure(figsize=[6, 4])
    plt.plot([results[0], ] * 17, '--', color='darkorange', linewidth=3, label='Raw Sampling')
    plt.plot(np.arange(0, 17, 2), results[1::2], color='steelblue', marker='o', linewidth=3,
             label='With Spherical Harmonics')
    plt.ylim([0.5, 1])
    plt.xlim([0, 16])
    plt.ylabel('Mean IoU', fontsize=14)
    plt.xlabel('Harmonic coefficients', fontsize=14)
    plt.yticks(np.arange(0.5, 1.05, 0.1), fontsize=12)
    plt.xticks(np.arange(0, 17, 2), (np.arange(0, 17, 2) + 1) ** 2, fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.tight_layout()


def membrane_validation(filepath, gt_path=None, cropping=True, prob_thresh=[0.5, 0.8, 0.5]):
    # load ground truth
    if gt_path is None:
        gt_path = '/images/BiomedicalImageAnalysis/MembraneSegmentation/Validation/pCLV3-validation-center-crop/pCLV3-raw_x_352_y_336_z_006_244x244x496_LabelsWithoutBoundaries.tif'
    instances_gt = np.transpose(io.imread(gt_path), (1, 2, 0))
    mask_gt = to_onehot(instances2multiclass(instances_gt[..., np.newaxis], bg_values=[1]), labels=[1, 2, 3])

    # load prediction
    mask_pred = np.transpose(io.imread(filepath), (1, 2, 0, 3))
    if cropping:
        mask_pred = mask_pred[6:-6, 6:-6, 6:-6, :]
    mask_pred[..., 0] = mask_pred[..., 0] > mask_pred.max() * prob_thresh[0]
    mask_pred[..., 1] = mask_pred[..., 1] > mask_pred.max() * prob_thresh[1]
    mask_pred[..., 2] = mask_pred[..., 2] > mask_pred.max() * prob_thresh[2]

    f1_scores = multiclass_f1_score(mask_gt, mask_pred)

    return f1_scores


def generate_2D_PNAS(plants=[0, 1, 2, 3, 4, 5], num_samples=200,
                     save_path='/work/local/Daten/MembraneSynthesis_validation_multiclass'):
    plant_paths = ['/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant1',
                   '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant2',
                   '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant4',
                   '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant13',
                   '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant15',
                   '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant18']
    plant_paths = [plant_paths[p] for p in plants]

    plant_ids = [1, 2, 4, 13, 15, 18]
    plant_ids = [plant_ids[p] for p in plants]

    for plant_id, plant_path in zip(plant_ids, plant_paths):

        os.makedirs(os.path.join(save_path, 'plant' + str(plant_id), 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'plant' + str(plant_id), 'images'), exist_ok=True)

        masklist = glob.glob(os.path.join(plant_path, 'segmentation_tiffs', '*.tif'))
        image_list = glob.glob(os.path.join(plant_path, 'processed_tiffs', '*.tif'))

        sample_count = 0
        while sample_count < num_samples:

            # choose random file
            mask_path = masklist[np.random.randint(0, high=len(masklist))]

            # get corresponding image
            image_path = [i for i in image_list if '/' + os.path.split(mask_path)[1][:6] in i and 'YFP' in i]
            if len(image_path) > 1:
                raise ValueError('Multiple corresponding images found!')
            else:
                image_path = image_path[0]

            # load file
            mask = io.imread(mask_path)
            image = io.imread(image_path)

            while 1:
                # choose random layer
                rnd_layer = np.random.randint(0, mask.shape[0] - 30)
                mask_patch = mask[rnd_layer, ...]

                # get size of each instance
                regions = regionprops(mask_patch)
                areas = []
                for props in regions[1:]:  # exclude background label (1)
                    areas.append(props.area)

                # check for flawless segmentations
                if len(areas) > 0:
                    if np.max(areas) < 4 * np.mean(areas):
                        # extract membrane positions
                        mask_patch = instances2multiclass(mask_patch[..., np.newaxis], bg_values=[0, 1])
                        mask_patch = to_onehot(mask_patch, labels=[1, 2, 3])
                        # mask_patch = mask_patch - erosion(mask_patch, selem=np.ones((5,5)))
                        # mask_patch = mask_patch > 0
                        mask_patch = 255 * mask_patch.astype(np.uint8)

                        # get corresponding image
                        image_patch = image[rnd_layer, ...]
                        image_patch = image_patch.astype(np.uint8)

                        # save patches
                        save_name = 'mask' + str(sample_count) + '_' + str(rnd_layer) + '_' + os.path.split(mask_path)[
                                                                                                  1][:-4] + '.tif'
                        io.imsave(os.path.join(save_path, 'plant' + str(plant_id), 'masks', save_name), mask_patch)
                        save_name = 'image' + str(sample_count) + '_' + str(rnd_layer) + '_' + os.path.split(mask_path)[
                                                                                                   1][:-4] + '.tif'
                        io.imsave(os.path.join(save_path, 'plant' + str(plant_id), 'images', save_name), image_patch)

                        sample_count += 1
                        break


def generate_2D_crops(folderlist, num_samples_per_file=10, data_shape=(512, 512),
                      save_path='/work/local/Daten/MembraneSynthesis2D_validation_multiclass/original/salvadi2'):
    # folderlist = ['/images/BiomedicalImageAnalysis/MembraneSegmentation_SavaldiGoldsteinTechnion/2019_02_15_Data/bri1_20170802_07',
    #              '/images/BiomedicalImageAnalysis/MembraneSegmentation_SavaldiGoldsteinTechnion/2019_02_15_Data/bri1_20170328_01',
    #              '/images/BiomedicalImageAnalysis/MembraneSegmentation_SavaldiGoldsteinTechnion/2019_02_15_Data/bri1_20170817',
    #              '/images/BiomedicalImageAnalysis/MembraneSegmentation_SavaldiGoldsteinTechnion/2019_02_15_Data/col_BL_20171217_A',
    #              '/images/BiomedicalImageAnalysis/MembraneSegmentation_SavaldiGoldsteinTechnion/2019_02_15_Data/col_BL_20171217_B']

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)
    sample_count = 0

    for folder in folderlist:

        files = glob.glob(os.path.join(folder, '*tif'))

        image_list = sorted([f for f in files if not '_Segmentation' in f])
        mask_list = sorted([f for f in files if '_Segmentation' in f])

        for image_file, mask_file in zip(image_list, mask_list):

            num_patch = 0
            while num_patch < num_samples_per_file:

                data_config = {'shape': data_shape,
                               'grayscale': True,
                               'axes_order': (2, 1, 0, 3)}

                # Load and normalize the image
                image, data_config = load_tiff(image_file, data_config)
                image = image - image.min()
                image = image / image.max() * 255
                image = image.astype(np.uint8)

                # Load and transform the mask
                mask, data_config = load_tiff(mask_file, data_config)

                # Get random layer
                rnd_layer = np.random.randint(0, image.shape[-2])

                # Get and transform the mask patch
                mask_patch = mask[..., rnd_layer, 0]
                mask_patch = instances2multiclass(mask_patch[..., np.newaxis], bg_values=[0, 1])
                mask_patch = to_onehot(mask_patch, labels=[1, 2, 3])
                mask_patch = 255 * mask_patch.astype(np.uint8)

                if np.sum(mask_patch[..., 1]) > 20 * 5 * 255:  # 5 pixels with value 255 per detection
                    # Save the mask patch
                    save_name = 'mask' + str(sample_count) + '_' + str(rnd_layer) + '_' + os.path.split(image_file)[1]
                    io.imsave(os.path.join(save_path, 'masks', save_name), mask_patch)

                    # Get the image patch
                    image_patch = image[..., rnd_layer, 0]
                    # Save the image patch
                    save_name = 'image' + str(sample_count) + '_' + str(rnd_layer) + '_' + os.path.split(image_file)[1]
                    io.imsave(os.path.join(save_path, 'images', save_name), image_patch)

                    num_patch += 1
                    sample_count += 1


def create_2D_PNAS_folds(data_path='/work/local/Daten/MembraneSynthesis2D_validation_multiclass',
                         experiment_path='global_correspondence', original_path='original'):
    plants = ['plant1', 'plant2', 'plant4', 'plant13', 'plant15', 'plant18']
    train_indices = [[0], [1], [2], [3], [4], [5]]  # [[2,3,4,5], [0,1,4,5], [0,1,2,3]]
    test_indices = [[1], [0], [3], [2], [5], [4]]  # [[0,1], [2,3], [4,5]]

    for fold in range(len(train_indices)):

        ## Generate train set

        train_images = []
        train_masks = []

        train_idx = train_indices[fold]

        for train_id in train_idx:
            train_images.extend(get_files(os.path.join(data_path, experiment_path, plants[train_id], 'images'),
                                          pre_path=os.path.join(experiment_path, plants[train_id], 'images'),
                                          extension='tif'))
            train_masks.extend(get_files(os.path.join(data_path, experiment_path, plants[train_id], 'masks'),
                                         pre_path=os.path.join(experiment_path, plants[train_id], 'masks'),
                                         extension='tif'))

        write_list(train_images, train_masks,
                   save_path=os.path.join(data_path, experiment_path, 'fold' + str(fold + 1)), test_split=0,
                   val_split=0)

        ## Generate test set

        test_images = []
        test_masks = []

        test_idx = test_indices[fold]

        for test_id in test_idx:
            test_images.extend(get_files(os.path.join(data_path, original_path, plants[test_id], 'images'),
                                         pre_path=os.path.join(original_path, plants[test_id], 'images'),
                                         extension='tif'))
            test_masks.extend(get_files(os.path.join(data_path, original_path, plants[test_id], 'masks'),
                                        pre_path=os.path.join(original_path, plants[test_id], 'masks'),
                                        extension='tif'))

        write_list(test_images, test_masks, save_path=os.path.join(data_path, experiment_path, 'fold' + str(fold + 1)),
                   test_split=1, val_split=0)


def resize_SASHIMI_images(data_path='/work/local/Daten/MembraneSynthesis2D_validation_multiclass/original/',
                          new_size=(512, 512)):
    plants = ['plant1', 'plant2', 'plant4', 'plant13', 'plant15', 'plant18']

    for plant in plants:

        filelist = glob.glob(os.path.join(data_path, plant, 'masks', '*.tif'))

        for file in filelist:
            im = io.imread(file)

            # im = im[...,0]
            # im = resize(im, new_size, anti_aliasing=True)
            # im = im - im.min()
            # im = im / im.max()
            # im = im * 255
            # im = im.astype(np.uint8)

            im = imresize(im, new_size, interp='nearest')

            io.imsave(file, im)


def get_SASHIMI_segmentation_scores(data_path='/work/scratch/eschweiler/results/'):
    scores_bg = []
    scores_mem = []
    scores_cent = []

    for experiment in ['no_correspondence', 'global_correspondence', 'shape_correspondence',
                       'original']:  # global_correspondence

        scores_bg_exp = []
        scores_mem_exp = []
        scores_cent_exp = []

        for fold in range(3):
            path = os.path.join(data_path, 'membrane_synthesis_SASHIMI_' + experiment + '_fold' + str(fold + 1),
                                'predictions', 'scores.txt')
            scores = np.loadtxt(path, delimiter=';')
            scores_bg_exp.extend(scores[:-1, 0])
            scores_mem_exp.extend(scores[:-1, 1])
            scores_cent_exp.extend(scores[:-1, 2])

        scores_bg.append(scores_bg_exp)
        scores_mem.append(scores_mem_exp)
        scores_cent.append(scores_cent_exp)

    plt.rcParams.update({'font.size': 32})

    fig_bg, ax_bg = plt.subplots(figsize=(10, 10))
    fig_bg.subplots_adjust(left=0.17)
    ax_bg.set_title('Background', pad=20)
    ax_bg.boxplot(scores_bg)
    ax_bg.set_ylabel('F1-Score', labelpad=5)
    ax_bg.set_ylim([0, 1])
    ax_bg.set_xticklabels(
        [r'$\mathcal{D}_{naive}$', r'$\mathcal{D}_{global}$', r'$\mathcal{D}_{local}$', r'$\mathcal{D}_{orig}$'])

    fig_mem, ax_mem = plt.subplots(figsize=(10, 10))
    fig_mem.subplots_adjust(left=0.17)
    ax_mem.set_title('Membrane', pad=20)
    ax_mem.boxplot(scores_mem)
    ax_mem.set_ylabel('Boundary F1-Score', labelpad=5)
    ax_mem.set_ylim([0, 1])
    ax_mem.set_xticklabels(
        [r'$\mathcal{D}_{naive}$', r'$\mathcal{D}_{global}$', r'$\mathcal{D}_{local}$', r'$\mathcal{D}_{orig}$'])

    fig_cent, ax_cent = plt.subplots(figsize=(10, 10))
    fig_cent.subplots_adjust(left=0.17)
    ax_cent.set_title('Centroids', pad=20)
    ax_cent.boxplot(scores_cent)
    ax_cent.set_ylabel('Detection Accuracy', labelpad=5)
    ax_cent.set_ylim([0, 1])
    ax_cent.set_xticklabels(
        [r'$\mathcal{D}_{naive}$', r'$\mathcal{D}_{global}$', r'$\mathcal{D}_{local}$', r'$\mathcal{D}_{orig}$'])


def get_SASHIMI_matching_images(
        ref_file='/work/scratch/eschweiler/results/membrane_synthesis_SASHIMI_original_fold4/predictions/183_image.tif'):
    im_ref = io.imread(ref_file)

    data_paths = ['/work/scratch/eschweiler/results/membrane_synthesis_SASHIMI_shape_correspondence_fold4/predictions',
                  '/work/scratch/eschweiler/results/membrane_synthesis_SASHIMI_global_correspondence_fold4/predictions',
                  '/work/scratch/eschweiler/results/membrane_synthesis_SASHIMI_no_correspondence_fold4/predictions']

    for data_path in data_paths:
        filelist = glob.glob(os.path.join(data_path, '*_image.tif'))
        error_min = 1000
        min_file = filelist[0]
        for file in filelist:
            im = io.imread(file)
            error = np.sum(im_ref - im)
            if error < error_min:
                min_file = file
        print(min_file)


def corr_coef_images(im1, im2):
    product = np.mean((im1 - im1.mean()) * (im2 - im2.mean()))
    stds = im1.std() * im2.std()

    if stds == 0:
        return 0
    else:
        return product / stds


def get_SASHIMI_similarity_scores(data_path='/work/local/Daten/MembraneSynthesis2D_validation_multiclass'):
    scores_ssim = []
    scores_ncc = []

    for experiment in ['no_correspondence', 'global_correspondence', 'shape_correspondence']:

        scores_ssim_exp = []
        scores_ncc_exp = []

        for plant in ['plant4', 'plant13', 'plant15', 'plant18']:  # plant1'plant2',
            # get original files
            filelist_ref = glob.glob(os.path.join(data_path, 'original', plant, 'images', '*.tif'))

            # global_correspondence
            filelist_exp = glob.glob(os.path.join(data_path, experiment, plant, 'images', '*.tif'))
            for file_exp in filelist_exp:
                file_ref = get_close_matches(file_exp, filelist_ref)[0]
                # print('_'*20)
                # print(os.path.split(file_ref)[-1])
                # print(os.path.split(file_exp)[-1])

                im_exp = io.imread(file_exp)
                im_ref = io.imread(file_ref)

                ssim = compare_ssim(im_exp, im_ref)
                ncc = corr_coef_images(im_exp, im_ref)

                scores_ssim_exp.append(ssim)
                scores_ncc_exp.append(ncc)

        scores_ssim.append(scores_ssim_exp)
        scores_ncc.append(scores_ncc_exp)

    plt.rcParams.update({'font.size': 42})

    fig_ssim, ax_ssim = plt.subplots(figsize=(10, 10))
    fig_ssim.subplots_adjust(left=0.22)
    ax_ssim.boxplot(scores_ssim)
    ax_ssim.set_ylabel('SSIM', labelpad=25)
    ax_ssim.set_ylim([0, 1])
    ax_ssim.set_xticklabels([r'$\mathcal{D}_{naive}$', r'$\mathcal{D}_{global}$', r'$\mathcal{D}_{local}$'])

    fig_ncc, ax_ncc = plt.subplots(figsize=(10, 10))
    fig_ncc.subplots_adjust(left=0.22)
    ax_ncc.boxplot(scores_ncc)
    ax_ncc.set_ylabel('NCC')
    ax_ncc.set_ylim([-1, 1])
    ax_ncc.set_xticklabels([r'$\mathcal{D}_{naive}$', r'$\mathcal{D}_{global}$', r'$\mathcal{D}_{local}$'])


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (255, 255, 255), 2)

    return img


def create_2D_abstraction():
    image_files = [
        '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant18/processed_tiffs/0hrs_plant18_trim-acylYFP.tif', \
        '/images/BiomedicalImageAnalysis/CellSegmentation_Simulated/SBDE4/BenchmarkImages/r=000/item_0007_AdditiveGaussianNoiseImageFilter/rawImage_r=000_t=0350_AdditiveGaussianNoiseImageFilter_Out1.tif']
    mask_files = [
        '/images/BiomedicalImageAnalysis/MembraneSegmentation/PNAS/plant18/segmentation_tiffs/0hrs_plant18_trim-acylYFP_hmin_2_asf_1_s_2.00_clean_3.tif', \
        '/images/BiomedicalImageAnalysis/CellSegmentation_Simulated/SBDE4/LabelImages/labelImage_r=000_t=0350.tif']
    image_types = ['membrane', 'cells']

    save_path = '/work/local/Daten/2D_registration_test'

    for num_patch in range(5):
        for image_path, mask_path, image_type in zip(image_files, mask_files, image_types):

            # Load image and select 2D slice
            image = io.imread(image_path)
            rnd_idx = np.random.randint(0, image.shape[0])
            image = image[rnd_idx, ...]
            image = image - image.min()
            image = image / image.max()
            image = 255 * image
            image = image.astype(np.uint8)

            # Load mask and select 2D slice
            mask = io.imread(mask_path)
            mask = mask[rnd_idx, ...]
            mask = mask - mask.min()

            mask_foreground = mask > 0
            mask_foreground = mask_foreground.astype(np.float)
            mask_border = mask_foreground - morphology.binary_erosion(mask_foreground, structure=np.ones((5, 5)))
            mask_boundaries = mask - morphology.grey_erosion(mask, size=5)
            mask_boundaries = mask_boundaries > 0
            mask_boundaries = 255 * np.uint8(mask_boundaries)

            # Get centroids of each instance
            regions = regionprops(mask)
            centroids = [tuple([int(prop.centroid[1]), int(prop.centroid[0])]) for prop in regions]
            radii = [prop.equivalent_diameter / 2 for prop in regions]

            if 'membrane' in image_type:
                # For membranes, construct membrane positions
                rect = (0, 0, mask.shape[0], mask.shape[1])
                subdiv = cv2.Subdiv2D(rect)
                for centroid in centroids:
                    subdiv.insert(centroid)
                mask_abstract = np.zeros_like(mask)
                mask_abstract = draw_voronoi(mask_abstract, subdiv)
                mask_abstract = np.minimum(mask_abstract * mask_foreground + 255 * mask_border, 255)


            elif 'cells' in image_type:
                mask_abstract = np.zeros_like(mask)
                for centroid, radius in zip(centroids, radii):
                    mask_abstract = cv2.circle(mask_abstract, centroid, int(radius), 255, -1)

            else:
                mask_abstract = np.zeros_like(mask)

            mask_abstract = mask_abstract + 100 * np.random.rand(*mask_abstract.shape)
            mask_abstract = mask_abstract - mask_abstract.min()
            mask_abstract = mask_abstract / mask_abstract.max()
            mask_abstract = mask_abstract * 255

            io.imsave(os.path.join(save_path, image_type + '_image' + str(num_patch) + '.tif'), image.astype(np.uint8))
            io.imsave(os.path.join(save_path, image_type + '_mask' + str(num_patch) + '.tif'), mask_boundaries)
            io.imsave(os.path.join(save_path, image_type + '_mask_abstract' + str(num_patch) + '.tif'),
                      mask_abstract.astype(np.uint8))


def create_spherical_dummies(theta_phi_sampling, save_path, radius_range=[5, 15], radius_roughness=10,
                             image_shape=(512, 512, 256), num_images=50):
    # Create save folder
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)

    sampling_count = len(theta_phi_sampling)

    for image_count in range(num_images):

        image_mask = np.ones(image_shape, dtype=np.bool)
        # Remove locations at the image border
        image_mask[:radius_range[0] * 4, ...] = False
        image_mask[:, :radius_range[0] * 4, :] = False
        image_mask[:, :, :radius_range[0] * 4] = False
        image_mask[-radius_range[0] * 4:, ...] = False
        image_mask[:, -radius_range[0] * 4:, :] = False
        image_mask[:, :, -radius_range[0] * 4:] = False

        radii_sampling = []
        centroids = []

        num_instances = range(np.random.randint(50, 100))
        for instance_count in num_instances:
            # Get radius sampling
            radius = np.random.choice(np.arange(radius_range[0], radius_range[1] + 1))
            radius_sampling = np.random.randint(radius - radius_roughness, radius + radius_roughness + 1,
                                                size=sampling_count)
            radius_sampling = medfilt(radius_sampling, kernel_size=(11,))
            radii_sampling.append(radius_sampling)

            # Get centroid
            centroid_idx = np.transpose(np.nonzero(image_mask))
            rnd_idx = np.random.randint(0, centroid_idx.shape[0])
            centroid = centroid_idx[rnd_idx, :].copy()
            centroids.append(centroid)

            # Mark area as used
            image_mask[np.maximum(0, centroid[0] - radius_range[1] * 2):np.minimum(image_shape[0],
                                                                                   centroid[0] + radius_range[1] * 2), \
            np.maximum(0, centroid[1] - radius_range[1] * 2):np.minimum(image_shape[1],
                                                                        centroid[1] + radius_range[1] * 2), \
            np.maximum(0, centroid[2] - radius_range[1] * 2):np.minimum(image_shape[2],
                                                                        centroid[2] + radius_range[1] * 2)] = False

        # Create instance mask
        instance_mask = descriptors2image([centroids, [1, ] * len(centroids), radii_sampling],
                                          theta_phi_sampling=theta_phi_sampling, shape=image_shape)
        instance_mask = np.transpose(instance_mask, (2, 1, 0))
        io.imsave(os.path.join(save_path, 'masks', 'mask_{0}.tif'.format(image_count)), instance_mask)

        # Create corresponding image
        instance_image = instance_mask.copy() > 0
        instance_image = instance_image.astype(np.uint16)
        instance_image = add_gaussian_noise(instance_image, gauss_mean=1, gauss_scale=[1, 3], max_val=2 ** 16)
        instance_image = normalization(instance_image, norm_method='minmax_data')
        instance_image = instance_image * 2 ** 16
        io.imsave(os.path.join(save_path, 'images', 'image_{0}.tif'.format(image_count)),
                  instance_image.astype(np.uint16))










