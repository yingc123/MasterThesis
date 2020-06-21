"""
Created on Thu Dec 13 11:59:27 2018

@author: eschweiler
Modification: YingChen
"""


from data_handling import get_files, write_list
import os
import sys
import pickle
from data_handling import *
from utils import *
from sklearn.utils import class_weight
from libtiff import TIFF
from skimage.io import imread, imsave
from scipy import stats
import numpy as np
from utils import membrane_score_calculator
from data_loader import image_loader
from data_handling import read_list
from segmentation_pipeline import mask_config

def list_gen(test_split, val_split):
    path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/arabidopsis'
    # Get the file lists
    image_list = get_files('/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/arabidopsis/image/origin/', pre_path='', extension='h5')
    mask_list = get_files('/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/arabidopsis/mask/full_mask/', pre_path='', extension='h5')
    print('image:', image_list)
    print('mask:', mask_list)

    # Write the csv files
    write_list(image_list, mask_list, save_path='/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/arabidopsis/list/5',
               test_split=test_split, val_split=val_split)

def weights_calculate():
    mask_path='/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/1/'
    mask_list = get_files(os.path.join(mask_path, 'mask'), pre_path='mask',
                          extension='tif')
    #print(mask_list)
    flag = True
    for i in range(len(mask_list)):
        image, patch_params=load_tiff(os.path.join(mask_path, mask_list[i]))
        print(np.shape(image))
        if flag:
            whole_mask = image
            flag = False
        else:
            whole_mask = np.concatenate((whole_mask, image), axis=0)
    #print(np.shape(whole_mask))
    weights = class_weight.compute_class_weight('balanced', np.unique(whole_mask), np.reshape(whole_mask, (3892*256*256,)))
    print(weights)

def renamefile():
    path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/drosophila/mask/full_mask'
    for i in range(1,8):
        print(i)
        src = os.path.join(path, 'fused_tp_{}_ch_0_full_Masked_rotate-BDV.h5'.format(i))
        dst = os.path.join(path, 'fused_tp_{}_ch_0_Masked.h5'.format(i))
        os.rename(src, dst)

def read_pickle():
    with open('/work/scratch/ychen/segmentation/network_method/3D_U_Net/model/arabidopsis/full_mask/origin_1_250/general_config.json', 'rb') as gc:
        general_config = pickle.load(gc)
    print(general_config)

def tiff_modify():
    path = '/work/scratch/ychen/preprocessing/arabidopsis/plant18/512*512/full_mask'
    for x in os.listdir(path):
        im = imread(os.path.join(path, x), plugin='tifffile')
        for i in im:
            i[i > 1] = 65535
            i[i <= 1] = 0
        imsave(os.path.join(path, x), im, plugin='tifffile')

def weight_cal():
    path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/4/mask'
    f_list = [2,7,5,3]
    total_weight = 0
    for i in f_list:
        tif = TIFF.open(os.path.join(path, 'fused_tp_{}_ch_0_Masked_rotated.tif'.format(i)))
        image = tif.read_image()
        for sin_image in tif.iter_images():
            weights = class_weight.compute_class_weight('balanced', np.unique(sin_image),
                                                np.reshape(sin_image, (256*256,)))
            total_weight += weights

    print(total_weight/(len(f_list)*317))

def sum_tif():
    path = '/work/scratch/ychen/preprocessing/arabidopsis/plant1/512*512/full_mask'
    sum=0
    for x in range(20):
        im = imread(os.path.join(path, '{}hrs_plant1_trim-acylYFP_hmin_2_asf_1_s_2.00_clean_3-BDV.tif'.format(4*x)), plugin='tifffile')
        for i in im:
            sum+=1
    print(sum)
    print(sum/20)

def evaluation():
    #datatype=['arabidopsis', 'drosophila']
    pred_path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/model/other/full_mask/test150_1/whole_images/item_0004_SubtractImageFilter'
    mask_path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/drosophila/mask/boundary'
    list_path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net/data/drosophila/list/1_test.csv'
    test_list = read_list(list_path)
    test_list = [[os.path.join(pred_path, i), os.path.join(mask_path, m)] for i, m
                 in test_list]
    image_score_accumulator = membrane_score_calculator(class_thresh=0.5, image_safety_margin=None)
    for test_files in test_list:
        try:
            print('Calculating scores.')

            # Get the true mask
            true_params = mask_config.copy()
            true_params['shape'] = None
            true_params['start'] = None
            pred_loader = image_loader(test_files[0], patch_params=true_params)
            true_loader = image_loader(test_files[1], patch_params=true_params)
            # Add the current example
            image_score_accumulator.add_example(pred_loader.image, true_loader.image)
        except Exception as err:
            print('Error raised: ({0})\nProceed without score calculation...'.format(err))

def filepath_modify():
    path='\Images\BiomedicalImageAnalysis\MembraneSegmentation\Drosophila_KellerLab\TimeFused.Blending.LongRange'
    for x in range(80):
        doc='Dme_E1_SpiderGFP-HisRFP.TM00{:02d}_timeFused_blending'.format(x)
        file=os.path.join(path, doc, 'SPC0_TM0000_CM0_CM1_CHN00_CHN01.fusedStack')
        print(file)

# preprocess images
def histogram_rescale():
    data_path='/work/scratch/ychen/preprocessing/arabidopsis/plant18/256x256/origin'
    save_path = '/work/scratch/ychen/preprocessing/arabidopsis/plant18/256x256/rescale'
    for i in os.listdir(data_path):
        data=imread(os.path.join(data_path, i), plugin='tifffile')
        p15, p95 = np.percentile(data, (15, 95))
        img_rescale = exposure.rescale_intensity(data, in_range=(p15, p95))
        imsave(os.path.join(save_path, i), img_rescale, plugin='tifffile')

# preprocess images
def histogram_equalization():
    data_path='/work/scratch/ychen/preprocessing/arabidopsis/plant2/256x256/origin'
    save_path = '/work/scratch/ychen/preprocessing/arabidopsis/plant2/256x256/equalization'
    for i in os.listdir(data_path):
        data=imread(os.path.join(data_path, i), plugin='tifffile')
        img_HE = exposure.equalize_hist(data)
        imsave(os.path.join(save_path, i), img_HE, plugin='tifffile')

def main():
    histogram_equalization()

if __name__=='__main__':
    main()