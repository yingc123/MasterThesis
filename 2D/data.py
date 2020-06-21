from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from sklearn.utils import class_weight

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

# normalization
def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img/255
        img[img < 0] = 0
        mask = mask/255
        #mask[mask < 0] = 0
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        # np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/6(fullmask)/drosophila/test/img.npy', img)
        # np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/6(fullmask)/drosophila/test/mask.npy', mask)
    return (img,mask)

def create_random_array(percent, sav_path):
    care_region = int(percent * 65536)
    x=np.ones(care_region)
    y=np.ones(65536-care_region)
    arr=np.concatenate((x,y))
    np.random.shuffle(arr)
    random_arr = np.reshape(arr, (256, 256))
    np.save(sav_path, random_arr)
    return random_arr

# Data generator function, apply data augmentation
def augGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                    save_to_dir = None, target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        interpolation='nearest')
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        interpolation='nearest')
    train_generator = zip(image_generator, mask_generator)
    while True:
        for (img,mask) in train_generator:
            img,mask = adjustData(img,mask)
            yield (img,mask)

# Data generator function, without data augmentation
def normalGenerator(dir_path, target_size = (256,256), as_gray = True, mask=False):
    if mask:
        for t in os.listdir(dir_path):
            img = io.imread(os.path.join(dir_path, t), as_gray=as_gray)
            img = img / 255
            img = trans.resize(img, target_size)
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
            img[img <= 0] = 0
            img[img > 0] = 1
            yield img
    else:
        for t in os.listdir(dir_path):
            img = io.imread(os.path.join(dir_path, t), as_gray=as_gray)
            img = img / 255
            img = trans.resize(img, target_size)
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
            img[img < 0] = 0
            yield img

def weightGenerator(train_path, train_list):
    flag=True
    for t in train_list:
        if t < 10:
            mask = io.imread(os.path.join(train_path, 'label', "label_00{}.png".format(t)), as_gray=True)
        elif t >= 100:
            mask = io.imread(os.path.join(train_path, 'label', "label_{}.png".format(t)), as_gray=True)
        else:
            mask = io.imread(os.path.join(train_path, 'label', "label_0{}.png".format(t)), as_gray=True)

        if flag:
            arr = mask
            flag = False
        else:
            arr = np.concatenate((arr, mask), axis=0)
    return arr

def geneTrainNpy(image_path,mask_path, image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path, results, test_img_path):
    idx = 0
    for t in os.listdir(test_img_path):
        mask = results[idx][:, :, 0]
        #mask[mask < 0] = 0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        #io.imsave(os.path.join(save_path, "predict_{}.png".format(t)), mask.astype(dtype=np.uint8))
        mask = mask * 255
        #np.save(os.path.join(save_path, 'pred.npy'), mask)
        io.imsave(os.path.join(save_path, 'pred_'+t), mask.astype(np.uint8))
        idx += 1