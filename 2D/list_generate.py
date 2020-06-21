import os
import csv
import glob
import numpy as np
from shutil import copyfile

# read a specific filelist
def read_list(list_path):
    filelist = []
    try:
        with open(list_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                #row = [sanitycheck_path(r, use_local=use_local) for r in row]
                filelist.append(row)
    except:
        filelist = []
    return filelist

# get files from a specific directory
def get_files(folder, pre_path='', extension=''):
    filelist = sorted(glob.glob(os.path.join(os.path.normpath(folder), '*.' + extension)))

    filelist = [os.path.join(os.path.normpath(pre_path), os.path.basename(f)) for f in filelist]

    return filelist

# rename file
def concatenate():
    dir = '/work/scratch/ychen/segmentation/network_method/2D_U_Net/model/arabidopsis_plant_1/mask_bt1000/pred_result'
    for img in os.listdir(dir):
        #print(img)
        tp = img.split('-')
        #print(tp)
        slice = (tp[0]).split(':')[1]
        print(slice)
        #print(slice+tp[1])
        #os.rename(os.path.join(dir,img), os.path.join(dir, slice+tp[1]))

# Create file list of train, test and valdidation splits
def write_list(im_list, mask_list, save_path='', test_split=0.2, val_split=0.1, rnd_seed=None):
    save_path = os.path.normpath(save_path)

    assert len(im_list) == len(mask_list), 'Number of images and masks does not match'

    # get indices for all sets
    idx_list = np.arange(len(im_list))
    np.random.seed(rnd_seed)
    np.random.shuffle(idx_list)
    im_list = [im_list[i] for i in idx_list]
    mask_list = [mask_list[i] for i in idx_list]

    test_count = int(len(idx_list) * test_split)
    val_count = int((len(idx_list) - test_count) * val_split)
    train_count = int(len(idx_list) - val_count - test_count)
    print(test_count, val_count, train_count)
    # write csv files
    if train_count > 0:
        with open(save_path + '_train.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[0:train_count], mask_list[0:train_count]))
    if test_count > 0:
        with open(save_path + '_test.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[train_count:train_count + test_count], \
                                 mask_list[train_count:train_count + test_count]))
    if val_count > 0:
        with open(save_path + '_valid.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(zip(im_list[train_count + test_count:], mask_list[train_count + test_count:]))

def main():
    # data_path='/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/6(fullmask)/arabidopsis/data1'
    #
    # # Get the file lists
    # image_list = get_files(os.path.join(data_path, 'image'), pre_path='image/', extension='png')
    # #print(image_list)
    # mask_list = get_files(os.path.join(data_path, 'label'),pre_path='label/', extension='png')
    # #print(mask_list)
    # # Write the csv files
    # write_list(image_list, mask_list, save_path=os.path.join(data_path, 'data'), test_split=0.2, val_split=0.2)
    #
    # train_path = os.path.join(data_path, 'train')
    # os.makedirs(train_path, exist_ok=True)
    # train_image = os.path.join(train_path, 'image')
    # os.makedirs(train_image, exist_ok=True)
    # train_label = os.path.join(train_path, 'label')
    # os.makedirs(train_label, exist_ok=True)
    #
    # valid_path = os.path.join(data_path, 'valid')
    # os.makedirs(valid_path, exist_ok=True)
    # valid_image = os.path.join(valid_path, 'image')
    # os.makedirs(valid_image, exist_ok=True)
    # valid_label = os.path.join(valid_path, 'label')
    # os.makedirs(valid_label, exist_ok=True)
    #
    # test_path = os.path.join(data_path, 'test')
    # os.makedirs(test_path, exist_ok=True)
    # test_image = os.path.join(test_path, 'image')
    # os.makedirs(test_image, exist_ok=True)
    # test_label = os.path.join(test_path, 'label')
    # os.makedirs(test_label, exist_ok=True)
    #
    # train_list = read_list(os.path.join(data_path, 'data_train.csv'))
    # for i in train_list:
    #     copyfile(os.path.join(data_path, i[0]), os.path.join(train_path, i[0]))
    #     copyfile(os.path.join(data_path, i[1]), os.path.join(train_path, i[1]))
    #
    # valid_list = read_list(os.path.join(data_path, 'data_valid.csv'))
    # for i in valid_list:
    #     copyfile(os.path.join(data_path, i[0]), os.path.join(valid_path, i[0]))
    #     copyfile(os.path.join(data_path, i[1]), os.path.join(valid_path, i[1]))
    #
    # test_list = read_list(os.path.join(data_path, 'data_test.csv'))
    # for i in test_list:
    #     copyfile(os.path.join(data_path, i[0]), os.path.join(test_path, i[0]))
    #     copyfile(os.path.join(data_path, i[1]), os.path.join(test_path, i[1]))
    concatenate()


if __name__== '__main__':
    main()