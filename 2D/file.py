# import numpy as np
# import os
# from random import shuffle
#
# sequence = [i for i in range(317)]
# shuffle(sequence)
# np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/shuffle.npy', sequence)
#
#
# whole = np.load('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/shuffle.npy').tolist()
# x = len(whole)//6
#
#
#
# valid_npy = whole[:x]
# test_npy = whole[x:2*x]
# train_npy = whole[2*x:]
# print(len(whole), whole)
# print(len(train_npy), train_npy)
# print(len(valid_npy), valid_npy)
# print(len(test_npy), test_npy)
# #
# # os.remove('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/train.npy')
# # os.remove('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/valid.npy')
# # os.remove('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/test.npy')
# np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/train.npy', train_npy)
# np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/valid.npy', valid_npy)
# np.save('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/test.npy', test_npy)
#
#
# for v in valid_npy:
#     if v < 10 :
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_00{}.png".format(v),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/image/Image_00{}.png".format(v))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_00{}.png'.format(v),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/label/label_00{}.png'.format(v))
#     elif v >= 100:
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_{}.png".format(v),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/image/Image_{}.png".format(v))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_{}.png'.format(v),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/label/label_{}.png'.format(v))
#     else:
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_0{}.png".format(v),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/image/Image_0{}.png".format(v))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_0{}.png'.format(v),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/valid/label/label_0{}.png'.format(v))
# #
# for t in test_npy:
#     if t < 10 :
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_00{}.png".format(t),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/image/Image_00{}.png".format(t))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_00{}.png'.format(t),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/label/label_00{}.png'.format(t))
#     elif t >= 100:
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_{}.png".format(t),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/image/Image_{}.png".format(t))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_{}.png'.format(t),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/label/label_{}.png'.format(t))
#     else:
#         os.rename("/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/image/Image_0{}.png".format(t),\
#                   "/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/image/Image_0{}.png".format(t))
#         os.rename('/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/train/label/label_0{}.png'.format(t),\
#                   '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/label/label_0{}.png'.format(t))

# # from PIL import Image
# # import numpy
# # im = Image.open('/work/scratch/ychen/images/raw_images/fused_tp_1_ch_0_Masked_SubtractImageFilter_Out1 (copy).tif')
# #
# # imarray = numpy.array(im)
# # print(imarray.shape, im.size)
#
#
# from libtiff import TIFF
# tif = TIFF.open('/work/scratch/ychen/images/raw_images/fused_tp_1_ch_0_Masked_SubtractImageFilter_Out1 (copy).tif')
# image = tif.read_image()

import numpy as np
import os
from model import *
from data import *

test_path = '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test'
sav_path = '/work/scratch/ychen/segmentation/network_method/2D_U_Net/data/4/test/label'
model_path = '/work/scratch/ychen/segmentation/network_method/2D_U_Net/model/unet_membrane_4.hdf5'
n_img_bit = 255

test_list = np.load(os.path.join(test_path, 'test.npy')).tolist()
test_img = testGenerator(test_path, test_list, n_img_bit)

model = unet()
model.load_weights(model_path)
results = model.predict_generator(test_img, len(test_list), verbose=1)
print(results)
saveResult(sav_path, results, test_list)