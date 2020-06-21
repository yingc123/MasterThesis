# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:59:27 2018

@author: eschweiler
Modification: YingChen
"""

import os
import sys
import importlib
import copy
import numpy as np
import pandas
import pickle
import argparse
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.utils import class_weight

# Append the path containing all customized utility functions
parent_path = os.path.abspath(os.path.join(os.path.abspath(''), os.pardir))
sys.path.insert(0, parent_path)

from data_handling import *
from data_loader import image_streamer, image_loader
from utils import *
from misc import predict_whole_image
from networks import f1_score, mean_iou
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy



###############################################################################
''' PARSER '''
###############################################################################
parser = argparse.ArgumentParser(description='General configurations')
parser.add_argument('--comment', type=str, default='_1_', help='Comment to quickly create a unique identifier for the experiment')
parser.add_argument('--network', type=str, default='unet3D', help='which network architecture')
parser.add_argument('--monitor', type=str, default='val_loss', help='monitor base for saving model')
parser.add_argument('--mode', type=str, default='min', help='mode for monitor evaluation')
parser.add_argument('--src', type=int, default=0, help='choose which data to train')
parser.add_argument('--img_tf', type=int, default=0, help='different dataset')
parser.add_argument('--mask_tf', type=int, default=0, help='different masks')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Number of patches to test on')
parser.add_argument('--overlap', type=int, default=40, help='Overlap of patches during evaluation')
parser.add_argument('--num_test_samples', type=int, default=100, help='Number of patches to test on')
parser.add_argument('--robustness', type=bool, default=True, help='Randomly change for original images')
parser.add_argument('--no_eval_patches', dest='eval_patches', action='store_true', help='No evaluation on patches')
parser.add_argument('--no_eval_images', dest='eval_images', action='store_false', help='No evaluation on whole images')
parser.add_argument('--no_training', dest='perform_training', action='store_true', help='Perform no training')
parser.add_argument('--overwrite', dest='overwrite_config', action='store_false', help='Overwrite the existing configuration')

args = parser.parse_args()



###############################################################################
''' CONFIGURATIONS '''
###############################################################################
path = '/work/scratch/ychen/segmentation/network_method/3D_U_Net'
source = ['arabidopsis_plant1', 'arabidopsis_plant2', 'arabidopsis_plant4', 'arabidopsis_plant13', 'arabidopsis_plant15',
          'arabidopsis_plant18', 'drosophila', 'drosophila_new_1', 'drosophila_new_2']
img_transform = ['origin', 'histogram_equalization', 'histogram_rescale']
mask_transform = ['boundary', 'full_mask', 'boundary_thick']
src = source[args.src]
img_tf = img_transform[args.img_tf]
mask_tf = mask_transform[args.mask_tf]

general_config = {'rnd_seed': 0,
                  'network': 'unet3D',
                  'comment':  img_tf + args.comment + str(args.epochs),
                  'save_path': os.path.join(path, 'model', src, mask_tf),
                  'img_path': os.path.join(path, 'data', src, 'image', img_tf),
                  'mask_path': os.path.join(path, 'data', src, 'mask', mask_tf),
                  'list_path': os.path.join(path, 'data', src, 'list', args.comment[1]),
                  'epochs': args.epochs,
                  'leanring_rate': args.learning_rate,
                  'batch_size': args.batch_size,
                  'activation_fcn': 'relu',
                  'apply_batch_norm': True,
                  'class_thresh': 0.5,
                  'iou_thresh':0.9,
                  'save_best': True,
                  'overlap': args.overlap,
                  'augmentation': {'rotations': [0,1,2,3],
                                   'mirroring': 0.5
                                   }
                  }
general_config['save_path'] = os.path.join(general_config['save_path'], general_config['comment'])
os.makedirs(general_config['save_path'], exist_ok=True)

# Load the existing general config, if there is one. Else save the current config
if os.path.isfile(os.path.join(general_config['save_path'], 'general_config.json')) and not args.overwrite_config:
    with open(os.path.join(general_config['save_path'], 'general_config.json'), 'rb') as gc:
        general_config = pickle.load(gc)
else:
    with open(os.path.join(general_config['save_path'], 'general_config.json'), 'wb') as gc:
        pickle.dump(general_config, gc, protocol=pickle.HIGHEST_PROTOCOL)


# Load existing image config or save the current one
if os.path.isfile(os.path.join(general_config['save_path'], 'image_config.json')) and not args.overwrite_config:
    with open(os.path.join(general_config['save_path'], 'image_config.json'), 'rb') as ic:
        image_config = pickle.load(ic)
else:
    image_config = {'shape': (64, 64, 64),
                    'grayscale': True,
                    'load_fcn': load_bdv_hdf5,
                    'axes_order': (2, 1, 0, 3),
                    'transforms': {'fcn': [gamma_transform, add_gaussian_noise, intensity_scale, normalization],
                                   'norm_method': 'scale',
                                   'max_val': 2 ** 15,
                                   'intens_scale_prob': 0.5,
                                   'intens_scale_min': 0.75,
                                   'intens_scale_max': 1.1,
                                   'gauss_prob': 0.25,
                                   'gauss_scale': [0, int(0.2 * 2 ** 15)],
                                   'gauss_sigma': 0.1,
                                   'gauss_mean': 0,
                                   'gamma_prob': 0.25,
                                   'gamma_min': 0.5,
                                   'gamma_max': 1.0}
                    }
    with open(os.path.join(general_config['save_path'], 'image_config.json'), 'wb') as ic:
        pickle.dump(image_config, ic, protocol=pickle.HIGHEST_PROTOCOL)


# Load existing mask config or save the current one
if os.path.isfile(os.path.join(general_config['save_path'], 'mask_config.json')) and not args.overwrite_config:
    with open(os.path.join(general_config['save_path'], 'mask_config.json'), 'rb') as mc:
        mask_config = pickle.load(mc)
else:
    mask_config = {'shape': (64,64,64),
                   'grayscale': True,
                   'load_fcn': load_bdv_hdf5,
                   'axes_order': (2,1,0,3),
                   'transforms':{'fcn':[normalization, instances2binaryclass],
                                 'max_val': 2 ** 15 - 1
                                 }
                   }
    with open(os.path.join(general_config['save_path'], 'mask_config.json'), 'wb') as mc:
        pickle.dump(mask_config, mc, protocol=pickle.HIGHEST_PROTOCOL)

def weightGenerator(gen, list):
    flag = True
    for num_train in range(len(list)):
        img, mask = gen.__getitem__(num_train)
        print(mask.shape())
        if flag:
            whole_mask = mask
            flag = False
        else:
            whole_mask = np.concatenate((whole_mask, mask), axis=0)
    return whole_mask

def main():
    # Import the network architecture
    net_module = importlib.import_module('networks')
    net_architecture = getattr(net_module, general_config['network'])

    ###############################################################################
    ''' TRAINING '''
    ###############################################################################

    if args.perform_training:
        # Get the train data
        train_list = read_list(os.path.normpath(general_config['list_path']+'_train.csv'))
        train_list = [[os.path.join(general_config['img_path'], i), os.path.join(general_config['mask_path'], m)] for i,m in train_list]
        #print(train_list)
        num_samples = 320
        # num_example means how much patches used
        train_gen = image_streamer(train_list, image_config, mask_config, num_samples=num_samples,  **general_config)
        train_gen = copy.deepcopy(train_gen)

        # Get the validation data
        val_list = read_list(os.path.normpath(general_config['list_path']+'_val.csv'))
        val_list = [[os.path.join(general_config['img_path'], i), os.path.join(general_config['mask_path'], m)] for i,m in val_list]
        val_gen = image_streamer(val_list, image_config, mask_config,  **general_config) if not val_list==[] else None
        val_gen = copy.deepcopy(val_gen)

        # Set up the model
        model = net_architecture(input_shape=image_config['shape'],lr=args.learning_rate,
                                  metrics=[f1_score, BinaryCrossentropy() ],verbose=True, **general_config)


        # Set up a model checkpoint
        os.makedirs(general_config['save_path'], exist_ok=True)
        model_checkpoint = ModelCheckpoint(os.path.join(general_config['save_path'], 'model.hdf5'),\
                                           monitor=args.monitor, mode=args.mode, verbose=1, save_best_only=general_config['save_best'])

        print('Model checkpoint set to {0}'.format(os.path.join(general_config['save_path'], 'model.hdf5')))

        # save model architecture and summary
        with open(os.path.join(general_config['save_path'], 'model.json'),'w') as mh:
            mh.write(model.to_json())
        with open(os.path.join(general_config['save_path'], 'architecture.txt'),'w') as fh:
            fh.write('Model: {0}\n\n'.format(general_config['network']))
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Start training
        train_history = model.fit_generator(generator=train_gen, epochs=general_config['epochs'],\
                                            steps_per_epoch=num_samples//args.batch_size, callbacks=[model_checkpoint], \
                                            validation_data=val_gen, validation_steps=80, shuffle=True)
        #print(train_history.history)
        # Save training history
        df_outcomes = pandas.DataFrame(data=train_history.history)
        df_outcomes.to_csv(os.path.join(general_config['save_path'], 'history.csv'), sep=';')
        try:
            # plot and save figure
            ax = df_outcomes.plot(title=general_config['comment'], grid=True, xlim=(0, train_history.epoch[-1]+1),
                                  xticks=np.arange(0, train_history.epoch[-1]+1, 10*(1+(train_history.epoch[-1]+1)//100)), figsize=(12,10))
            ax.set_xlabel('epochs')
            ax.set_ylabel('scores')
            ax.figure.savefig(os.path.join(general_config['save_path'], 'history.tif'))
            plt.close(ax.figure)
        except Exception as e:
            print('Could not plot training progress due to the following error: {0}'.format(e))
        else:
            pass

    ###############################################################################
    ''' PATCH EVALUATION '''
    ###############################################################################

    if args.eval_patches:

        print('Starting patch evaluation...')

        # Load the saved state of the model
        with open(os.path.join(general_config['save_path'], 'model.json'), 'r') as mh:
            model = model_from_json(mh.read())
        model.load_weights(os.path.join(general_config['save_path'], 'model.hdf5'))

        # Create folder for segmentation results
        os.makedirs(os.path.join(general_config['save_path'], 'predictions'), exist_ok=True)

        # Get the test data
        general_config['batch_size'] = 1
        test_list = read_list(os.path.normpath(general_config['list_path'] + '_test.csv'))
        test_list = [[os.path.join(general_config['img_path'], i), os.path.join(general_config['mask_path'], m)] for
                     i, m in test_list]
        test_gen = image_streamer(test_list, image_config, mask_config, **general_config)
        test_gen = copy.deepcopy(test_gen)

        # Set up the score accumulator
        if not general_config['class_thresh'] is None:
            patch_score_accumulator = membrane_score_calculator(class_thresh=general_config['class_thresh'],
                                                    image_safety_margin=None)
        else:patch_score_accumulator = membrane_score_calculator(image_safety_margin=None)

        for num_test in range(args.num_test_samples):
            print('\r' * 12 + 'Patch {0:0>3d}/{1:0>3d}'.format(num_test, args.num_test_samples), end='\r')

            # Load data and predict the mask
            test_im, true_mask = test_gen.__getitem__(num_test)
            pred_mask = model.predict(test_im)
            if not general_config['class_thresh'] is None:
                pred_mask[pred_mask < general_config['class_thresh']] = 0
                pred_mask[pred_mask >= general_config['class_thresh']] = 1

            # Add the current example
            patch_score_accumulator.add_example(pred_mask[0, ...], true_mask[0, ...])

            # Save image
            save_image(test_im[0, ...],
                       os.path.join(general_config['save_path'], 'predictions', str(num_test) + '_image.tif'),
                       axes_order=image_config['axes_order'])

            # Save true mask
            save_image(true_mask[0, ...],
                       os.path.join(general_config['save_path'], 'predictions', str(num_test) + '_true.tif'),
                       axes_order=mask_config['axes_order'], normalize=False)

            # Save predicted mask
            save_image(pred_mask[0, ...],
                       os.path.join(general_config['save_path'], 'predictions', str(num_test) + '_pred.tif'),
                       axes_order=mask_config['axes_order'], normalize=False)

        print('\r' * 12 + 'Finished processing {0} slices.'.format(args.num_test_samples))

        # Save scores
        patch_scores = patch_score_accumulator.get_scores()
        with open(os.path.join(general_config['save_path'], 'predictions', 'patch_scores.csv'), 'w') as f:
            for key in patch_scores.keys():
                f.write("%s,%s\n" % (key, np.round(patch_scores[key], decimals=5)))

    ###############################################################################
    ''' WHOLE IMAGE EVALUATION '''
    ###############################################################################

    if args.eval_images:

        print('Starting image evaluation...')

        # Load the saved state of the model
        with open(os.path.join(general_config['save_path'], 'model.json'),'r') as mh:
            model = model_from_json(mh.read())
        model.load_weights(os.path.join(general_config['save_path'], 'model.hdf5'))

        # Disable gaussion noise
        image_config['transforms']= {'fcn':[normalization],
                                     'norm_method': 'scale',
                                     'max_val': 2**15
                                     }

        # Create folder for segmentation results
        os.makedirs(os.path.join(general_config['save_path'], 'whole_images'), exist_ok=True)

        # Get the test data
        general_config['batch_size'] = 1
        test_list = read_list(os.path.normpath(general_config['list_path']+'_test.csv'))
        test_list = [[os.path.join(general_config['img_path'], i), os.path.join(general_config['mask_path'], m)] for i,m in test_list]

        # Set up the score accumulator
        if not general_config['class_thresh'] is None:
            image_score_accumulator = membrane_score_calculator(class_thresh=general_config['class_thresh'],
                                                    image_safety_margin=None)
        else:image_score_accumulator = membrane_score_calculator(image_safety_margin=None)

        for test_files in test_list:

            print('-'*40+'\nProcessing file {0}'.format(os.path.basename(test_files[0])))

            # Get the predicted mask
            pred_mask = predict_whole_image(test_files[0], image_config, model, **general_config)

            # Save the predicted mask
            save_name = os.path.splitext(os.path.basename(test_files[0]))[0]
            save_image(pred_mask, os.path.join(general_config['save_path'], 'whole_images', save_name+'_pred.tif'), axes_order=mask_config['axes_order'], normalize=False)

            try:
                print('Calculating scores.')

                # Get the true mask
                true_params = mask_config.copy()
                true_params['shape'] = None
                true_params['start'] = None
                true_loader = image_loader(test_files[1], patch_params=true_params)

                # Add the current example
                image_score_accumulator.add_example(pred_mask, true_loader.image)

            except Exception as err:
                print('Error raised: ({0})\nProceed without score calculation...'.format(err))


        # Save scores
        image_scores = image_score_accumulator.get_scores()
        with open(os.path.join(general_config['save_path'], 'whole_images', 'image_scores.csv'), 'w') as f:
            for key in image_scores.keys():
                f.write("%s,%s\n"%(key,np.round(image_scores[key],decimals=5)))


if __name__== '__main__':
    main()