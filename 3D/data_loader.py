# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:18:53 2018

@author: eschweiler
Modification: YingChen
"""

# packet imports
import numpy as np
import warnings
import itertools
import os

from keras.utils import Sequence

# local imports
from utils import sanitycheck_patch_params
from data_handling import load_mamut_annotations, indices2mask, indices2image, apply_augmentation, save_image

###############################################################################
''' LOADER '''


###############################################################################


## Loading images
class image_loader:

    def __init__(self, image_path, patch_params=None, rnd_seed=0, mask_prob=0, **kwargs):
        self.image_path = image_path
        self.rnd_seed = rnd_seed
        self.mask_prob = mask_prob

        self.set_params(patch_params)

    def get_number_of_channels(self):
        return self.image.shape[-1]

    def get_spatial_image_size(self):
        return self.image.shape[:-1]

    def get_image_shape(self):
        return self.image.shape

    def get_data_shape(self):
        return self.patch_params['data_shape']

    def get_dtype(self):
        return self.image.dtype

    def set_params(self, patch_params):
        self.patch_params = sanitycheck_patch_params(patch_params)
        self._load_data()

    def _load_data(self):
        self.image = None
        initial_start = self.patch_params['start']
        for scale in sorted(self.patch_params['scales']) if not self.patch_params['scales'] is None else [0]:
            self.patch_params['scale_level'] = scale
            # match center points of each scaled patch
            if not self.patch_params['start'] is None and not self.patch_params['scales'] is None:
                self.patch_params['start'] = [np.int(1 / (2 ** scale) * (i - p / 2 * (2 ** scale - 1))) for i, p in
                                              zip(initial_start, self.patch_params['shape'])]
            image, self.patch_params = self.patch_params['load_fcn'](self.image_path, patch_params=self.patch_params,
                                                                     mask_prob=self.mask_prob)
            if self.image is None:
                self.image = image
                initial_start = self.patch_params['start']
                initial_data_shape = self.patch_params['data_shape']
            else:
                self.image = np.concatenate((self.image, image), axis=-1)
        self.patch_params['start'] = initial_start
        self.patch_params['data_shape'] = initial_data_shape

        # apply transformations/normalizations if desired
        if not self.patch_params['transforms'] is None:
            for trans_fcn in self.patch_params['transforms']['fcn']:
                self.image = trans_fcn(self.image, **self.patch_params['transforms'], **self.patch_params)

                # apply augmentation, in case there are augmentations specified within the parameter dict
        self.image = apply_augmentation(self.image, self.patch_params)


## Load detections
class bdv_detection_loader:

    def __init__(self, detection_path, patch_params=None, rnd_seed=0, generate=None, **kwargs):

        self.detection_path = detection_path
        self.patch_params = sanitycheck_patch_params(patch_params)

        self.rnd_seed = rnd_seed
        np.random.seed(rnd_seed)

        self._load_annotations()

        self.generate = generate
        self._generate_output()

    def get_number_of_mask_channels(self):
        if self.generate == 'mask':
            return self.patch_params['dets_per_region'] * 4
        elif self.generate == 'image':
            return 1
        elif self.generate == 'label':
            return 1
        else:
            return None

    def get_spatial_mask_size(self):
        return self.mask.shape[:-1]

    def get_mask_shape(self):
        return self.mask.shape

    def get_mask_dtype(self):
        return self.mask.dtype

    def _load_annotations(self):
        # load annotations
        annotations, start_idx, shape = load_mamut_annotations(self.detection_path, **self.patch_params)
        annotations['prob'] = [1, ] * len(annotations['pos_x'])

        # save results
        self.patch_params['shape'] = shape
        self.patch_params['start'] = start_idx
        self.indices = list(zip(annotations['pos_x'], annotations['pos_y'], annotations['pos_z']))
        self.probs = annotations['prob']

    def _generate_output(self):
        if self.generate == 'mask':
            self.mask = indices2mask(self.indices, self.probs, num_channels=self.get_number_of_mask_channels(),
                                     **self.patch_params)
        elif self.generate == 'image':
            self.mask = indices2image(self.indices, self.patch_params['shape'], self.probs, **self.patch_params)
            self.mask = self.mask[..., np.newaxis] / np.maximum(1, self.mask.max())
            self.mask = np.array(self.mask, dtype=np.float32)
        elif self.generate == 'label':
            self.mask = np.array(len(self.indices) > 0, dtype=np.uint8)
        else:
            self.mask = None


###############################################################################
''' STREAMER '''


###############################################################################

class image_streamer(Sequence):

    def __init__(self, filelist, image_config, mask_config, batch_size=1, save_path=None, \
                 num_samples=0, mask_prob=0.5, rnd_seed=0, augmentation=None, multi_output=False, **kwargs):

        self.filelist = filelist
        self.image_config = sanitycheck_patch_params(image_config.copy())  # shape, load_fcn, axes_order, grayscale
        self.mask_config = sanitycheck_patch_params(mask_config.copy())  # shape, load_fcn, axes_order, grayscale
        self.augmentation = augmentation

        self.batch_size = batch_size
        self.save_path = save_path
        self.num_samples = num_samples
        self.mask_prob = mask_prob
        self.rnd_seed = rnd_seed

        np.random.seed(rnd_seed)
        self.on_epoch_end()

        # load one image and mask for testing
        im_loader = image_loader(self.filelist[0][0], patch_params=self.image_config, \
                                 augmentation=self.augmentation, rnd_seed=self.rnd_seed)
        self.image_channels = im_loader.get_number_of_channels()

        mask_loader = image_loader(self.filelist[0][1], patch_params=self.mask_config, \
                                   augmentation=self.augmentation, rnd_seed=self.rnd_seed)
        self.mask_channels = mask_loader.get_number_of_channels()

        del im_loader
        del mask_loader

    # determine the number of batches, which can be yield from the data set
    def __len__(self):
        return np.maximum(len(self.filelist) // self.batch_size, 1)

    # shuffle indices at the end of each epoch, shuffle the list of images, but the patch position did not change
    def on_epoch_end(self):
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.filelist)

    # yield one batch of data in order
    def __getitem__(self, index):
        # restrict index number to not exceed the data set length
        index = index % self.__len__()
        self.rnd_seed += index
        batch_indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size, 1)
        # restrict batch indices to not exceed the data set length
        batch_indices = [bi % len(self.filelist) for bi in batch_indices]
        im_batch, mask_batch = self.__data_generation(batch_indices)

        return im_batch, mask_batch

    # generate one batch of data randomly
    def __data_generation(self, batch_indices):

        batch_images = None
        batch_masks = None

        for patch_count, batch_indice in enumerate(batch_indices):

            # get loader for image and mask

            # reset start index
            self.mask_config['start'] = None

            # specify augmentations
            if not self.augmentation is None:
                if 'rotations' in self.augmentation.keys(): #0,1,2,3
                    self.mask_config['rotation_count'] = np.random.choice(self.augmentation['rotations'])
                if 'mirroring' in self.augmentation.keys(): #True /False
                    self.mask_config['mirror_x'] = np.random.rand() > self.augmentation['mirroring']
                    self.mask_config['mirror_y'] = np.random.rand() > self.augmentation['mirroring']

            mask_loader = image_loader(self.filelist[batch_indice][1], patch_params=self.mask_config, \
                                       rnd_seed=self.rnd_seed, mask_prob=self.mask_prob)

            # set the position to match the position of the extracted mask
            self.image_config['start'] = self.mask_config['start']
            self.image_config['rotation_count'] = self.mask_config['rotation_count']
            self.image_config['mirror_x'] = self.mask_config['mirror_x']
            self.image_config['mirror_y'] = self.mask_config['mirror_y']
            im_loader = image_loader(self.filelist[batch_indice][0], patch_params=self.image_config, \
                                     augmentation=None, rnd_seed=self.rnd_seed)

            # create the batch matrix if not already created (create here to handle arbitrary shapes)
            if batch_images is None:
                batch_images = np.zeros((self.batch_size,) + im_loader.get_image_shape(), \
                                        dtype=im_loader.get_dtype())
            if batch_masks is None:
                batch_masks = np.zeros((self.batch_size,) + mask_loader.get_image_shape(), \
                                       dtype=mask_loader.get_dtype())

            # stack current patch to the batch matrix
            batch_images[patch_count, ...] = im_loader.image
            batch_masks[patch_count, ...] = mask_loader.image

            # save sample patches for visual debugging and control
            if self.num_samples > 0 and not self.save_path is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sample_path = os.path.join(self.save_path, 'samples')
                    os.makedirs(sample_path, exist_ok=True)

                    # save image
                    save_image(im_loader.image, os.path.join(sample_path, str(self.num_samples) + '_image_' + \
                                                             os.path.splitext(
                                                                 os.path.basename(self.filelist[batch_indice][0]))[
                                                                 0] + '.tif'), \
                               axes_order=self.image_config['axes_order'], normalize=False)

                    # save mask
                    save_image(mask_loader.image, os.path.join(sample_path, str(self.num_samples) + '_mask_' + \
                                                               os.path.splitext(
                                                                   os.path.basename(self.filelist[batch_indice][1]))[
                                                                   0] + '.tif'), \
                               axes_order=self.mask_config['axes_order'], normalize=False)

                    self.num_samples -= 1

        return batch_images, batch_masks


class image_tiler(Sequence):

    def __init__(self, filepath, config, overlap=0, application_scales=[0], **kwargs):

        self.filepath = filepath
        self.config = config.copy()

        self.application_scales = [np.int(scale) for scale in application_scales]
        if any([a < 0 for a in self.application_scales]):
            self.application_scales = [np.maximum(0, a) for a in self.application_scales]
            print('Only scales > 0 are allowed. Replacing illegal scales by 0...')
        self.config['scale_level'] = np.min(self.application_scales)
        self.overlap = overlap

        # initialize the data loader
        self.loader = image_loader(self.filepath, patch_params=self.config)
        self.data_shape = self.loader.get_data_shape()

        assert len(self.config['shape']) == len(
            self.loader.get_spatial_image_size()), 'Dimension of requested shape and image shape do not match'

        # construct a list of possible patch locations
        self.locations = self._get_patch_locations()

    # determine the number of patches, which can be extracted from the image
    def _get_patch_locations(self):
        locations = []
        for i, p in zip(self.data_shape, self.config['shape']):
            # get left coords
            coords = np.arange(np.ceil(i / (p - self.overlap)), dtype=np.uint16) * (p - self.overlap)
            # ensure to not get out of bounds at the image boundaries
            coords = np.minimum(coords, np.maximum(0, i - p))
            locations.append(coords)
        locations = sorted(list(set(itertools.product(*locations))))
        return locations

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        index = index % self.__len__()

        scale_pyramid_tiles = np.zeros(
            (len(self.application_scales),) + self.config['shape'] + (self.loader.get_number_of_channels(),),
            dtype=self.loader.get_dtype())

        for num_scale, application_scale in enumerate(self.application_scales):
            # set current configuration
            self.config['start'] = [i // (2 ** application_scale) for i in self.locations[index]]
            self.config['scale_level'] = application_scale
            self.loader.set_params(self.config)
            # append to the current pyramid
            scale_pyramid_tiles[num_scale, ...] = self.loader.image

        return scale_pyramid_tiles


class detection_streamer(Sequence):

    def __init__(self, filelist, config, batch_size=1, save_path=None, num_samples=0, rnd_seed=0, \
                 get_indices=False, generate='mask', **kwargs):

        self.filelist = filelist
        self.config = config.copy()

        self.batch_size = batch_size
        self.save_path = save_path
        self.num_samples = num_samples
        self.rnd_seed = rnd_seed
        self.get_indices = get_indices
        self.generate = generate

        np.random.seed(rnd_seed)
        self.on_epoch_end()

        # initial test
        test_detection_loader = bdv_detection_loader(self.filelist[0][1], patch_params=self.config, \
                                                     rnd_seed=self.rnd_seed, generate=self.generate)
        test_image_loader = image_loader(self.filelist[0][0], patch_params=self.config, \
                                         rnd_seed=self.rnd_seed)

        self.mask_channels = test_detection_loader.get_number_of_mask_channels()
        self.image_channels = test_image_loader.get_number_of_channels()

    # determine the number of files, which can be yield from the data set
    def __len__(self):
        return np.maximum(len(self.filelist), 1)

    # shuffle indices at the end of each epoch
    def on_epoch_end(self):
        np.random.seed(self.rnd_seed)
        np.random.shuffle(self.filelist)

    # yield one batch of data
    def __getitem__(self, index):
        # restrict index number to not exceed the data set length
        index = index % self.__len__()
        self.rnd_seed += index
        batch_indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size, 1)
        # restrict batch indices to not exceed the data set length
        batch_indices = [bi % len(self.filelist) for bi in batch_indices]
        im_batch, mask_batch, indices = self.__data_generation(batch_indices)

        if self.get_indices:
            return im_batch, mask_batch, indices
        else:
            return im_batch, mask_batch

    # generate one batch of data
    def __data_generation(self, batch_indices):

        batch_images = None
        batch_masks = None
        indices = []

        for patch_count, batch_index in enumerate(batch_indices):

            # get loader for image and mask
            self.config['start'] = None  # reset start index
            det_loader = bdv_detection_loader(self.filelist[batch_index][1], patch_params=self.config, \
                                              rnd_seed=self.rnd_seed, generate=self.generate)
            self.config['start'] = det_loader.patch_params['start']
            im_loader = image_loader(self.filelist[batch_index][0], patch_params=self.config, rnd_seed=self.rnd_seed)

            # create the batch matrix if not already created (create here to handle arbitrary shapes)
            if batch_images is None:
                batch_images = np.zeros((self.batch_size,) + im_loader.get_image_shape(), \
                                        dtype=im_loader.get_dtype())
            if batch_masks is None:
                batch_masks = np.zeros((self.batch_size,) + det_loader.get_mask_shape(), \
                                       dtype=det_loader.get_mask_dtype())

            # stack current patch to the batch matrix
            batch_images[patch_count, ...] = im_loader.image
            batch_masks[patch_count, ...] = det_loader.mask
            indices.extend(det_loader.indices)

            # save sample patches for visual debugging and control
            if self.num_samples > 0 and not self.save_path is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sample_path = os.path.join(self.save_path, 'samples')
                    os.makedirs(sample_path, exist_ok=True)
                    # save image
                    save_image(im_loader.image, os.path.join(sample_path, str(self.num_samples) + '_image_' + \
                                                             os.path.basename(self.filelist[batch_index][0]) + '.tif'), \
                               axes_order=self.config['axes_order'], normalize=True)

                    # save mask
                    np.save(os.path.join(sample_path, str(self.num_samples) + '_mask_' + \
                                         os.path.basename(self.filelist[batch_index][1])), \
                            det_loader.mask)

                    # save mask image
                    mask_save = indices2image(det_loader.indices, im_loader.get_image_shape())
                    save_image(mask_save, os.path.join(sample_path, str(self.num_samples) + '_mask_' + \
                                                       os.path.basename(self.filelist[batch_index][1]) + '.tif'), \
                               axes_order=self.config['axes_order'], normalize=True)

                    # save indices
                    with open(os.path.join(sample_path, str(self.num_samples) + '_indices_' + os.path.basename(
                            self.filelist[batch_index][0]) + '.txt'), 'w') as fh:
                        for i in det_loader.indices: fh.write("{}\n".format(i))

                    self.num_samples -= 1

        return batch_images, batch_masks, indices


class h5_tiler(Sequence):

    def __init__(self, image_path, config, overlap=0, **kwargs):
        self.image_path = image_path
        self.overlap = overlap
        self.config = config.copy()
        self.config['start'] = None

        self.image_loader = image_loader(self.image_path, patch_params=self.config)
        self.data_shape = self.image_loader.get_data_shape()
        # construct a list of possible patch locations
        self.locations = self.get_patch_locations()

    # determine the location of patches, which can be extracted from the image
    def get_patch_locations(self):
        locations = []
        for i, p in zip(self.data_shape, self.config['shape']):
            # get left coords
            coords = np.arange(np.ceil(i / (p - self.overlap)), dtype=np.uint16) * (p - self.overlap)
            # ensure to not get out of bounds at the rigth image boundary
            coords = np.minimum(coords, np.maximum(0, i - p))
            locations.append(coords)
        locations = list(itertools.product(*locations))
        return locations

    # return the total number of patches, which can be extracted from the image
    def __len__(self):
        return len(self.locations)

        # load one patch

    def __getitem__(self, index):
        index = index % self.__len__()
        self.config['start'] = self.locations[index]
        self.image_loader.set_params(self.config)
        return self.image_loader.image


