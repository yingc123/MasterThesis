# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:23:13 2018

@author: eschweiler
Modification: YingChen
"""

import numpy as np

from scipy.ndimage.morphology import distance_transform_edt

## add keras packages
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Conv2DTranspose, Conv3DTranspose, \
    BatchNormalization, Activation, AveragePooling3D, Add, Dropout, ReLU
from keras.layers.merge import concatenate
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras_radam import RAdam
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
import tensorflow as tf
from sklearn import metrics
from skimage.io import imread, imsave
import os
import warnings
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy

# Define Swish activation function
class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    return (K.sigmoid(x) * x)


get_custom_objects().update({'swish': Swish(swish)})

########################################################################################################################
''' SEGMENTATION '''


########################################################################################################################


## mean intersection over union
def mean_iou_wrapper(num_classes=3):
    def mean_iou(y_true, y_pred):
        prec = []
        y_true = tf.to_int32(y_true > 0)
        for t in np.arange(0.5, 1, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, np.maximum(num_classes, 2))
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    return mean_iou

## mean intersection over union
def mean_iou(y_true, y_pred):
    prec = []
    y_true = tf.to_int32(y_true > 0)
    for t in np.arange(0.5, 1, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

## calculate F1 score
def f1_score(y_true, y_pred):
    """
    f1 score
    :param y_true:
    :param y_pred:
    :return:
    """
    prec = []
    y_true = tf.to_int32(y_true > 0)
    y_pred_ = tf.to_int32(y_pred > 0.5)

    precision, up_opt1 = tf.metrics.precision(y_true, y_pred_)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt1]):
        precision = tf.identity(up_opt1)

    recall, up_opt2 = tf.metrics.recall(y_true, y_pred_)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt2]):
        recall = tf.identity(recall)

    prec.append(2 * ((precision * recall) / (precision + recall)))
    return 2 * ((precision * recall) / (precision + recall))



## custom loss function allowing class weighting
def weighted_binary_class_crossentropy_wrapper(class_weights=(0.3)):
    def weighted_binary_class_crossentropy(y_true, y_pred):
        results = None
        for num_class, class_weight in enumerate(class_weights):
            # get ground truth and prediction of current class
            gt = tf.to_float(y_true[..., num_class])
            pred = tf.to_float(y_pred[..., num_class])
            # determine intra-class-weighting
            foregroundPixels = K.sum(tf.to_float(gt > 0))
            backgroundPixels = K.sum(1 - tf.to_float(gt > 0))
            totalNumberOfPixels = foregroundPixels + backgroundPixels
            foregroundWeights = K.clip(totalNumberOfPixels / foregroundPixels, 1e-7, 1e+2)  # prevent zero division
            backgroundWeights = K.clip(totalNumberOfPixels / backgroundPixels, 1e-7, 1e+2)
            # calculate weighted crossentropy and apply inter-class-weighting
            if results == None:
                results = class_weight * np.multiply(foregroundWeights * gt + backgroundWeights * (1 - gt),
                                                     K.binary_crossentropy(tf.to_float(gt > 0), pred))
            else:
                results += class_weight * np.multiply(foregroundWeights * gt + backgroundWeights * (1 - gt),
                                                      K.binary_crossentropy(tf.to_float(gt > 0), pred))
        return K.mean(results)

    return weighted_binary_class_crossentropy

## custom loss function allowing weightings depending on class occurrences
def weighted_binary_crossentropy1(y_true, y_pred):
    gt = tf.to_float(y_true)
    pred = tf.to_float(y_pred)
    foregroundPixels = K.sum(tf.to_float(gt > 0))
    backgroundPixels = K.sum(1 - tf.to_float(gt > 0))
    totalNumberOfPixels = foregroundPixels + backgroundPixels
    foregroundWeights = K.clip(totalNumberOfPixels / foregroundPixels, 1e-7, 1e+2)  # prevent zero division
    backgroundWeights = K.clip(totalNumberOfPixels / backgroundPixels, 1e-7, 1e+2)
    results = np.multiply(foregroundWeights * gt + backgroundWeights * (1 - gt),
                                         K.binary_crossentropy(tf.to_float(gt > 0), pred))
    return K.mean(results)

## custom loss function allowing weightings depending on class occurrences
def weighted_binary_crossentropy(y_true, y_pred):
    foregroundPixels = K.sum(tf.to_float(y_true > 0))
    backgroundPixels = K.sum(1 - tf.to_float(y_true > 0))
    totalNumberOfPixels = foregroundPixels + backgroundPixels
    foregroundWeights = totalNumberOfPixels / (foregroundPixels + 1)  # prevent zero division
    backgroundWeights = totalNumberOfPixels / (backgroundPixels + 1)
    return K.mean(np.multiply(foregroundWeights * y_true + backgroundWeights * (1 - y_true),
                              K.binary_crossentropy(tf.to_float(y_true > 0), y_pred)), axis=-1)


## distance dependend loss function based on cross entropy
def weighted_distance_crossentropy_wrapper(class_weights=(0.3, 0.3, 0.3)):
    def weighted_distance_crossentropy(y_true, y_pred):
        results = None
        for num_class, class_weight in enumerate(class_weights):
            # get ground truth and prediction of current class
            gt = tf.to_float(y_true[..., num_class])
            pred = tf.to_float(y_pred[..., num_class])
            # determine distance weights
            try:
                weight_map = distance_transform_edt(1 - gt)
                weight_map[gt == 1] = weight_map.max()
                '''
                if num_class==1:
                    # determine intra-class-weighting
                    foregroundPixels = K.sum(tf.to_float(gt>0))
                    backgroundPixels = K.sum(1-tf.to_float(gt>0))
                    totalNumberOfPixels = foregroundPixels + backgroundPixels
                    foregroundWeights = K.clip(totalNumberOfPixels / (foregroundPixels), 1e-15, 1e+2) # prevent zero division
                    backgroundWeights = K.clip(totalNumberOfPixels / (backgroundPixels), 1e-15, 1e+2)                
                    weight_map = np.multiply(weight_map, foregroundWeights*gt + backgroundWeights*(1-gt))                
                '''
            except:
                weight_map = 1
                print('Not using class weights!')
            # calculate weighted crossentropy and apply inter-class-weighting
            if results == None:
                results = class_weight * np.multiply(weight_map, K.binary_crossentropy(tf.to_float(gt > 0), pred))
            else:
                results += class_weight * np.multiply(weight_map, K.binary_crossentropy(tf.to_float(gt > 0), pred))
        return K.mean(results)

    return weighted_distance_crossentropy


# focal loss for improved handling of sparse binary labels
def focal_loss_wrapper(gamma=2., alpha=.75):
    def focal_loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss


## Ronneberger15
def unet2D(input_shape=(512, 512), input_channels=1, output_channels=1, verbose=True, **kwargs):
    inputs = Input((input_shape[0], input_shape[1], input_channels))

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=[weighted_binary_class_crossentropy_wrapper(class_weights=(0.3, 0.3, 0.3))],
                  metrics=[mean_iou_wrapper(num_classes=output_channels)])
    if verbose: model.summary()

    return model


## Cicek16
# modify the network architecture, by setting the order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
def unet3D(input_shape=(50, 50, 50), lr= 1e-4, metrics=f1_score,input_channels=1, output_channels=1, verbose=True, activation_fcn='relu',
         apply_batch_norm=False, **kwargs):
    inputs = Input((*input_shape, input_channels))
    layer = ReLU()
    c1 = Conv3D(8, (3, 3, 3),  padding='same')(inputs)
    if apply_batch_norm: c1 = BatchNormalization()(c1)
    c1 = layer(c1)
    c1 = Conv3D(16, (3, 3, 3),  padding='same')(c1)
    if apply_batch_norm: c1 = BatchNormalization()(c1)
    c1 = layer(c1)
    #d1 = Dropout(0.2)(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(16, (3, 3, 3),  padding='same')(p1)
    if apply_batch_norm: c2 = BatchNormalization()(c2)
    c2 = layer(c2)
    c2 = Conv3D(32, (3, 3, 3),  padding='same')(c2)
    if apply_batch_norm: c2 = BatchNormalization()(c2)
    c2 = layer(c2)
    #d2 = Dropout(0.2)(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(32, (3, 3, 3),  padding='same')(p2)
    if apply_batch_norm: c3 = BatchNormalization()(c3)
    c3 = layer(c3)
    c3 = Conv3D(64, (3, 3, 3),  padding='same')(c3)
    if apply_batch_norm: c3 = BatchNormalization()(c3)
    c3 = layer(c3)
    #apply dropout
    #d3 = Dropout(0.35)(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(64, (3, 3, 3),  padding='same')(p3)
    if apply_batch_norm: c4 = BatchNormalization()(c4)
    c4 = layer(c4)
    c4 = Conv3D(128, (3, 3, 3),  padding='same')(c4)
    if apply_batch_norm: c4 = BatchNormalization()(c4)
    c4 = layer(c4)
    #apply dropout
    #d5 = Dropout(0.5)(c4)
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u6 = concatenate([u6, c3])
    c6 = Conv3D(64, (3, 3, 3),  padding='same')(u6)
    if apply_batch_norm: c6 = BatchNormalization()(c6)
    c6 = layer(c6)
    c6 = Conv3D(64, (3, 3, 3),  padding='same')(c6)
    if apply_batch_norm: c6 = BatchNormalization()(c6)
    c6 = layer(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c2])
    c7 = Conv3D(32, (3, 3, 3),  padding='same')(u7)
    if apply_batch_norm: c7 = BatchNormalization()(c7)
    c7 = layer(c7)
    c7 = Conv3D(32, (3, 3, 3),  padding='same')(c7)
    if apply_batch_norm: c7 = BatchNormalization()(c7)
    c7 = layer(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c1])
    c8 = Conv3D(16, (3, 3, 3),  padding='same')(u8)
    if apply_batch_norm: c8 = BatchNormalization()(c8)
    c8 = layer(c8)
    c8 = Conv3D(16, (3, 3, 3),  padding='same')(c8)
    if apply_batch_norm: c8 = BatchNormalization()(c8)
    c8 = layer(c8)

    outputs = Conv3D(output_channels, (1, 1, 1), activation='sigmoid')(c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mean_absolute_error'])
    model.compile(optimizer=RAdam(learning_rate=lr), loss=[weighted_binary_crossentropy1],
                  metrics=[metrics])
    if verbose: model.summary()
    return model

def unet3D1(input_shape=(50, 50, 50), lr= 1e-4, metrics=f1_score,input_channels=1, output_channels=1, verbose=True, activation_fcn='relu',
           apply_batch_norm=False, **kwargs):
    inputs = Input((*input_shape, input_channels))
    layer = ReLU()
    c1 = Conv3D(8, (3, 3, 3),  padding='same')(inputs)
    if apply_batch_norm: c1 = BatchNormalization()(c1)
    c1 = layer(c1)
    c1 = Conv3D(16, (3, 3, 3),  padding='same')(c1)
    if apply_batch_norm: c1 = BatchNormalization()(c1)
    c1 = layer(c1)
    #d1 = Dropout(0.2)(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(16, (3, 3, 3),  padding='same')(p1)
    if apply_batch_norm: c2 = BatchNormalization()(c2)
    c2 = layer(c2)
    c2 = Conv3D(32, (3, 3, 3),  padding='same')(c2)
    if apply_batch_norm: c2 = BatchNormalization()(c2)
    c2 = layer(c2)
    #d2 = Dropout(0.2)(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(32, (3, 3, 3),  padding='same')(p2)
    if apply_batch_norm: c3 = BatchNormalization()(c3)
    c3 = layer(c3)
    c3 = Conv3D(64, (3, 3, 3),  padding='same')(c3)
    if apply_batch_norm: c3 = BatchNormalization()(c3)
    c3 = layer(c3)
    #apply dropout
    #d3 = Dropout(0.5)(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(64, (3, 3, 3),  padding='same')(p3)
    if apply_batch_norm: c4 = BatchNormalization()(c4)
    c4 = layer(c4)
    c4 = Conv3D(128, (3, 3, 3),  padding='same')(c4)
    if apply_batch_norm: c4 = BatchNormalization()(c4)
    c4 = layer(c4)
    #apply dropout
    #d5 = Dropout(0.5)(c4)
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u6 = concatenate([u6, c3])
    c6 = Conv3D(64, (3, 3, 3),  padding='same')(u6)
    if apply_batch_norm: c6 = BatchNormalization()(c6)
    c6 = layer(c6)
    c6 = Conv3D(64, (3, 3, 3),  padding='same')(c6)
    if apply_batch_norm: c6 = BatchNormalization()(c6)
    c6 = layer(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c2])
    c7 = Conv3D(32, (3, 3, 3),  padding='same')(u7)
    if apply_batch_norm: c7 = BatchNormalization()(c7)
    c7 = layer(c7)
    c7 = Conv3D(32, (3, 3, 3),  padding='same')(c7)
    if apply_batch_norm: c7 = BatchNormalization()(c7)
    c7 = layer(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c1])
    c8 = Conv3D(16, (3, 3, 3),  padding='same')(u8)
    if apply_batch_norm: c8 = BatchNormalization()(c8)
    c8 = layer(c8)
    c8 = Conv3D(16, (3, 3, 3),  padding='same')(c8)
    if apply_batch_norm: c8 = BatchNormalization()(c8)
    c8 = layer(c8)

    outputs = Conv3D(output_channels, (1, 1, 1), activation='sigmoid')(c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=['mean_squared_error'], metrics=['mean_absolute_error'])
    model.compile(optimizer=RAdam(learning_rate=lr), loss=[weighted_binary_crossentropy1],
                  metrics=[metrics])
    if verbose: model.summary()
    return model

########################################################################################################################
''' DETECTION '''


########################################################################################################################

def masked_mse3D_wrapper(output_channels=12):
    def masked_mse3D(y_true, y_pred):
        # classification
        y_true_classify = tf.to_float(y_true[..., :output_channels // 4])
        weighting_positional = tf.transpose(y_true_classify, list(range(4, -1, -1)))
        weighting_positional = tf.gather(weighting_positional, [i // 3 for i in range(output_channels * 3 // 4)])
        weighting_positional = tf.transpose(weighting_positional, list(range(4, -1, -1)))
        # positioning
        y_true_position = tf.to_float(y_true[..., output_channels // 4:])
        y_pred_position = tf.to_float(y_pred[..., output_channels // 4:])
        # mse
        mse_pos = weighting_positional * K.sqrt(y_true_position - y_pred_position)
        return K.sum(mse_pos) / K.clip(K.sum(y_true_classify), K.epsilon(), np.inf)

    return masked_mse3D


def distance_loss3D_wrapper(output_channels=12, weight_class=0.6, weight_pos=0.4, max_weight_scale=100):
    def distance_loss3D(y_true, y_pred):
        # classification fg-bg weighting
        y_true_classify = tf.to_float(y_true[..., :output_channels // 4])
        y_pred_classify = tf.to_float(y_pred[..., :output_channels // 4])
        foregroundCount = K.sum(tf.to_float(y_true_classify > 0))
        backgroundCount = K.sum(1 - tf.to_float(y_true_classify > 0))
        foregroundWeight = K.clip((backgroundCount + foregroundCount) / (foregroundCount), 1e-15, max_weight_scale)
        backgroundWeight = K.clip((backgroundCount + foregroundCount) / (backgroundCount), 1e-15, max_weight_scale)
        weighting_classify = foregroundWeight * y_true_classify + backgroundWeight * (1 - y_true_classify)
        # classification MSLE
        first_log = K.log(K.clip(y_pred_classify, K.epsilon(), None) + 1.)
        second_log = K.log(K.clip(y_true_classify, K.epsilon(), None) + 1.)
        loss_msle = K.square(first_log - second_log)
        loss_classify = K.sum(weighting_classify * loss_msle, axis=-1, keepdims=True)
        # positional L1 loss
        weighting_positional = tf.transpose(y_true_classify, list(range(4, -1, -1)))
        weighting_positional = tf.gather(weighting_positional, [i // 3 for i in range(output_channels * 3 // 4)])
        weighting_positional = tf.transpose(weighting_positional, list(range(4, -1, -1)))
        y_true_position = tf.to_float(y_true[..., output_channels // 4:])
        y_pred_position = tf.to_float(y_pred[..., output_channels // 4:])
        loss_positional = K.sum(weighting_positional * K.abs(y_true_position - y_pred_position), axis=-1,
                                keepdims=True)  # square the distance???
        # return weighted sum of both losses
        return K.mean(weight_class * loss_classify + weight_pos * loss_positional)

    return distance_loss3D


def detection_rcnn_3D(input_shape=(112, 112, 112), input_channels=1, output_channels=12, weight_class=0.5,
                      weight_pos=0.5, max_weight_scale=100, use_batchnorm=True, verbose=True, **kwargs):
    def VoxRes_module(input_layer, num_channels=64):
        vrn = BatchNormalization()(input_layer)
        if use_batchnorm: vrn = Activation('relu')(vrn)
        vrn = Conv3D(num_channels, (3, 3, 3), strides=(1, 1, 1), padding='same')(vrn)
        if use_batchnorm: vrn = BatchNormalization()(vrn)
        vrn = Activation('relu')(vrn)
        vrn = Conv3D(num_channels, (3, 3, 3), strides=(1, 1, 1), padding='same')(vrn)
        vrn = Add()([input_layer, vrn])
        return vrn

    inputs = Input((input_shape) + (input_channels,))

    l = Conv3D(32, (3, 3, 3), padding='same')(inputs)
    if use_batchnorm: l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv3D(32, (3, 3, 3), padding='same')(l)
    if use_batchnorm: l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    if use_batchnorm: l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    if use_batchnorm: l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    if use_batchnorm: l = BatchNormalization()(l)
    l = Activation('relu')(l)

    # inputs_down = AveragePooling3D((8,8,8))(inputs)
    # l = concatenate([l, inputs_down])

    ## split network for classification and positioning prediction

    # classification
    l_cla = Conv3D(64, (3, 3, 3), padding='same')(l)
    if use_batchnorm: l_cla = BatchNormalization()(l_cla)
    l_cla = Activation('relu')(l_cla)
    l_cla = Conv3D(32, (3, 3, 3), padding='same')(l_cla)
    if use_batchnorm: l_cla = BatchNormalization()(l_cla)
    l_cla = Activation('relu')(l_cla)
    l_cla = Conv3D(output_channels // 4, (1, 1, 1), padding='same')(l_cla)
    out_cla = Activation('relu')(l_cla)

    # positioning
    l_pos = Conv3D(64, (3, 3, 3), padding='same')(l)
    if use_batchnorm: l_pos = BatchNormalization()(l_pos)
    l_pos = Activation('relu')(l_pos)
    l_pos = Conv3D(32, (3, 3, 3), padding='same')(l_pos)
    if use_batchnorm: l_pos = BatchNormalization()(l_pos)
    l_pos = Activation('relu')(l_pos)
    l_pos = Conv3D(output_channels * 3 // 4, (1, 1, 1), padding='same')(l_pos)
    out_pos = Activation('relu')(l_pos)

    ## refinements
    ref_input = concatenate([out_cla, out_pos])

    # refine classification
    l_ref_cla = Conv3D(32, (3, 3, 3), padding='same')(ref_input)
    if use_batchnorm: l_ref_cla = BatchNormalization()(l_ref_cla)
    l_ref_cla = Activation('relu')(l_ref_cla)
    l_ref_cla = Conv3D(16, (3, 3, 3), padding='same')(l_ref_cla)
    if use_batchnorm: l_ref_cla = BatchNormalization()(l_ref_cla)
    l_ref_cla = Activation('relu')(l_ref_cla)
    l_ref_cla = Conv3D(output_channels // 4, (1, 1, 1), padding='same')(l_ref_cla)
    out_cla_ref = Activation('sigmoid')(l_ref_cla)

    # refine positioning
    l_ref_pos = Conv3D(32, (3, 3, 3), padding='same')(ref_input)
    if use_batchnorm: l_ref_pos = BatchNormalization()(l_ref_pos)
    l_ref_pos = Activation('relu')(l_ref_pos)
    l_ref_pos = Conv3D(16, (3, 3, 3), padding='same')(l_ref_pos)
    if use_batchnorm: l_ref_pos = BatchNormalization()(l_ref_pos)
    l_ref_pos = Activation('relu')(l_ref_pos)
    l_ref_pos = Conv3D(output_channels * 3 // 4, (1, 1, 1), padding='same')(l_ref_pos)
    out_pos_ref = Activation('sigmoid')(l_ref_pos)

    outputs = concatenate([out_cla_ref, out_pos_ref])

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=distance_loss3D_wrapper(output_channels=output_channels, \
                                                                 weight_class=weight_class, \
                                                                 weight_pos=weight_pos, \
                                                                 max_weight_scale=max_weight_scale), \
                  metrics=[masked_mse3D_wrapper(output_channels=output_channels)])

    if verbose: model.summary()

    return model


########################################################################################################################
''' HARMONICS '''


########################################################################################################################


def harmonic_loss3D_wrapper(dets_per_region=3, num_coefficients=81, weight_class=0.3, weight_pos=0.3, weight_shape=0.3,
                            max_weight_scale=100):
    def harmonic_loss3D(y_true, y_pred):
        # classification loss (MSLE)
        y_true_classify = tf.to_float(y_true[..., :dets_per_region])
        y_pred_classify = tf.to_float(y_pred[..., :dets_per_region])
        foregroundCount = K.sum(tf.to_float(y_true_classify > 0))
        backgroundCount = K.sum(1 - tf.to_float(y_true_classify > 0))
        foregroundWeight = K.clip((backgroundCount + foregroundCount) / (foregroundCount), 1e-15, max_weight_scale)
        backgroundWeight = K.clip((backgroundCount + foregroundCount) / (backgroundCount), 1e-15, max_weight_scale)
        weighting_classify = foregroundWeight * y_true_classify + backgroundWeight * (1 - y_true_classify)
        first_log = K.log(K.clip(y_pred_classify, K.epsilon(), None) + 1.)
        second_log = K.log(K.clip(y_true_classify, K.epsilon(), None) + 1.)
        loss_msle = K.square(first_log - second_log)
        loss_classify = K.sum(weighting_classify * loss_msle, axis=-1, keepdims=True)
        # positional loss (L1)
        weighting_positional = tf.transpose(y_true_classify, list(range(4, -1, -1)))
        weighting_positional = tf.gather(weighting_positional, [i // 3 for i in range(
            dets_per_region * 3)])  # for each region check if there actually is a cell (same weight for x_n, y_n, z_n)
        weighting_positional = tf.transpose(weighting_positional, list(range(4, -1, -1)))
        y_true_position = tf.to_float(y_true[..., dets_per_region:dets_per_region * 4])
        y_pred_position = tf.to_float(y_pred[..., dets_per_region:dets_per_region * 4])
        loss_positional = K.sum(weighting_positional * K.abs(y_true_position - y_pred_position), axis=-1, keepdims=True)
        # shape loss (L1)
        weighting_shape = tf.transpose(y_true_classify, list(range(4, -1, -1)))
        weighting_shape = tf.gather(weighting_shape, [i // num_coefficients for i in range(
            dets_per_region * num_coefficients)])  # for each region check if there actually is a cell (same weight for each coefficient of detection n)
        weighting_shape = tf.transpose(weighting_shape, list(range(4, -1, -1)))
        y_true_shape = tf.to_float(y_true[..., dets_per_region * 4:])
        y_pred_shape = tf.to_float(y_pred[..., dets_per_region * 4:])
        loss_shape = K.sum(weighting_shape * K.abs(y_true_shape - y_pred_shape), axis=-1, keepdims=True)
        # return weighted sum of all losses
        return K.mean(weight_class * loss_classify + weight_pos * loss_positional + weight_shape * loss_shape)

    return harmonic_loss3D


def harmonic_rcnn_3D(input_shape=(112, 112, 112), activation_fcn='relu', input_channels=1, dets_per_region=3,
                     num_coefficients=81, weight_class=0.3, weight_pos=0.3, weight_shape=0.3, max_weight_scale=100,
                     verbose=True, **kwargs):
    def VoxRes_module(input_layer, num_channels=64):
        vrn = BatchNormalization()(input_layer)
        vrn = Activation(activation_fcn)(vrn)
        vrn = Conv3D(num_channels, (3, 3, 3), strides=(1, 1, 1), padding='same')(vrn)
        vrn = BatchNormalization()(vrn)
        vrn = Activation(activation_fcn)(vrn)
        vrn = Conv3D(num_channels, (3, 3, 3), strides=(1, 1, 1), padding='same')(vrn)
        vrn = Add()([input_layer, vrn])
        return vrn

    inputs = Input((input_shape) + (input_channels,))

    l = Conv3D(32, (3, 3, 3), padding='same')(inputs)
    l = BatchNormalization()(l)
    l = Activation(activation_fcn)(l)

    l = Conv3D(32, (3, 3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation(activation_fcn)(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    l = BatchNormalization()(l)
    l = Activation(activation_fcn)(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    l = BatchNormalization()(l)
    l = Activation(activation_fcn)(l)

    l = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same')(l)
    l = VoxRes_module(l, num_channels=64)
    l = VoxRes_module(l, num_channels=64)
    l = BatchNormalization()(l)
    l = Activation(activation_fcn)(l)

    inputs_down = AveragePooling3D((8, 8, 8))(inputs)
    l = concatenate([l, inputs_down])

    # split network for classification, positioning and shape prediction
    l_cla = Conv3D(64, (3, 3, 3), padding='same')(l)
    l_cla = VoxRes_module(l_cla, num_channels=64)
    l_cla = VoxRes_module(l_cla, num_channels=64)
    l_cla = BatchNormalization()(l_cla)
    l_cla = Activation(activation_fcn)(l_cla)
    l_cla = Conv3D(32, (3, 3, 3), padding='same')(l_cla)
    l_cla = BatchNormalization()(l_cla)
    l_cla = Activation(activation_fcn)(l_cla)
    l_cla = Conv3D(dets_per_region, (1, 1, 1), padding='same')(l_cla)
    out_cla = Activation('sigmoid')(l_cla)

    l_pos = Conv3D(64, (3, 3, 3), padding='same')(l)
    l_pos = VoxRes_module(l_pos, num_channels=64)
    l_pos = VoxRes_module(l_pos, num_channels=64)
    l_pos = BatchNormalization()(l_pos)
    l_pos = Activation(activation_fcn)(l_pos)
    l_pos = Conv3D(32, (3, 3, 3), padding='same')(l_pos)
    l_pos = BatchNormalization()(l_pos)
    l_pos = Activation(activation_fcn)(l_pos)
    l_pos = Conv3D(dets_per_region * 3, (1, 1, 1), padding='same')(l_pos)
    out_pos = Activation('sigmoid')(l_pos)

    l_shape = Conv3D(128, (3, 3, 3), padding='same')(l)
    l_shape = VoxRes_module(l_shape, num_channels=128)
    l_shape = VoxRes_module(l_shape, num_channels=128)
    l_shape = BatchNormalization()(l_shape)
    l_shape = Activation(activation_fcn)(l_shape)
    l_shape = Conv3D(256, (3, 3, 3), padding='same')(l_shape)
    l_shape = BatchNormalization()(l_shape)
    l_shape = Activation(activation_fcn)(l_shape)
    out_shape = Conv3D(dets_per_region * num_coefficients, (1, 1, 1), padding='same')(l_shape)

    outputs = concatenate([out_cla, out_pos, out_shape])

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=RAdam(lr=0.001), loss=harmonic_loss3D_wrapper(dets_per_region=dets_per_region, \
                                                                          num_coefficients=num_coefficients, \
                                                                          weight_class=weight_class, \
                                                                          weight_pos=weight_pos, \
                                                                          weight_shape=weight_shape, \
                                                                          max_weight_scale=max_weight_scale))

    # model = Model(inputs=[inputs], outputs=[out_cla, out_pos, out_shape])
    # model.compile(optimizer=Adam(lr=0.0003), loss=['binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'])

    if verbose: model.summary()

    return model




