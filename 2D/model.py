import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf

## custom loss function allowing weightings depending on class occurrences
# the weights comes from each patch
def weighted_binary_crossentropy(y_true, y_pred):
    foregroundPixels = K.sum(tf.to_float(y_true>0))
    backgroundPixels = K.sum(1-tf.to_float(y_true>0))
    totalNumberOfPixels = foregroundPixels + backgroundPixels
    foregroundWeights = totalNumberOfPixels / (foregroundPixels+1) # prevent zero division
    backgroundWeights = totalNumberOfPixels / (backgroundPixels+1)
    return K.mean(np.multiply(foregroundWeights * y_true + backgroundWeights * (1-y_true), K.binary_crossentropy(tf.to_float(y_true>0), y_pred)), axis=-1)

## custom loss function allowing weightings depending on class occurrences
# the weights comes from each patch
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
# for region of interest, choose
def total_weighted_binary_crossentropy_wrapper(percent):
    def total_weighted_binary_crossentropy(y_true, y_pred):
        care_region = int(percent * 256)
        tf_weight = tf.concat([tf.zeros([256 - care_region, 256], tf.float32), tf.ones([care_region, 256], tf.float32)], 0)
        y_true_new = tf.math.multiply(y_true, tf_weight)
        y_pred_new = tf.math.multiply(y_pred, tf_weight)
        foregroundPixels = K.sum(tf.to_float(y_true > 0))
        backgroundPixels = K.sum(1-tf.to_float(y_true > 0))
        totalNumberOfPixels = foregroundPixels + backgroundPixels
        foregroundWeights = totalNumberOfPixels / (foregroundPixels+1) # prevent zero division
        backgroundWeights = totalNumberOfPixels / (backgroundPixels+1)
        return K.mean(np.multiply(foregroundWeights * y_true_new + backgroundWeights * (1-y_true_new), K.binary_crossentropy(tf.to_float(y_true_new>0), y_pred_new)), axis=-1)
    return total_weighted_binary_crossentropy

## custom loss function allowing weightings depending on class occurrences
# for region of interest
def part_weighted_binary_crossentropy_wrapper(percent):
    def part_weighted_binary_crossentropy(y_true, y_pred):
        care_region = int(percent * 256)
        tf_weight = tf.concat([tf.zeros([256 - care_region, 256], tf.float32), tf.ones([care_region, 256], tf.float32)], 0)
        y_true_new = tf.math.multiply(y_true, tf_weight)
        y_pred_new = tf.math.multiply(y_pred, tf_weight)
        foregroundPixels = K.sum(tf.to_float(y_true_new > 0))
        backgroundPixels = K.sum(1 - tf.to_float(y_true_new > 0))
        totalNumberOfPixels = foregroundPixels + backgroundPixels
        foregroundWeights = totalNumberOfPixels / (foregroundPixels+1) # prevent zero division
        backgroundWeights = totalNumberOfPixels / (backgroundPixels+1)
        return K.mean(np.multiply(foregroundWeights * y_true_new + backgroundWeights * (1-y_true_new), K.binary_crossentropy(tf.to_float(y_true_new>0), y_pred_new)), axis=-1)
    return part_weighted_binary_crossentropy

def create_random_array(percent, sav_path):
    care_region = int(percent * 65536)
    x=np.ones(care_region)
    y=np.ones(65536-care_region)
    arr=np.concatenate((x,y))
    np.random.shuffle(arr)
    random_arr = np.reshape(arr, (256, 256))
    np.save(os.path.join(sav_path, 'random_array.npy'), random_arr)
    return random_arr

# for region of interest
def shuffle_part_WBC_wrapper(arr):
    def randompart_weighted_binary_crossentropy(y_true, y_pred):
        tf_weight = tf.to_float(arr)
        y_true_new = tf.math.multiply(y_true, tf_weight)
        y_pred_new = tf.math.multiply(y_pred, tf_weight)
        foregroundPixels = K.sum(tf.to_float(y_true_new > 0))
        backgroundPixels = K.sum(1 - tf.to_float(y_true_new > 0))
        totalNumberOfPixels = foregroundPixels + backgroundPixels
        foregroundWeights = totalNumberOfPixels / (foregroundPixels+1) # prevent zero division
        backgroundWeights = totalNumberOfPixels / (backgroundPixels+1)
        return K.mean(np.multiply(foregroundWeights * y_true_new + backgroundWeights * (1-y_true_new), K.binary_crossentropy(tf.to_float(y_true_new>0), y_pred_new)), axis=-1)
    return randompart_weighted_binary_crossentropy

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


def unet(pretrained_weights = None,input_size = (256,256,1), metrics=f1_score):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=weighted_binary_crossentropy1, metrics=[metrics])
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model