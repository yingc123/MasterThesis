'''
The code for 2D U-net training
Based on https://github.com/zhixuhao/unet
'''


from model import *
from data import *
import argparse
import matplotlib.pyplot as plt
from keras.optimizers import *
from list_generate import *
from keras.models import load_model
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy
import pandas

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s', '--steps_per_epoch', type=int, default=2000)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-a', '--aug', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-t', '--target_size', type=int, nargs='+', default=(256,256))
    parser.add_argument('-d', '--dataset', type=str, default='drosophila')
    parser.add_argument('-c', '--comment', type=str, default=None)
    parser.add_argument('-m', '--mask', type=str, default='mask')
    parser.add_argument('--monitor', type=str, default='val_loss')
    return parser.parse_args()
args = parse_args()

def plot_loss(history, sav_path):
    # training and validation loss
    acc = history.history['binary_crossentropy']
    val_acc = history.history['val_binary_crossentropy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(args.epoch)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(sav_path, 'History.png'))

def main():
    path = '/work/scratch/ychen/segmentation/network_method/2D_U_Net'
    data_path= os.path.join(path, 'data', args.dataset)
    save_path= os.path.join(path, 'model', args.dataset, args.mask+str(args.epoch))
    os.makedirs(save_path, exist_ok=True)
    #print(data_path, save_path)

    # training process
    if args.train:
        if args.aug:
            aug_path = os.path.join(save_path, 'aug')
            os.makedirs(aug_path, exist_ok=True)
            valid_aug_path = os.path.join(save_path, 'valid_aug')
            os.makedirs(valid_aug_path, exist_ok=True)

            train_img = augGenerator(args.batch_size, data_path, 'img_train', args.mask+'_train',
                                 data_gen_args, save_to_dir=aug_path, target_size=args.target_size)
            valid_img = augGenerator(args.batch_size, data_path, 'img_valid', args.mask+'_valid',
                                 data_gen_args, save_to_dir=valid_aug_path, target_size=args.target_size)
        else:
            train_img = augGenerator(args.batch_size, data_path, 'img_train', args.mask+'_train',
                                 data_gen_args, save_to_dir=None, target_size=args.target_size)
            valid_img = augGenerator(args.batch_size, data_path, 'img_valid', args.mask+'_valid',
                                 data_gen_args, save_to_dir=None, target_size=args.target_size)
        # train_img = augGenerator(args.batch_size, os.path.join(data_path, 'train'), 'image', 'label',
        #                          data_gen_args, save_to_dir=os.path.join(data_path, 'aug'), target_size=args.target_size)
        # valid_img = augGenerator(args.batch_size, os.path.join(data_path, 'valid'), 'image', 'label',
        #                          data_gen_args, save_to_dir=None, target_size=args.target_size)

        model = unet(metrics=[BinaryCrossentropy(), f1_score])
        model_checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.hdf5'), monitor=args.monitor, verbose=1, save_best_only=True)
        num_train_img = len(os.listdir(os.path.join(data_path, 'img_train')))
        num_valid_img = len(os.listdir(os.path.join(data_path, 'img_valid')))
        print(num_train_img, num_valid_img)
        train_history = model.fit_generator(train_img, steps_per_epoch=num_train_img//args.batch_size, epochs=args.epoch, \
                                      verbose=1, callbacks=[model_checkpoint], validation_data=valid_img, \
                                      validation_steps=num_valid_img//args.batch_size) #, class_weight=[0.51077903, 25000.69317881])
        # plot and save plot
        df_outcomes = pandas.DataFrame(data=train_history.history)
        df_outcomes.to_csv(os.path.join(save_path, 'history.csv'), sep=';')
        try:
            # plot and save figure
            ax = df_outcomes.plot(title=args.mask+str(args.epoch), grid=True, xlim=(0, train_history.epoch[-1] + 1),
                                  xticks=np.arange(0, train_history.epoch[-1] + 1,
                                                   10 * (1 + (train_history.epoch[-1] + 1) // 100)), figsize=(12, 10))
            ax.set_xlabel('epochs')
            ax.set_ylabel('scores')
            ax.figure.savefig(os.path.join(save_path, 'history.tif'))
            plt.close(ax.figure)
        except Exception as e:
            print('Could not plot training progress due to the following error: {0}'.format(e))
        else:
            pass

    # test process
    if args.test:
        # predict test data
        test_img_path = os.path.join(data_path, 'img_test')
        test_image = normalGenerator(test_img_path, target_size=args.target_size, mask=False)
        model = unet()
        model.load_weights(os.path.join(save_path, 'model.hdf5'))
        results = model.predict_generator(test_image, len(os.listdir(test_img_path)), verbose=50)
        #np.save(os.path.join(save_path, 'pred.npy'), results, allow_pickle=True)
        test_save_path = os.path.join(save_path, 'pred_result')
        os.makedirs(test_save_path, exist_ok=True)
        saveResult(test_save_path, results, test_img_path)

        # test_mask_path = os.path.join(data_path, 'mask_test')
        # test_label = normalGenerator(test_mask_path, target_size=args.target_size, mask=True)
        #
        # test_loss, test_acc = model.evaluate(test_image, test_label)
        # np.save(os.path.join(save_path, 'test_loss_acc.npy'), (test_loss, test_acc))




if __name__=='__main__':
    main()