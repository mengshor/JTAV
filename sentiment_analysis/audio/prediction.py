from build_model import build_model
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from evaluation import evaluation
import tensorflow as tf
from tensorflow.contrib.losses import cosine_distance
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = (ylen + batch_size - 1) // batch_size
    while (True):
        for i in range(loopcount):
            yield x[i * batch_size:min((i + 1) * batch_size, ylen)], y[i * batch_size:min((i + 1) * batch_size, ylen)]


def cosine(x, y):
    x = tf.nn.l2_normalize(x, dim=1)
    y = tf.nn.l2_normalize(y, dim=1)
    return cosine_distance(predictions=x, labels=y, dim=1)

if __name__ == '__main__':
    kind = 'mel'
    fre = 96
    sr = 22050
    act='relu'
    mod = build_model(input_shape=(215, fre, 1))
    mod.summary()
    mod.compile(optimizer='rmsprop', loss='binary_crossentropy')
    Wsave = mod.get_weights()
    batch_size = 32

    print('loading data and label')
    train_patch = np.load('data/X_train_patches_shutter_mood_1x10_{}_{}_{}.npy'.format(sr, kind, fre))
    train_labels = np.load('label/y_train_class_1_shutter.npy')

    test_patch = np.load('data/X_test_patches_shutter_mood_1x10_{}_{}_{}.npy'.format(sr, kind, fre))
    test_labels = np.load('label/y_test_class_1_shutter.npy')

    val_patch = np.load('data/X_val_patches_shutter_mood_1x10_{}_{}_{}.npy'.format(sr, kind, fre))
    val_labels = np.load('label/y_val_class_1_shutter.npy')
    print('finish loading')
    print('train count:', train_patch.shape)
    print('test count:', test_patch.shape)
    print('val count:', val_patch.shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    logger = CSVLogger('loss_log_1.csv')

    print('start training')
    recorder = open('result_dis.tsv', 'w')
    mod.fit_generator(generate_batch_data_random(train_patch, train_labels, batch_size),
                          steps_per_epoch=(len(train_labels) + batch_size - 1) //
                                          batch_size,
                          epochs=500,
                          validation_data=[val_patch, val_labels],
                          verbose=0,
                          callbacks=[early_stopping, logger])
    results = mod.predict(test_patch)

    auc, f1, precision = evaluation(results, test_labels, '22050')