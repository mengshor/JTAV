from BiGRU import BiGRU
import numpy as np
from keras.layers import Input, Dense, Lambda, BatchNormalization, Flatten, RepeatVector, \
    MaxPool1D, \
    Multiply
from keras import Model
import keras.backend as K

if __name__ == '__main__':
    dataset = 'b'
    if dataset == 'a':
        rec_k_list = [1, 5, 10]
    else:
        rec_k_list = [10, 50, 100]

    supply = True

    # data file
    data_file = 'train_data/'
    train_lyric_file = data_file + 'train_lyric.npy'
    train_caption_file = data_file + 'train_caption.npy'

    test_lyric_file = data_file + 'test_lyric.npy'
    test_caption_file = data_file + 'test_caption.npy'

    val_lyric_file = data_file + 'val_lyric.npy'
    val_caption_file = data_file + 'val_caption.npy'

    train_lyric = np.load(train_lyric_file)
    train_caption = np.load(train_caption_file)

    test_lyric = np.load(test_lyric_file)
    test_caption = np.load(test_caption_file)

    val_lyric = np.load(val_lyric_file)
    val_caption = np.load(val_caption_file)

    input_lyrics_dim = (100, 300,)
    input_caption_dim = (5, 300,)

    # label file
    label_file = 'splits/'.format(dataset)
    train_label_file = label_file + 'train.npy'
    test_label_file = label_file + 'test.npy'

    train_label = np.load(train_label_file)
    test_label = np.load(test_label_file)

    activation = 'tanh'
    bias = 'True'
    opt = 'rmsprop'
    loss = 'binary_crossentropy'

    predict_batch = 1

    lyrics_input = Input(shape=input_lyrics_dim)
    lyrics = lyrics_input
    lyrics = BatchNormalization()(lyrics)

    summary = Input(shape=input_caption_dim)
    weight = MaxPool1D(pool_size=5)(summary)
    weight = Flatten()(weight)
    weight = Dense(300, activation='softmax', use_bias=True)(weight)
    weight = RepeatVector(n=100)(weight)
    weight = BatchNormalization()(weight)
    lyrics = Multiply()([lyrics, weight])
    lyrics = BatchNormalization()(lyrics)
    lyrics = BiGRU(lyrics)

    lyrics = Lambda(lambda t: K.l2_normalize(t, axis=-1))(lyrics)

    y = Dense(1, activation='sigmoid', use_bias=True)(lyrics)
    model = Model([summary, lyrics_input], y)
    model.summary()
    batch_size = 32
    model.compile(loss=loss, optimizer=opt)
    model.fit([train_caption, train_lyric], train_label, verbose=2, batch_size=batch_size,
              epochs=5)
    results = model.predict([test_caption, test_lyric])

