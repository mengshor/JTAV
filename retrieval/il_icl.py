import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Permute, Input, Dense, BatchNormalization, Concatenate, Flatten, RepeatVector, \
    MaxPool1D, Multiply

from BiGRU import BiGRU
from attention_layer import AttentionWithContext
from evaluation import get_rank_matrix, get_result_by_ranks
from matmul import MatMul


def generate_batch_data(X, Y, batch_size):
    lyr_len = len(Y[0])
    while True:
        n = 0
        while n < lyr_len:
            out_X = []
            for x in X:
                x = x[n: min(n + batch_size, lyr_len)]
                x = np.repeat(x, lyr_len, axis=0)
                out_X.append(x)
            out_Y = []
            for y in Y:
                y = np.tile(y, (min(batch_size, lyr_len - n, ), 1, 1))
                out_Y.append(y)
            n += batch_size
            yield out_X + out_Y


if __name__ == '__main__':
    dataset = 'b'
    if dataset == 'a':
        rec_k_list = [1, 5, 10]
    else:
        rec_k_list = [10, 50, 100]

    supply = True

    # data file
    data_file = 'dataset_{}/train_data/'.format(dataset)
    train_lyric_file = data_file + 'train_lyric.npy'
    train_image_file = data_file + 'train_image_densenet121.npy'

    test_lyric_file = data_file + 'test_lyric.npy'
    test_image_file = data_file + 'test_image_densenet121.npy'

    train_lyric = np.load(train_lyric_file)
    train_image = np.load(train_image_file)
    train_image = np.reshape(train_image, (train_image.shape[0], train_image.shape[3]))

    test_lyric = np.load(test_lyric_file)
    test_image = np.load(test_image_file)
    test_image = np.reshape(test_image, (test_image.shape[0], test_image.shape[3]))
    input_lyrics_dim = (100, 300,)
    input_image_dim = (1024,)
    if supply:
        train_caption_file = data_file + 'train_caption.npy'
        test_caption_file = data_file + 'test_caption.npy'
        train_caption = np.load(train_caption_file)
        test_caption = np.load(test_caption_file)
        input_caption_dim = (5, 300,)

    # label file
    label_file = 'dataset_{}/splits/'.format(dataset)
    train_label_file = label_file + 'train.npy'
    test_label_file = label_file + 'test.npy'

    train_label = np.load(train_label_file)
    test_label = np.load(test_label_file)

    activation = 'tanh'
    bias = 'True'
    opt = 'rmsprop'
    loss = 'binary_crossentropy'

    predict_batch = 5
    predict_steps = test_image.shape[0] // predict_batch

    image_input = Input(shape=input_image_dim)
    image = image_input
    lyrics_input = Input(shape=input_lyrics_dim)
    lyrics = lyrics_input
    image = BatchNormalization()(image)
    image = Dense(300, activation='relu', use_bias=True)(image)
    if supply:
        # COMPUTE EXTRA ATTENTION WEIGHTS
        summary = Input(shape=input_caption_dim)
        weight = MaxPool1D(pool_size=5)(summary)
        weight = Flatten()(weight)
        weight = Dense(300, activation='softmax', use_bias=True)(weight)
        weight = RepeatVector(n=100)(weight)
        weight = BatchNormalization()(weight)
        lyrics = Multiply()([lyrics, weight])
        lyrics = BatchNormalization()(lyrics)
    lyrics = BiGRU(lyrics)

    ################################################
    m = MatMul((300, 1), (1, 300))([image, lyrics])

    lyrics = AttentionWithContext(hidden_dim=300)(m)
    m = Permute((2, 1))(m)
    image = AttentionWithContext(hidden_dim=300)(m)
    ################################################

    rep = Concatenate()([image, lyrics])
    x = BatchNormalization()(rep)

    x = Dropout(0.2)(x)
    rep = Dense(10, activation='relu', use_bias=bias)(rep)
    y = Dense(1, activation='sigmoid', use_bias=True)(rep)
    if supply:
        model = Model([image_input, summary, lyrics_input], y)
    else:
        model = Model([image_input, lyrics_input], y)
    model.summary()
    model.compile(loss=loss, optimizer=opt)
    early = EarlyStopping('val_loss', patience=2)
    if supply:
        model.fit([train_image, train_caption, train_lyric], train_label, verbose=2, batch_size=256, epochs=5)
        results = model.predict_generator(
                generate_batch_data([test_image, test_caption], [test_lyric], predict_batch), steps=predict_steps)
    else:
        model.fit([train_image, train_lyric], train_label, verbose=2, batch_size=192, epochs=5)
        results = model.predict_generator(generate_batch_data([test_image], [test_lyric], predict_batch),
                                              steps=predict_steps)
    results = np.reshape(results, (-1, test_lyric.shape[0]))
    test_label = np.reshape(test_label, (-1, test_lyric.shape[0]))
    ranks = get_rank_matrix(results, test_label)
    scores = get_result_by_ranks(ranks, rec_k_list)
    print('########### {} ###########'.format('complete'))
    print(scores)
