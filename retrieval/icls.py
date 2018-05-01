from BiGRU import BiGRU
from evaluation import get_rank_matrix, get_result_by_ranks
import numpy as np
from keras.layers import Permute, Input, Dense, Lambda, BatchNormalization, Concatenate, Flatten, RepeatVector, \
    MaxPool1D, \
    Multiply
from keras import Model
import keras.backend as K
from audio_model import build_model
from matmul import MatMul
from attention_layer import AttentionWithContext


def generate_batch_train_data(X, y, batch_size):
    ylen = y.shape[0]
    loopcount = (ylen + batch_size - 1) // batch_size
    while (True):
        for i in range(loopcount):
            out_X = []
            for x in X:
                out_X.append(x[i * batch_size:min((i + 1) * batch_size, ylen)])
            yield out_X, y[i * batch_size:min((i + 1) * batch_size, ylen)]


def generate_batch_data(X, Y, batch_size):
    lyr_len = Y[0].shape[0]
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
                y = np.tile(y, tuple([min(batch_size, lyr_len - n, )] + [1] * len(y.shape[1:])))
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
    train_audio_file = data_file + 'train_audio.npy'
    train_lyric_file = data_file + 'train_lyric.npy'
    train_image_file = data_file + 'train_image_densenet121.npy'
    train_caption_file = data_file + 'train_caption.npy'

    test_audio_file = data_file + 'test_audio.npy'
    test_lyric_file = data_file + 'test_lyric.npy'
    test_image_file = data_file + 'test_image_densenet121.npy'
    test_caption_file = data_file + 'test_caption.npy'

    train_audio = np.load(train_audio_file)
    train_lyric = np.load(train_lyric_file)
    train_image = np.load(train_image_file)
    train_caption = np.load(train_caption_file)
    train_image = np.reshape(train_image, (train_image.shape[0], train_image.shape[3]))

    test_audio = np.load(test_audio_file)
    test_lyric = np.load(test_lyric_file)
    test_image = np.load(test_image_file)
    test_caption = np.load(test_caption_file)
    test_image = np.reshape(test_image, (test_image.shape[0], test_image.shape[3]))

    input_audios_dim = (215, 96, 1,)
    input_lyrics_dim = (100, 300,)
    input_caption_dim = (5, 300,)
    input_image_dim = (1024,)

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

    predict_batch = 1
    predict_steps = test_image.shape[0] // predict_batch

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

    audio_input = Input(shape=input_audios_dim)
    audio = BatchNormalization()(audio_input)
    audio_model = build_model(input_shape=input_audios_dim)
    audio = audio_model(audio)

    image_input = Input(shape=input_image_dim)
    image = image_input
    image = BatchNormalization()(image)
    image = Dense(300, activation='relu', use_bias=True)(image)
    audio = Lambda(lambda t: K.l2_normalize(t, axis=-1))(audio)
    lyrics = Lambda(lambda t: K.l2_normalize(t, axis=-1))(lyrics)

    ################################################
    m_1 = MatMul((300, 1), (1, 300))([audio, lyrics])
    m_2 = MatMul((300, 1), (1, 300))([audio, image])
    m_3 = MatMul((300, 1), (1, 300))([lyrics, image])

    h_1 = AttentionWithContext(hidden_dim=300)(m_1)
    m_1 = Permute((2, 1))(m_1)
    h_2 = AttentionWithContext(hidden_dim=300)(m_1)
    m_1 = Concatenate()([h_1, h_2])

    h_3 = AttentionWithContext(hidden_dim=300)(m_2)
    m_2 = Permute((2, 1))(m_2)
    h_4 = AttentionWithContext(hidden_dim=300)(m_2)
    m_2 = Concatenate()([h_3, h_4])
    #
    h_5 = AttentionWithContext(hidden_dim=300)(m_3)
    m_3 = Permute((2, 1))(m_3)
    h_6 = AttentionWithContext(hidden_dim=300)(m_3)
    m_3 = Concatenate()([h_5, h_6])
    ################################################

    rep = Concatenate()([m_1, m_2, m_3])
    rep = BatchNormalization()(rep)
    rep = Dense(10, activation='relu', use_bias=bias)(rep)
    y = Dense(1, activation='sigmoid', use_bias=True)(rep)
    model = Model([image_input, summary, audio_input, lyrics_input], y)
    model.summary()
    batch_size = 32
    model.compile(loss=loss, optimizer=opt)
    model.fit([train_image, train_caption, train_audio, train_lyric], train_label, verbose=2, batch_size=batch_size,
              epochs=5)
    results = model.predict_generator(
        generate_batch_data([test_image, test_caption], [test_audio, test_lyric], predict_batch),
        steps=predict_steps)
    results = np.reshape(results, (-1, test_lyric.shape[0]))
    test_label = np.reshape(test_label, (-1, test_lyric.shape[0]))
    ranks = get_rank_matrix(results, test_label)
    scores = get_result_by_ranks(ranks, rec_k_list)
    print('########### {} ###########'.format('complete'))
    print(scores)
