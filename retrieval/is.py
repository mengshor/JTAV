import numpy as np
from keras import Model
from keras.layers import Permute, Input, Dense, BatchNormalization, Concatenate

from attention_layer import AttentionWithContext
from audio_model import build_model
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
                y = np.tile(y, tuple([min(batch_size, lyr_len - n, )]+[1] * len(y.shape[1:])))
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
    train_image_file = data_file + 'train_image_densenet121.npy'

    test_audio_file = data_file + 'test_audio.npy'
    test_image_file = data_file + 'test_image_densenet121.npy'

    train_audio = np.load(train_audio_file)
    train_image = np.load(train_image_file)
    train_image = np.reshape(train_image, (train_image.shape[0], train_image.shape[3]))

    test_audio = np.load(test_audio_file)
    test_image = np.load(test_image_file)
    test_image = np.reshape(test_image, (test_image.shape[0], test_image.shape[3]))
    input_audios_dim = (215, 96, 1,)
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

    image_input = Input(shape=input_image_dim)
    image = image_input
    audio_input = Input(shape=input_audios_dim)
    audio_model = build_model(input_shape=input_audios_dim)
    audio = audio_model(audio_input)
    image = BatchNormalization()(image)
    image = Dense(300, activation='relu', use_bias=True)(image)
    audio = BatchNormalization()(audio)

    ################################################
    m = MatMul((300, 1), (1, 300))([image, audio])

    lyrics = AttentionWithContext(hidden_dim=300)(m)
    m = Permute((2, 1))(m)
    image = AttentionWithContext(hidden_dim=300)(m)
    ################################################

    rep = Concatenate()([image, lyrics])
    x = BatchNormalization()(rep)

    rep = Concatenate()([image, audio])
    rep = Dense(10, activation='relu', use_bias=bias)(rep)
    y = Dense(1, activation='sigmoid', use_bias=True)(rep)
    model = Model([image_input, audio_input], y)
    model.summary()
    model.compile(loss=loss, optimizer=opt)
    batch_size = 256
    model.fit([train_image, train_audio], train_label, verbose=2, batch_size=batch_size, epochs=5)
    results = model.predict_generator(generate_batch_data([test_image], [test_audio], predict_batch),
                                              steps=predict_steps)
    results = np.reshape(results, (-1, test_audio.shape[0]))
    test_label = np.reshape(test_label, (-1, test_audio.shape[0]))
    ranks = get_rank_matrix(results, test_label)
    scores = get_result_by_ranks(ranks, rec_k_list)
    print('########### {} ###########'.format('ia'))
    print(scores)
