from keras.layers import Dense, Dropout, CuDNNGRU, Activation, Bidirectional, BatchNormalization
from attention_layer import AttentionWithContext


def BiGRU(Layer):
    x=Layer
    # FIRST BIGRU LAYER
    rnn_1 = Bidirectional(CuDNNGRU(150, return_sequences=True))(x)
    rnn_1 = Activation('relu')(rnn_1)
    rnn_1 = BatchNormalization()(rnn_1)
    rnn_1 = Dropout(0.2)(rnn_1)
    # SECOND BIGRU LAYER
    rnn_2 = Bidirectional(CuDNNGRU(150, return_sequences=True))(rnn_1)
    rnn_2 = Activation('relu')(rnn_2)
    rnn_2 = BatchNormalization()(rnn_2)
    rnn_2 = Dropout(0.2)(rnn_2)

    # ATTENTION WITH CONTEXT
    AC_layer = AttentionWithContext(hidden_dim=300)(rnn_2)
    AC_layer = BatchNormalization()(AC_layer)

    # representation
    rep = Dense(300, activation='relu', use_bias=True)(AC_layer)
    rep = BatchNormalization()(rep)

    return rep


