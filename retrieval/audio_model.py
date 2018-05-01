import keras.backend as K
from keras import Model
from keras.layers import Input, Dense, CuDNNGRU, Conv2D, BatchNormalization, Activation, AveragePooling2D, \
    Concatenate, ZeroPadding2D, MaxPooling2D, Bidirectional, Reshape


def dense_block(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 64, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def build_model(input_shape):
    H_f = (((input_shape[0] + 2) // 2 - 1) // 2 + 1) // 2
    W_f = (((input_shape[1] - 64) // 2 - 1) // 2 + 1) // 2
    audio = Input(shape=input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = audio
    x = BatchNormalization()(x)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
    x = Conv2D(256, (4, 70), strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(4, strides=2, name='pool1')(x)

    x = dense_block(x, 2, name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, 2, name='conv3')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)
    x = Reshape((H_f, -1,))(x)
    x = Bidirectional(CuDNNGRU(W_f * 192, return_sequences=True))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(CuDNNGRU(W_f * 96, return_sequences=False))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(name='rep')(x)
    x = Dense(300, activation='relu')(x)
    model = Model(audio, x)
    return model
