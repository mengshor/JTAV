from keras.engine import Layer
import tensorflow as tf
import keras.backend as K


class MatMul(Layer):
    def __init__(self, left_shape, right_shape, **kwargs):
        self.left_shape = left_shape
        self.right_shape = right_shape
        super(MatMul, self).__init__(**kwargs)

    def call(self, mat_pair, mask=None):
        '''
        mat_pair: a tuple or list of the two matrixs to be dot multiplied
        '''
        left, right = mat_pair

        left = K.expand_dims(left)
        right = K.expand_dims(right, axis=1)
        x = tf.matmul(left, right)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.left_shape[0], self.right_shape[1])
