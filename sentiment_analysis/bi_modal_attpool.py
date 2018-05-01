from keras.callbacks import EarlyStopping
from keras.layers import Input, BatchNormalization, Dropout, Dense, Permute, Concatenate, Lambda
from keras import Model
import keras.backend as K
from evaluation import evaluation
from matmul import MatMul
import numpy as np
from keras import optimizers
from attention_layer import AttentionWithContext

# modal_1 = 'image'
modal_1 = 'audio'
# modal_1 = 'text'

modal_2 = 'image'
# modal_2 = 'audio'
# modal_2 = 'text'

train_data_path_1 = 'data/{}/X_train_{}_shutter_mood.npy'.format(modal_1, modal_1)
val_data_path_1 = 'data/{}/X_val_{}_shutter_mood.npy'.format(modal_1, modal_1)
test_data_path_1 = 'data/{}/X_test_{}_shutter_mood.npy'.format(modal_1, modal_1)

print('start loading modal 1')
train_data_1 = np.load(train_data_path_1)
val_data_1 = np.load(val_data_path_1)
test_data_1 = np.load(test_data_path_1)
print('finish loading modal 1')
input_dim_1 = (test_data_1.shape[1],)

train_data_path_2 = 'data/{}/X_train_{}_shutter_mood.npy'.format(modal_2, modal_2)
val_data_path_2 = 'data/{}/X_val_{}_shutter_mood.npy'.format(modal_2, modal_2)
test_data_path_2 = 'data/{}/X_test_{}_shutter_mood.npy'.format(modal_2, modal_2)

print('start loading modal 2')
train_data_2 = np.load(train_data_path_2)
val_data_2 = np.load(val_data_path_2)
test_data_2 = np.load(test_data_path_2)
print('finish loading modal 2')
input_dim_2 = (test_data_2.shape[1],)

train_label_path = 'label/y_train_shutter_mood.npy'
val_label_path = 'label/y_val_shutter_mood.npy'
test_label_path = 'label/y_test_shutter_mood.npy'

print('start loading label')
train_label = np.load(train_label_path).astype(dtype='float32')
val_label = np.load(val_label_path).astype(dtype='float32')
test_label = np.load(test_label_path).astype(dtype='float32')
print('finish loading label')

modal_1_input = Input(shape=input_dim_1)
x_1 = modal_1_input
x_1 = BatchNormalization()(x_1)

modal_2_input = Input(shape=input_dim_2)
x_2 = modal_2_input
x_2 = BatchNormalization()(x_2)

x = MatMul((input_dim_1[0], 1), (1, input_dim_2[0]))([x_1, x_2])

h_1 = AttentionWithContext(hidden_dim=100)(x)
x = Permute((2, 1))(x)
h_2 = AttentionWithContext(hidden_dim=100)(x)

h_1 = Lambda(lambda x: K.l2_normalize(x))(h_1)
h_2 = Lambda(lambda x: K.l2_normalize(x))(h_2)

x = Concatenate()([h_1, h_2])
rep = x

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

y = Dense(1, activation='sigmoid')(x)

model = Model([modal_1_input, modal_2_input], y)
model.summary()
opt = optimizers.Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer='adam')

batch_size = 32

early_stopping = EarlyStopping('val_loss', patience=1)

model.fit([train_data_1, train_data_2], train_label, batch_size=batch_size,
                              epochs=500,
                              validation_data=[[val_data_1, val_data_2], val_label],
                              verbose=0,
                              callbacks=[early_stopping])
results = model.predict([test_data_1, test_data_2])
auc, f1, precision = evaluation(results, test_label, 0.7)


