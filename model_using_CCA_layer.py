import gzip
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from layer_CCA import CCA
# layers.Activation
from tf_objectives import cca_loss
tf.enable_eager_execution()


def constant_loss(y_true, y_pred):
    return y_pred


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


print('tf version: ' + tf.version.VERSION)
print('tf.keras version :' + tf.keras.__version__)


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')

train_set_x1, train_set_y1 = data1[0]
valid_set_x1, valid_set_y1 = data1[1]
test_set_x1, test_set_y1 = data1[2]

train_set_x2, train_set_y2 = data2[0]
valid_set_x2, valid_set_y2 = data2[1]
test_set_x2, test_set_y2 = data2[2]

i_shape1 = (784, )
i_shape2 = (784, )

o_dim = 10

l_size1 = 1024
l_size2 = 1024

learning_rate = 1e-3
use_all_singular_values = True
epoch_num = 100
batch_size = 800
reg_par = 1e-5

act = "sigmoid"
o_act = 'linear'


input1 = layers.Input(shape=i_shape1)
x1 = layers.Dense(l_size1, activation=act)(input1)
x1 = layers.Dense(l_size1, activation=act)(input1)
x1 = layers.Dense(l_size1, activation=act)(input1)
x1 = layers.Dense(o_dim, activation=o_act)(input1)

input2 = layers.Input(shape=i_shape2)
x2 = layers.Dense(l_size2, activation=act)(input2)
x2 = layers.Dense(l_size2, activation=act)(input2)
x2 = layers.Dense(l_size2, activation=act)(input2)
x2 = layers.Dense(o_dim, activation=o_act)(input2)

x3 = layers.concatenate([x1, x2])
print(x3.dtype)
a = CCA(x3, use_all_singular_values)
output = layers.Lambda(a.call, tuple((None, 1)))(x3)
print(output.dtype)

model = keras.Model(inputs=[input1, input2], outputs=output)
model.summary()

model.compile(optimizer='RMSprop',
              loss='binary_crossentropy', metrics='mean')
model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
          batch_size=batch_size, epochs=epoch_num, shuffle=True, verbose=1,
          validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))))

model.summary()


'''model1 = Sequential()
model1.add(layers.Dense(
    l_size1, input_dim=i_shape1[0], activation=act, kernel_regularizer=l2(reg_par)))
model1.add(layers.Dense(l_size1, activation=act,
                        kernel_regularizer=l2(reg_par)))
model1.add(layers.Dense(l_size1, activation=act,
                        kernel_regularizer=l2(reg_par)))
model1.add(layers.Dense(o_dim, activation=o_act,
                        kernel_regularizer=l2(reg_par)))

model2 = Sequential()
model2.add(layers.Dense(
    l_size2, input_dim=i_shape2[0], activation=act, kernel_regularizer=l2(reg_par)))
model2.add(layers.Dense(l_size2, activation=act,
                        kernel_regularizer=l2(reg_par)))
model2.add(layers.Dense(l_size2, activation=act,
                        kernel_regularizer=l2(reg_par)))
model2.add(layers.Dense(o_dim, activation=o_act,
                        kernel_regularizer=l2(reg_par))) '''

'''
model.add(keras.layers.Concatenate([model1, model2]))

model_optimizer = keras.optimizers.RMSprop(lr=learning_rate)
model.compile(loss=cca_loss(1, use_all_singular_values),
              optimizer=model_optimizer)
# model.build(tf.convert_to_tensor([i_shape1, i_shape2]))


# best weights are saved in "temp_weights.hdf5" during training
# it is done to return the best model based on the validation loss
checkpointer = ModelCheckpoint(
    filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

# used dummy Y because labels are not used in the loss function
model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
          batch_size=batch_size, epochs=epoch_num, shuffle=True,
          validation_data=([valid_set_x1, valid_set_x2],
                           np.zeros(len(valid_set_x1))),
          callbacks=[checkpointer]) '''

'''
# used dummy Y because labels are not used in the loss function
model.fit([train_set_x1, train_set_x2], np.zeros(50000),
          batch_size=batch_size, epochs=epoch_num, shuffle=True,
          validation_data=([valid_set_x1, valid_set_x2], np.zeros(50000)))
model.summary() '''
