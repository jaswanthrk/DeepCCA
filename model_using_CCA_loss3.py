try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
import gzip
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
#from tensorflow.keras.utils.generic_utils import get_custom_objects
from c_objectives import cca_loss
import matplotlib.pyplot as plt
from linear_cca import linear_cca
from utilities import load_data, svm_classify
# layers.Activation
# tf.enable_eager_execution()


print('tf version: ' + tf.version.VERSION)
print('tf.keras version :' + tf.keras.__version__)


def model_learn(data1, data2, model, epoch_num, batch_size, learning_rate, use_all_singular_values):

    train_set_x1, train_set_y1 = data1[0]
    valid_set_x1, valid_set_y1 = data1[1]
    test_set_x1, test_set_y1 = data1[2]

    train_set_x2, train_set_y2 = data2[0]
    valid_set_x2, valid_set_y2 = data2[1]
    test_set_x2, test_set_y2 = data2[2]

    checkpointer = ModelCheckpoint(
        filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(o_dim, use_all_singular_values),
                  optimizer=model_optimizer)

    history = model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)), epochs=epoch_num, shuffle=True, validation_data=(
        [valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))), callbacks=[checkpointer])

    loss = plt.plot(history.history['loss'])
    val_loss = plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('losses.png')

    model.load_weights("temp_weights.h5")

    results = model.evaluate([test_set_x1, test_set_x2],
                             np.zeros(len(test_set_x1)), batch_size=batch_size, verbose=1)

    print('loss on test data: ', results)

    results = model.evaluate([valid_set_x1, valid_set_x2],
                             np.zeros(len(valid_set_x1)), batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)

    return model


def test_model(model, data1, data2, outdim_size, apply_linear_cca):
    """produce the new features by using the trained model
    # Arguments
        model: the trained model
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively.
            Data should be packed like
            ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        outdim_size: dimension of new features
        apply_linear_cca: if to apply linear CCA on the new features
    # Returns
        new features packed like
            ((new X for train - view 1, new X for train - view 2, Y for train),
            (new X for validation - view 1, new X for validation - view 2, Y for validation),
            (new X for test - view 1, new X for test - view 2, Y for test))
    """

    # producing the new features
    new_data = []
    for k in range(3):
        pred_out = model.predict([data1[k][0], data2[k][0]])
        r = int(pred_out.shape[1] / 2)
        new_data.append([pred_out[:, :r], pred_out[:, r:], data1[k][1]])

    # based on the DCCA paper, a linear CCA should be applied on the output of the networks because
    # the loss function actually estimates the correlation when a linear CCA is applied to the output of the networks
    # however it does not improve the performance significantly
    if apply_linear_cca:
        w = [None, None]
        m = [None, None]
        print("Linear CCA started!")
        w[0], w[1], m[0], m[1], _ = linear_cca(
            new_data[0][0], new_data[0][1], outdim_size)
        print("Linear CCA ended!")

        # Something done in the original MATLAB implementation of DCCA, do not know exactly why;)
        # it did not affect the performance significantly on the noisy MNIST dataset
        #s = np.sign(w[0][0,:])
        #s = s.reshape([1, -1]).repeat(w[0].shape[0], axis=0)
        #w[0] = w[0] * s
        #w[1] = w[1] * s
        ###

        for k in range(3):
            data_num = len(new_data[k][0])
            for v in range(2):
                new_data[k][v] -= m[v].reshape([1, -1]
                                               ).repeat(data_num, axis=0)
                new_data[k][v] = np.dot(new_data[k][v], w[v])

    return new_data


save_to = './new_features.gz'

i_shape1 = (784, )
i_shape2 = (784, )

o_dim = 10

l_size1 = 1024
l_size2 = 1024

learning_rate = 1e-4
use_all_singular_values = True
epoch_num = 50
batch_size = 32
reg_par = 1e-4

act = "sigmoid"
o_act = 'linear'

data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')


input1 = layers.Input(shape=i_shape1)
x1 = layers.Dense(l_size1, activation=act,
                  kernel_regularizer=l2(reg_par))(input1)
x1 = layers.Dense(l_size1, activation=act,
                  kernel_regularizer=l2(reg_par))(input1)
x1 = layers.Dense(o_dim, activation=o_act,
                  kernel_regularizer=l2(reg_par))(input1)

input2 = layers.Input(shape=i_shape2)
x2 = layers.Dense(l_size2, activation=act,
                  kernel_regularizer=l2(reg_par))(input2)
x2 = layers.Dense(l_size2, activation=act,
                  kernel_regularizer=l2(reg_par))(input2)
x2 = layers.Dense(o_dim, activation=o_act,
                  kernel_regularizer=l2(reg_par))(input2)

x3 = layers.concatenate([x1, x2])

model = keras.Model(inputs=[input1, input2], outputs=x3)
model.summary()

model = model_learn(data1, data2, model, epoch_num, batch_size,
                    learning_rate, use_all_singular_values)


apply_linear_cca = True
new_data = test_model(model, data1, data2, o_dim, apply_linear_cca)

[test_acc, valid_acc] = svm_classify(new_data, C=0.01)
print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
print("Accuracy on view 1 (test data) is:", test_acc*100.0)

# Saving new features in a gzip pickled file specified by save_to
print('saving new features ...')
f1 = gzip.open(save_to, 'wb')
thepickle.dump(new_data, f1)
f1.close()
