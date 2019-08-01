from layer_CCA import CCA
import tensorflow as tf
import tensorflow.keras as keras

a = CCA()
model = tf.keras.Sequential()
model.add(a)
