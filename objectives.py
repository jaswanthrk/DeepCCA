import tensorflow as tf


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_cca_objective(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented by Theano tensor operations, and does not work on Tensorflow backend
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = y_pred.shape[1]//2

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = tf.transpose(y_pred[:, 0:o1])
        H2 = tf.transpose(y_pred[:, o1:o1+o2])

        m = H1.shape[1]

        H1bar = H1 - (tf.math.divide(1, m)) * tf.dot(H1, tf.ones([m, m]))
        H2bar = H2 - (tf.math.divide(1, m)) * tf.dot(H2, tf.ones([m, m]))

        SigmaHat12 = (tf.math.divide(1, m-1)) * \
            tf.dot(H1bar, tf.transpose(H2bar))
        SigmaHat11 = (tf.math.divide(1, m-1)) * tf.dot(H1bar,
                                                       tf.transpose(H1bar)) + r1 * tf.eye(o1)
        SigmaHat22 = (tf.math.divide(1, m-1)) * tf.dot(H2bar,
                                                       tf.transpose(H2bar)) + r2 * tf.eye(o2)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.nlinalg.eigh(SigmaHat11)
        [D2, V2] = tf.nlinalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = tf.gt(D1, eps).nonzero()[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = tf.gt(D2, eps).nonzero()[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = tf.dot(
            tf.dot(V1, tf.nlinalg.diag(D1 ** -0.5)), tf.transpose(V1))
        SigmaHat22RootInv = tf.dot(
            tf.dot(V2, tf.nlinalg.diag(D2 ** -0.5)), tf.transpose(V2))

        Tval = tf.dot(tf.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = tf.sqrt(tf.nlinalg.trace(tf.dot(tf.transpose(Tval), Tval)))
        else:
            # just the top outdim_size singular values are used
            [U, V] = tf.nlinalg.eigh(T.dot(tf.transpose(Tval), Tval))
            U = U[tf.gt(U, eps).nonzero()[0]]
            U = U.sort()
            corr = tf.sum(tf.sqrt(U[0:outdim_size]))

        return -corr

    return inner_cca_objective


""" WHAT IS theano.tensor.nonzero() ?

In numpy, if I have a boolean array, I can use it to select elements of another array:

>>> import numpy as np
>>> x = np.array([1, 2, 3])
>>> idx = np.array([True, False, True])
>>> x[idx] 
array([1, 3])
I need to do this in theano. This is what I tried, but I got an unexpected result.

>>> from theano import tensor as T
>>> x = T.vector()
>>> idx = T.ivector()
>>> y = x[idx]
>>> y.eval({x: np.array([1,2,3]), idx: np.array([True, False, True])})
array([ 2.,  1.,  2.])
Can someone explain the theano result and suggest how to get the numpy result? I

SOLUTION : 
y = x[idx.nonzero()]
y.eval({x: np.array([1,2,3]),idx: np.array([True, False, True])})

"""
