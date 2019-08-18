from __future__ import print_function

# import theano.tensor as T
import tensorflow as tf


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_cca_objective(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented on Tensorflow based on github@VahidooX's cca loss on Theano.
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = int(y_pred.shape[1] // 2)
        dim = outdim_size

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = tf.transpose(y_pred[:, 0:o1])
        H2 = tf.transpose(y_pred[:, o1:o1 + o2])

        m = tf.shape(H1)[1]
        N = m
        d1 = o1
        d2 = o2
        rcov1 = r1
        rcov2 = r2
        eps_eig = eps

        m1 = tf.reduce_mean(H1, axis=0, keep_dims=True)
        H1 = tf.subtract(H1, m1)

        m2 = tf.reduce_mean(H2, axis=0, keep_dims=True)
        H2 = tf.subtract(H2, m2)

        S11 = tf.cast(tf.divide(1, N - 1), tf.float32) * \
            tf.matmul(tf.transpose(H1), H1) + rcov1 * tf.eye(d1)
        S22 = tf.cast(tf.divide(1, N - 1), tf.float32) * \
            tf.matmul(tf.transpose(H2), H2) + rcov2 * tf.eye(d2)
        S12 = tf.cast(tf.divide(1, N - 1), tf.float32) * \
            tf.matmul(tf.transpose(H1), H2)

        E1, V1 = tf.self_adjoint_eig(S11)
        E2, V2 = tf.self_adjoint_eig(S22)

        # For numerical stability.
        idx1 = tf.where(E1 > eps_eig)[:, 0]
        E1 = tf.gather(E1, idx1)
        V1 = tf.gather(V1, idx1, axis=1)

        idx2 = tf.where(E2 > eps_eig)[:, 0]
        E2 = tf.gather(E2, idx2)
        V2 = tf.gather(V2, idx2, axis=1)

        K11 = tf.matmul(tf.matmul(V1, tf.diag(
            tf.reciprocal(tf.sqrt(E1)))), tf.transpose(V1))
        K22 = tf.matmul(tf.matmul(V2, tf.diag(
            tf.reciprocal(tf.sqrt(E2)))), tf.transpose(V2))
        T = tf.matmul(tf.matmul(K11, S12), K22)

        # Eigenvalues are sorted in increasing order.
        E2, U = tf.self_adjoint_eig(tf.matmul(T, tf.transpose(T)))

        return tf.reduce_sum(tf.sqrt(E2[-dim:]))

    return inner_cca_objective
