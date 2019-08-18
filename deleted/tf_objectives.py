import tensorflow as tf
import numpy as np


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
        H1 = y_pred[:, 0:o1].T
        H2 = y_pred[:, o1:o1+o2].T

        m = H1.shape[1]

        H1bar = H1 - (1.0 / m) * np.dot(H1, np.ones([m, m]))
        H2bar = H2 - (1.0 / m) * np.dot(H2, np.ones([m, m]))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar, H1bar.T) + r1 * np.eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar, H2bar.T) + r2 * np.eye(o2)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = np.nonzero(D1 > eps)[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = np.nonzero(D2 > eps)[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = np.dot(
            np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(
            np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = np.sqrt(np.matrix.trace(np.dot(Tval.T, Tval)))
        else:
            # just the top outdim_size singular values are used
            [U, V] = np.linalg.eigh(np.dot(Tval.T, Tval))
            U = U[np.gt(U, eps).nonzero()[0]]
            U = U.sort()
            corr = np.sum(np.sqrt(U[0:outdim_size]))

        return -corr

    return inner_cca_objective
