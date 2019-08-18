import numpy as np

import pickle

'''
function[W1, b1, W2, b2, P1, P2, m1, m2, D, randseed] = randKCCA(
    X1, X2, K, M1, s1, rcov1, M2, s2, rcov2, randseed)
% [W1, b1, W2, b2, P1, P2, m1, m2, D, randseed] = randKCCA(...
                                                           % X1, X2, K, M1, s1, rcov1, M2, s2, rcov2, randseed) trains the randomized
%     KCCA model of the following paper
%     David Lopez-Paz, Suvrit Sra, Alex Smola, Zoubin Ghahramani and
%     Bernhard Schoelkopf. "Randomized Nonlinear Component Analysis".
%     ICML 2014.
%
% Inputs
%   X1/X2: training data for view 1/view 2, containing samples rowwise.
%   K: dimensionality of CCA projection.
%   M1/M2: the number of random samples for each view.
%   s1/s2: Gaussian kernel width s for each view for the kernel function
%     k(x, y) = exp(-0.5*|x-y | ^2/s ^ 2) .
%   rcov1/rcov2: regularization parameter for each view.
%   randseed: random seed for random Fourier features.
%
% Outputs
%   W1/W2: random weights for view 1/view 2.
%   b1/b2: random bias for view 1/view 2.
%   P1/P2: the CCA projection matrix for view 1/view 2.
%   m1/m2: feature mean for view 1/view 2.
%   D: vector of canonical correlations for each of the K dimensions. '''

with open('data1.pkl', 'rb') as f:
    data1 = pickle.load(f)

with open('data2.pkl', 'rb') as f:
    data2 = pickle.load(f)

X1 = data1[0][0]
X2 = data2[0][0]
M1 = 10000
M2 = 10000
s1 = 16
s2 = 16
rcov1 = 1e-4
rcov2 = 1e-4
eps = 1e-5
K = 50

# % Generate random features.
sh1 = X1.shape
D1 = sh1[1]
W1 = np.random.randn(D1, M1) / s1
b1 = np.random.rand(1, M1) * 2 * np.pi
sh2 = X2.shape
D2 = sh1[1]
W2 = np.random.randn(D2, M2) / s2
b2 = np.random.rand(1, M2) * 2 * np.pi

# % Compute things by blocks in case large M1/M2 is used.
N = X1.shape[0]
print('N is : ' + str(N))
T = 6000
NUMBATCHES = N // T
print('NUMBATCHES is : ' + str(NUMBATCHES))

# % Estimate mean.
print('Computing Means')
m1 = np.zeros((1, M1))
m2 = np.zeros((1, M2))
for j in range(NUMBATCHES):
    if j % 100 == 0:
        print('j is : ' + str(j))
    batchidx = range(T*(j-1)+1, min(N, T*j))
    FEA1 = np.cos(np.dot(X1[batchidx, :], W1) + b1)
    FEA2 = np.cos(np.dot(X2[batchidx, :], W2) + b2)
    m1 = m1 + np.sum(FEA1, axis=0)
    m2 = m2 + np.sum(FEA2, axis=0)


m1 = m1/N
m2 = m2/N

# % Compute covariance.
print('Computing Covariances')
S11 = np.zeros((M1, M1))
S22 = np.zeros((M2, M2))
S12 = np.zeros((M1, M2))
for j in range(NUMBATCHES):
    if j % 100 == 0:
        print('j is : ' + str(j))
    batchidx = range(T*(j-1)+1, min(N, T*j))
    FEA1 = np.cos(np.dot(X1[batchidx, :], W1) + b1)
    FEA1 = FEA1 - m1
    FEA2 = np.cos(np.dot(X2[batchidx, :], W2) + b2)
    FEA2 = FEA2 - m2
    S11 = S11 + np.dot(FEA1.transpose(), FEA1)
    S22 = S22 + np.dot(FEA2.transpose(), FEA2)
    S12 = S12 + np.dot(FEA1.transpose(), FEA2)


S11 = S11/(N-1)
S12 = S12/(N-1)
S22 = S22/(N-1)

# % Add regularization.
print('Regularizing..')
S11 = S11 + np.dot(rcov1, np.eye(M1))
S22 = S22 + np.dot(rcov2, np.eye(M2))
print('Regularized')

print('Finding eigenvectors and eigenvalues..')
[D1, V1] = np.linalg.eigh(S11)
[D2, V2] = np.linalg.eigh(S22)
print('FOUND !')

# % For numerical stability.
print('Handpicking your eigens..')
idx1 = np.squeeze(np.where(abs(D1) > eps))
D1 = D1[idx1]
V1 = V1[:, idx1]

idx2 = np.squeeze(np.where(abs(D2) > eps))
D2 = D2[idx2]
V2 = V2[:, idx2]
print('PICKED !')

print('Finding T')
K11 = np.dot(V1, np.dot(np.diag(D1 ** -0.5), V1.transpose()))
K22 = np.dot(V2, np.dot(np.diag(D2 ** -0.5), V2.transpose()))
T = np.dot(np.dot(K11, S12), K22)
print('FOUND !')

[U, D, V] = np.linalg.svd(T)
P1 = np.dot(K11, U[:, 0:K])
P2 = np.dot(K22, V[:, 0:K])
np.save(D, D)
D = D[0:K]

corr = np.sum(np.sum(D))
np.save(corrs_rand, corr)

print('corr is : ' + str(D))
