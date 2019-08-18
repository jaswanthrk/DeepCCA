import numpy as np
import pickle

with open('data1.pkl', 'rb') as f:
    data1 = pickle.load(f)

print('loaded data1')

with open('data2.pkl', 'rb') as f:
    data2 = pickle.load(f)

print('loaded data2')

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

print('Got X1, X2, M1, M2, s1, s2, r1, r2, eps, K ')

# % Generate random features.
sh1 = X1.shape
D1 = sh1[1]
sh2 = X2.shape
D2 = sh1[1]

'''
print("K1 started : ")
K1 = []
for i, rowi in enumerate(X1):
    if i % 200 == 0:
        print(i)
    temp = np.reshape(rowi, (1, -1))
    if i != 0:
        temp = np.concatenate([
            np.zeros((i, D1)), np.repeat(temp, len(X1)-i, axis=0)], axis=0)
    else:
        temp = np.repeat(temp, len(X1), axis=0)
    diff = (temp - X1)
    diff = diff[i:]
    diff = np.square(np.linalg.norm(diff, axis=1))
    k1i = np.exp(-diff / (2 * np.square(s1)))
    print(len(k1i))
    K1.append(k1i)

sums_s = []
final = 0
for i in K1:
    sumi = np.sum(i)
    sums_s.append(sumi)
    final = final + sumi

K1_info = [K1, sums_s, final]

with open('K1_info.pkl', 'wb') as f:
    pickle.dump(K1_info, f)
print('K1 saved!')
'''

print("K2 started : ")
K2 = []
for i, rowi in enumerate(X2):
    if i % 200 == 0:
        print(i)
    temp = np.reshape(rowi, (1, -1))
    if i != 0:
        temp = np.concatenate([
            np.zeros((i, D2)), np.repeat(temp, len(X2)-i, axis=0)], axis=0)
    else:
        temp = np.repeat(temp, len(X2), axis=0)
    diff = (temp - X2)
    diff = diff[i:]
    diff = np.square(np.linalg.norm(diff, axis=1))
    k2i = np.exp(-diff / (2 * np.square(s2)))
    print('current k2i length is' + str(len(k2i)))
    K2.append(k2i)
    print('K2 length is ' + str(len(K2)))

sums_s = []
final = 0
for i in K2:
    sumi = np.sum(i)
    sums_s.append(sumi)
    final = final + sumi

K2_info = [K2, sums_s, final]

with open('K2_info.pkl', 'wb') as f:
    pickle.dump(K2_info, f)
print('K2 saved!')

'''
ONE = np.eye(len(X2))
first = np.dot(ONE, K2)
sec = np.dot(first, ONE)
thir = sec - np.dot(ONE, K2)
K2 = K2 - np.dot(K2, ONE) + thir

np.save(K2_rbf, K2)
print('K2 saved!')

print('regularizing.. K1 and K2')
K11 = K1 + rcov1 * np.eye(len(K1))
K22 = K2 + rcov2 * np.eye(len(K2))
print('DONE')

print('INVERSION OF COV started..')
print('Finding eigenvectors and eigenvalues..')
[D1, V1] = np.linalg.eigh(K11)
[D2, V2] = np.linalg.eigh(K22)
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
print('INVERSE PROCESS ENDED.')

print('Finding T')
K11_inv = np.dot(V1, np.dot(np.diag(D1 ** -1), V1.transpose()))
K22_inv = np.dot(V2, np.dot(np.diag(D2 ** -1), V2.transpose()))
T = np.dot(np.dot(np.dot(K11_inv, K22), K22_inv), K11)
print('DONE')

print('Doing SVD..')
[U, D, V] = np.linalg.svd(T)
P1 = np.dot(K11_inv, U[:, 0:K])
P2 = np.dot(K22_inv, V[:, 0:K])
np.save(D_rbf, D)
D = D[0:K]
print('Got D and saved D')

corr = np.sum(np.sum(D))
np.save(corrs_rbf, corr)

print('corr is : ' + str(D))
'''
