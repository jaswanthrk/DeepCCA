import random
import pickle
import numpy
import gzip
import _pickle
from PIL import Image
import numpy as np
import cv2

# Load the dataset

f = gzip.open('mnist.pkl.gz', 'rb')
f.seek(0)
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

data = [train_set, valid_set, test_set]

data1 = []
data2 = []
for i in range(10):
    idx = np.squeeze(np.where(train_set[1] == i))
    train_i_1 = train_set[0][idx]
    data1.append(train_i_1)
    train_i_2 = train_set[0][idx]
    view2 = []
    for count,j in enumerate(train_i_1):
        if count % 100 == 0:
            print('i : ' +str(i) + 'j :' + str(count) )
        x = random.choice(train_i_1)
        b = np.reshape(x, (28, -1))
        sh = b.shape
        M = cv2.getRotationMatrix2D(
            (sh[1] // 2, sh[0] // 2), 45 * np.random.rand(1), 1)
        res = cv2.warpAffine(b, M, (sh[1], sh[0]))
        res = res + np.random.rand(28, 28)*0.4
        np.clip(res, 0, 1)
        new = np.reshape(res, (1, 784))
        if view2 == []:
            view2 = new
        else:
            view2 = np.vstack([view2, new])
    data2.append(view2)

with open("data1.pkl", "wb") as f:  # Pickling
    pickle.dump(data1, f)

with open("data2.pkl", "wb") as f:  # Pickling
    pickle.dump(data2, f)


'''
for datas in data2:
    for x in datas:
        b = np.reshape(x, (28, -1))
        img1 = Image.fromarray(np.uint8(b * 255), 'L')
        img1.show()


a = data[0][0][0]
b = np.reshape(a, (28, -1))
img1 = Image.fromarray(np.uint8(b * 255), 'L')
img1.show()
sh = b.shape
M = cv2.getRotationMatrix2D(
    (sh[1] // 2, sh[0] // 2), 45 * np.random.rand(1), 1)
res = cv2.warpAffine(b, M, (sh[1], sh[0]))
img2 = Image.fromarray(np.uint8(res * 255), 'L')
img2.show()


l_train = [data[0][0][:, :sh[0]//2], data[0][1]]
l_val = [data[1][0][:, :sh[0]//2], data[1][1]]
l_test = [data[2][0][:, :sh[0]//2], data[2][1]]

data1 = [l_train, l_val, l_test]

r_train = [data[0][0][:, sh[0]//2:], data[0][1]]
r_val = [data[1][0][:, sh[0]//2:], data[1][1]]
r_test = [data[2][0][:, sh[0]//2:], data[2][1]]

data2 = [r_train, r_val, r_test]

with open('data1.pkl', 'wb') as f:
    pickle.dump(data1, f)

print('data1 saved.')

with open('data2.pkl', 'wb') as f:
    pickle.dump(data2, f)

print('data2 saved.')

'''
with open("data1.pkl", "rb") as fp:   # Unpicklin
    b = pickle.load(fp)
