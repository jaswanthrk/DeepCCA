import random
import pickle
import numpy
import gzip
import _pickle
from PIL import Image
import numpy as np
import cv2
import random

# Load the dataset

f = gzip.open('mnist.pkl.gz', 'rb')
f.seek(0)
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

data = [train_set, valid_set, test_set]
print('data loaded')

data1_trv = []
data1_trl = []
data2_trv = []
data2_trl = []
for i in range(10):
    print('train : ' + str(i))
    idx = np.squeeze(np.where(train_set[1] == i))
    train_i_1 = train_set[0][idx]
    data1_trv.append(train_i_1)
    data1_trl.append(i * np.squeeze(np.ones((1, len(train_i_1)))))
    train_i_2 = train_set[0][idx]
    view2 = []
    for cnt, j in enumerate(train_i_1):
        if cnt % 300 == 0:
            print('i is ' + str(i) + ' j is : ' + str(cnt))
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
    data2_trv.append(view2)
    data2_trl.append(i * np.squeeze(np.ones((1, len(train_i_1)))))


data1_valv = []
data1_vall = []
data2_valv = []
data2_vall = []
for i in range(10):
    print('val : ' + str(i))
    idx = np.squeeze(np.where(valid_set[1] == i))
    val_i_1 = valid_set[0][idx]
    data1_valv.append(val_i_1)
    data1_vall.append(i * np.squeeze(np.ones((1, len(val_i_1)))))
    val_i_2 = valid_set[0][idx]
    view2 = []
    for cnt, j in enumerate(val_i_1):
        if cnt % 300 == 0:
            print('i is ' + str(i) + ' j is : ' + str(cnt))
        x = random.choice(val_i_1)
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
    data2_valv.append(view2)
    data2_vall.append(i * np.squeeze(np.ones((1, len(val_i_1)))))


data1_tev = []
data1_tel = []
data2_tev = []
data2_tel = []
for i in range(10):
    print('test : ' + str(i))
    idx = np.squeeze(np.where(test_set[1] == i))
    test_i_1 = test_set[0][idx]
    data1_tev.append(test_i_1)
    data1_tel.append(i * np.squeeze(np.ones((1, len(test_i_1)))))
    test_i_2 = test_set[0][idx]
    view2 = []
    for cnt, j in enumerate(test_i_1):
        if cnt % 300 == 0:
            print('i is ' + str(i) + ' j is : ' + str(cnt))
        x = random.choice(test_i_1)
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
    data2_tev.append(view2)
    data2_tel.append(i * np.squeeze(np.ones((1, len(test_i_1)))))

data1_trv = np.concatenate(data1_trv , axis=0)
data1_trl = np.concatenate(data1_trl , axis=0)
data1_valv = np.concatenate(data1_valv , axis=0)
data1_vall = np.concatenate(data1_vall , axis=0)
data1_tev = np.concatenate(data1_tev , axis=0)
data1_tel = np.concatenate(data1_tel , axis=0)

data2_trv = np.concatenate(data2_trv , axis=0)
data2_trl = np.concatenate(data2_trl , axis=0)
data2_valv = np.concatenate(data2_valv , axis=0)
data2_vall = np.concatenate(data2_vall , axis=0)
data2_tev = np.concatenate(data2_tev , axis=0)
data2_tel = np.concatenate(data2_tel , axis=0)

print(data1_trl == data2_trl)
combined = list(zip(data1_trv,data1_trl,data2_trv,data2_trl))
random.shuffle(combined)
data1_trv,data1_trl,data2_trv,data2_trl = zip(*combined)
print(data1_trl == data2_trl)

print(data1_vall == data2_vall)
combined = list(zip(data1_valv,data1_vall,data2_valv,data2_vall))
random.shuffle(combined)
data1_valv,data1_vall,data2_valv,data2_vall = zip(*combined)
print(data1_vall == data2_vall)

print(data1_tel == data2_tel)
combined = list(zip(data1_tev,data1_tel,data2_tev,data2_tel))
random.shuffle(combined)
data1_tev,data1_tel,data2_tev,data2_tel = zip(*combined)
print(data1_tel == data2_tel)

data1_tr = [np.array(data1_trv), np.array(data1_trl)]
data1_val = [np.array(data1_valv), np.array(data1_vall)]
data1_te = [np.array(data1_tev), np.array(data1_tel)]

data2_tr = [np.array(data2_trv), np.array(data2_trl)]
data2_val = [np.array(data2_valv), np.array(data2_vall)]
data2_te = [np.array(data2_tev), np.array(data2_tel)]

data1 = [data1_tr, data1_val, data1_te]
data2 = [data2_tr, data2_val, data2_te]

with open("data1_15.pkl", "wb") as f:  # Pickling
    pickle.dump(data1, f)

with open("data2_15.pkl", "wb") as f:  # Pickling
    pickle.dump(data2, f)

with open("data1_15.pkl", "rb") as fp:   # Unpicklin
    a = pickle.load(fp)

print(len(b))
print(len(b[0]))
print(len(b[0][0]))
print(len(b[0][1]))
print(len(b[1]))
print(len(b[1][0]))
print(len(b[1][1]))
print(len(b[2]))
print(len(b[2][0]))
print(len(b[2][1]))
