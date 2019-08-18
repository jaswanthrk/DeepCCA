import pickle
import numpy
import gzip
import _pickle
from PIL import Image
import numpy as np
from utilities import load_data

# Load the dataset

f = gzip.open('mnist.pkl.gz', 'rb')
f.seek(0)
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

data = [train_set, valid_set, test_set]

a = data[0][0][0]
sh = a.shape
b = np.reshape(a, (28, -1))
img = Image.fromarray(np.uint8(b * 255), 'L')
img.show()

a1 = a[0:sh[0]//2]
a1 = np.reshape(a1, (28, -1))
img1 = Image.fromarray(np.uint8(a1 * 255), 'L')
img1.show()
a2 = a[sh[0]//2:]
a2 = np.reshape(a2, (28, -1))
img2 = Image.fromarray(np.uint8(a2 * 255), 'L')
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
