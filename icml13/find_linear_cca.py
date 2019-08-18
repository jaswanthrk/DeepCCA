import numpy as np
from linear_cca import linear_cca
from utilities import load_data
from sklearn import svm
from sklearn.metrics import accuracy_score


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, train_label = data[0]
    valid_data, valid_label = data[1]
    test_data, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')

dim = 50

A, B, M1, M2, D1 = linear_cca(data1[1][0], data2[1][0], dim)

new_tr1 = [np.dot(data1[0][0], A), data1[0][1]]
new_val1 = [np.dot(data1[1][0], A), data1[1][1]]
new_te1 = [np.dot(data1[2][0], A), data1[2][1]]
new_data1 = [new_tr1, new_val1, new_te1]

print(sum(np.abs(D1)))

prev_tr1 = [data1[0][0], data1[0][1]]
prev_val1 = [data1[1][0], data1[1][1]]
prev_te1 = [data1[2][0], data1[2][1]]
prev_data1 = [prev_tr1, prev_val1, prev_te1]

[test_acc0, valid_acc0] = svm_classify(prev_data1, C=0.01)
print("Accuracy on prev view 1 (validation data) is:", valid_acc0 * 100.0)
print("Accuracy on prev view 1 (test data) is:", test_acc0 * 100.0)

[test_acc, valid_acc] = svm_classify(new_data1, C=0.01)
print("Accuracy on new view 1 (validation data) is:", valid_acc * 100.0)
print("Accuracy on new view 1 (test data) is:", test_acc*100.0)
