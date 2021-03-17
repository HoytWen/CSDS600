from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np
from sklearn.model_selection import StratifiedKFold


datapath = 'DogsVsCats/DogsVsCats/'
y, x = svm_read_problem(f'{datapath}DogsVsCats.train')
y_test, x_test = svm_read_problem(f'{datapath}DogsVsCats.test')

svm_param1 = '-t 0'
svm_param2 = '-t 1 -d 5'

print(list(x[0].values()))


def data_index(data, label, ind):
    data_subset = []
    label_subset = []
    for x in ind:
        data_subset.append(data[x])
        label_subset.append(label[x])

    return data_subset, label_subset

def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        miss = [int(x) for x in (pred_train_i != Y_train)]
        miss2 = [x if x == 1 else -1 for x in miss]
        err_m = np.dot(w, miss) / sum(w)
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


skf = StratifiedKFold(n_splits=10)
train_fold = []
val_fold = []


for train, val in skf.split(x, y):
    train_fold.append(train)
    val_fold.append(val)

accu = []
for i in range(len(train_fold)):

    train_ind = train_fold[i]
    val_ind = val_fold[i]

    x_train, y_train = data_index(x, y, train_ind)
    x_val, y_val = data_index(x, y, val_ind)

    print(f'{i+1}th training start...')
    model = svm_train(y_train, x_train, svm_param2)
    print(f'the validation result of {i+1} subset')
    p_label, p_acc, p_val = svm_predict(y_val, x_val, model)
    accu.append(p_acc[0])

print(accu)
print(np.mean(accu))



# model = svm_train(y, x, svm_param1)
# p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
