import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def smote_bi(data, label):
    sm = SMOTEENN(ratio='minority',
                  n_jobs=6,
                  random_state=42)
    data_resampled, label_resampled = sm.fit_sample(data, label)
    return data_resampled, label_resampled

def smote_tri(data, label):
    ratio = {0: 45000, 1: 90000, 2: 111534}
    sm = SMOTEENN(ratio=ratio,
                  n_jobs=6,
                  random_state=42)
    # sm = SMOTE(ratio=ratio,
    #            n_jobs=6,
    #            kind='svm',
    #            random_state=42)
    data_resampled, label_resampled = sm.fit_sample(data, label)
    return data_resampled, label_resampled


def input_data(year, n_classes=3, one_hot=False, preprocessing=None):
    '''
    :param 0: Fatal | 1 - 3 : Injury | 4: PDO | 5: Noise
    :return: training set | test set
    '''
    X_train = pd.read_csv('./data_encoded/'+str(year)+'_data.csv')
    X_train = X_train.as_matrix().astype(float)
    y_train = pd.read_csv('./data_encoded/'+str(year)+'_label.csv')
    if n_classes == 2:
        y_train = y_train.replace([1, 2, 3, 5], 0)
        y_train = y_train.replace(4, 1).values.flatten()
    if n_classes == 3:
        y_train = y_train.replace([1, 2, 3, 5], 1)
        y_train = y_train.replace(4, 2).values.flatten()
    if one_hot == True:
        one_hot = OneHotEncoder()
        one_hot.fit(y_train)
        y_train = one_hot.transform(y_train).toarray()
        y_train.flatten()
        y_train = np.reshape(y_train, (-1, y_train.shape[0]))
    if preprocessing == 'smote_2_class':
        X_train, y_train = smote_bi(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
    if preprocessing == 'smote_3_class':
        X_train, y_train = smote_tri(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
    return X_train, y_train

def batch_feed(i, X, y, batch_size):
    offset = (i * batch_size) % (y.shape[0] - batch_size)
    X_batch = X[offset : (offset+batch_size), : ]
    y_batch = y[offset : (offset + batch_size)]
    return X_batch, y_batch