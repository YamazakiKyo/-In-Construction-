import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def smote(data, label):
    sm = SMOTE(ratio='minority',
               k_neighbors=30,
               m_neighbors=13,
               kind='svm',
               random_state=42,
               n_jobs=6)
    data_resampled, label_resampled = sm.fit_sample(data, label)
    # data_resampled, label_resampled = data, label
    return data_resampled, label_resampled

def enn(data, label):
    # enn = EditedNearestNeighbours(ratio='majority', n_neighbors=30, kind_sel='mode', random_state=42)
    # data_resampled, label_resampled = enn.fit_sample(data, label)
    data_resampled, label_resampled = data, label
    return data_resampled, label_resampled

def smote_bi(data, label):
    ratio = {0: 250, 1: 450}
    sm = SMOTEENN(random_state=42,
                  smote=SMOTE(ratio='minority',k_neighbors=1, m_neighbors=10, kind='svm'),
                  enn=EditedNearestNeighbours(ratio='all', n_neighbors=3, kind_sel='all'))
    # sm = SMOTEENN()
    data_resampled, label_resampled = sm.fit_sample(data, label)
    return data_resampled, label_resampled

def smote_tri(data, label):
    ratio = {0: 45000, 1: 90000, 2: 111534}
    # ratio = {}
    sm = SMOTEENN(ratio='minority',
                  n_jobs=6,
                  random_state=0)
    # sm = SMOTE(ratio=ratio,
    #            n_jobs=6,
    #            kind='svm',
    #            random_state=42)
    data_resampled, label_resampled = sm.fit_sample(data, label)
    return data_resampled, label_resampled

def smote_tomek(data, label):
    # dict = {0: 50000, 1: 50000}
    sm = SMOTETomek(n_jobs=6,
                    random_state=0,
                    tomek=hyper_tomek,
                    smote=hyper_smote)
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
        X_train, y_train = smote(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
    if preprocessing == 'smote_3_class':
        X_train, y_train = smote_tri(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
    if preprocessing == 'smote_tomek':
        X_train, y_train = smote_tomek(X_train, y_train)
        X_train, y_train = shuffle(X_train, y_train)
    return X_train, y_train

def batch_feed(i, X, y, batch_size):
    offset = (i * batch_size) % (y.shape[0] - batch_size)
    X_batch = X[offset : (offset+batch_size), : ]
    y_batch = y[offset : (offset + batch_size)]
    return X_batch, y_batch

def output_data(X, y):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.to_csv('data_plot.csv', index=False)
    y.to_csv('label_plot.csv', index=False)

def count_value(nparr):
    unique_elements, counts_elements = np.unique(nparr, return_counts=True)
    return np.asarray((unique_elements, counts_elements))

def DE_adjust(data):
    tau1, tau2, tau3, tau4 = 0.1, 0.1, 0.03, 0.07
    SFGSS, SFHC, Fl, Fu = 8, 20, 0.1, 0.9
    KK = np.random.rand()
    Fi = np.random.rand(1, 5).flatten()
    if Fi[4] < tau3:
        Fi = SFGSS
    elif tau3 <= Fi[4] and Fi[4] < tau4:
        Fi = SFHC
    elif Fi[1] < tau1:
        Fi = Fl + Fu * Fi[0]
    else:
        Fi = np.random.rand()
    for i in range(data.shape[0]):
        TR1 = data[np.random.randint(data.shape[0]-1), :]
        TR2 = data[np.random.randint(data.shape[0]-1), :]
        TR3 = data[np.random.randint(data.shape[0]-1), :]
        data[i, :] = data[i, :] + KK * (TR1-data[i, :]) + Fi * (TR2-TR3)
    return data

