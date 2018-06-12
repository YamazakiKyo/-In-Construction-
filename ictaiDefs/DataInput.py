import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks, NearMiss
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
import math

def smote(data, label):
    # data, label = nearmiss(data, label)
    sm = SMOTE(ratio='minority',
               k_neighbors=5,
               m_neighbors=10,
               kind='borderline2',
               random_state=42,
               n_jobs=6)
    data_resampled, label_resampled = sm.fit_sample(data, label)
    # data_resampled, label_resampled = data, label
    return data_resampled, label_resampled

def nearmiss(data, label):
    n_pos_label_0 = data[label==0, :].shape[0]
    # n_neg_label_1 = data[label==1, :].shape[0]
    n_neg_kep = 3 * n_pos_label_0
    dict = {0 : n_pos_label_0, 1 : n_neg_kep}
    nm = NearMiss(ratio = dict,
                  version = 2,
                  random_state=42,
                  n_jobs=6,
                  n_neighbors=5)
    data_resampled, label_resampled = nm.fit_sample(data, label)
    return data_resampled, label_resampled
    # return data, label

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
    :param 0: Fatal | 1: severe Injury | 3 : Complain Injury | 4: PDO | 5: Noise
    :return: training set | test set
    '''
    X_train = pd.read_csv('./data_encoded/'+str(year)+'_data.csv')
    X_train = X_train.as_matrix().astype(float)
    y_train = pd.read_csv('./data_encoded/'+str(year)+'_label.csv')
    if n_classes == 2:
        y_train = y_train.replace([1, 2], 0)
        y_train = y_train.replace([3, 4, 5], 1).values.flatten()
        # y_train = y_train.replace(1, 0)
        # y_train = y_train.replace([2, 3, 4, 5], 1).values.flatten()
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
    X.to_csv('X_2.csv', index=False)
    y.to_csv('y_2.csv', index=False)

def count_value(nparr):
    unique_elements, counts_elements = np.unique(nparr, return_counts=True)
    return np.asarray((unique_elements, counts_elements))

def DE_adjust(data):
    new_data = np.zeros(shape=data.shape)
    tau1, tau2, tau3, tau4 = 0.1, 0.1, 0.03, 0.07
    SFGSS, SFHC, Fl, Fu = 8, 20, 0.1, 0.9
    KK = np.random.rand()
    Fi = np.random.rand(1, 5).flatten()
    # print('random trigger: %s' %Fi)
    if Fi[4] < tau3:
        Fi = SFGSS
    elif tau3 <= Fi[4] and Fi[4] < tau4:
        Fi = SFHC
    elif Fi[1] < tau1:
        Fi = Fl + Fu * Fi[0]
    else:
        Fi = np.random.rand()
    # print('data size: %.1f' %(data.shape[0]))
    for i in range(data.shape[0]):
        TR1 = data[np.random.randint(data.shape[0]-1), :].reshape(1, -1)
        TR2 = data[np.random.randint(data.shape[0]-1), :].reshape(1, -1)
        TR3 = data[np.random.randint(data.shape[0]-1), :].reshape(1, -1)
        new_data[i, :] = data[i, :] + KK * (TR1-data[i, :]) + Fi * (TR2-TR3)
        # print('Scaning row: %.1f ' %i)
    # print('TR1: %s, TR2: %s, TR3: %s' %(TR1, TR2, TR3))
    print('New DE_adjusted: %s' %(new_data[0, 0: 3]))
    return new_data

def DE_synthetic(data, label, batch_size, evo_round, super=None, req=0):
    clf = KNeighborsClassifier()
    n_window = math.ceil(data.shape[0]/batch_size)
    new_data = data[0, :].reshape(1, data.shape[1])
    new_label = label[0].reshape(1, )
    data, data_test, label, label_test = train_test_split(data, label, test_size=0.2, random_state=42)
    for i in range(n_window):
        print('%d th window: ' %i)
        X_bat, y_bat = batch_feed(i, data, label, batch_size)
        X, y = nearmiss(X_bat, y_bat)
        X_resampled, y_resampled = smote(X, y)
        # print('combined_syn size: %d' %(X_resampled.shape[0]))
        X_new = X_resampled[(X.shape[0]): (X_resampled.shape[0]), :]
        y_new = y_resampled[(y.shape[0]): (y_resampled.shape[0])]
        clf.fit(X_resampled, y_resampled)
        if super == 'f1':
            accuracy_real = f1_score(label_test, clf.predict(data_test), pos_label=0)
        elif super == 'recall':
            accuracy_real = recall_score(label_test, clf.predict(data_test), pos_label=0)
        else:
            accuracy_real = accuracy_score(label_test, clf.predict(data_test))
        print('benchmark: %.2f' %accuracy_real)
        X_DE = DE_adjust(X_new)
        # print('ready to evo: %d' %(X_DE.shape[0]))
        X_resampled_DE = np.append(X, X_DE, axis=0)
        # print('1st evo finished: %d' %(X_resampled_DE.shape[0]))
        y_resampled_DE = np.append(y, y_new, axis=0)
        clf.fit(X_resampled_DE, y_resampled_DE)
        if super == 'f1':
            accuracy_DE = f1_score(label_test, clf.predict(data_test), pos_label=0)
        elif super == 'recall':
            accuracy_DE = recall_score(label_test, clf.predict(data_test), pos_label=0)
        else:
            accuracy_DE = accuracy_score(label_test, clf.predict(data_test))
        count = 0
        print('evolution round: %d, DE_adjusted: %.2f' %(count, accuracy_DE))
        ###################################################################
        while (accuracy_DE <= accuracy_real + req) and (count < evo_round):
            count += 1
            X_DE = DE_adjust(X_DE) #according to the last generation
            X_resampled_DE = np.append(X, X_DE, axis=0)
            clf.fit(X_resampled_DE, y_resampled_DE)
            if super == 'f1':
                accuracy_DE = f1_score(label_test, clf.predict(data_test), pos_label=0)
            elif super == 'recall':
                accuracy_DE = recall_score(label_test, clf.predict(data_test), pos_label=0)
            else:
                accuracy_DE = accuracy_score(label_test, clf.predict(data_test))
            print('evolution round: %d, DE_adjusted: %.2f' %(count, accuracy_DE))
        if accuracy_DE <= accuracy_real:
            X_resampled_DE = X_resampled
            y_resampled_DE = y_resampled
        new_data = np.append(new_data, X_resampled_DE, axis=0)
        new_label = np.append(new_label, y_resampled_DE, axis=0)
    new_data = np.append(new_data, data_test, axis=0)
    new_label = np.append(new_label, label_test, axis=0)
    return new_data, new_label

if __name__ == "__main__":
    X = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_data.csv')
    X = X.as_matrix().astype(float)
    y = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_label.csv')
    y = y.replace([1, 2], 0)
    y = y.replace([3, 4, 5], 1).values.flatten()

    X, y = smote(X, y)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.to_csv('X_smote.csv', index=False)
    y.to_csv('y_smote.csv', index=False)

    X = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_data.csv')
    X = X.as_matrix().astype(float)
    y = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_label.csv')
    y = y.replace([1, 2], 0)
    y = y.replace([3, 4, 5], 1).values.flatten()

    X, y = nearmiss(X, y)
    X, y = smote(X, y)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.to_csv('X_combo.csv', index=False)
    y.to_csv('y_combo.csv', index=False)

    X = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_data.csv')
    X = X.as_matrix().astype(float)
    y = pd.read_csv('D:\PyProject\Distraction_Affected_Crashes\data_encoded/2014_label.csv')
    y = y.replace([1, 2], 0)
    y = y.replace([3, 4, 5], 1).values.flatten()

    X, y = DE_synthetic(X, y, X.shape[0], 20)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.to_csv('X_DEsyn.csv', index=False)
    y.to_csv('y_DEsyn.csv', index=False)





