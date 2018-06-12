from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from imblearn.over_sampling import ADASYN


import ictaiDefs.DataInput as iD

def make_clf(X, y, X_test, clf=None):
    if clf == 'MLP':
        clf = MLPClassifier(alpha=1)
    if clf == 'KNN':
        clf = KNeighborsClassifier(3)
    if clf == 'SVM':
        clf = SVC(kernel="linear", C=0.025)
    if clf == 'SVN-nonLinear':
        clf = SVC(gamma=2, C=1)
    if clf == 'RF':
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    if clf == 'C4.5':
        clf = DecisionTreeClassifier(max_depth=5)
    if clf == 'AdaBoost':
        clf = AdaBoostClassifier()
    if clf == 'NB':
        clf = GaussianNB()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    return y_pred

X, y = iD.input_data('2014', n_classes=2)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# X_ADASYN = X
# y_ADASYN = y



# clfs1 = ['C4.5', 'MLP', 'KNN']
clfs1 = ['MLP']
clfs2 = ['SVM', 'SVN-nonLinear']
clfs3 = ['AdaBoost', 'NB', 'RF']

# ad = ADASYN(n_jobs=6, random_state=42)
# X_ADASYN, y_ADASYN = ad.fit_sample(X, y)
# X_ADASYN, y_ADASYN = shuffle(X_ADASYN, y_ADASYN)
# X_NM, y_NM = iD.nearmiss(X, y)
# X_NM, y_NM = shuffle(X_NM, y_NM)
# X_SMOTE, y_SMOTE = iD.smote(X, y)
# X_SMOTE, y_SMOTE = shuffle(X_SMOTE, y_SMOTE)
# X_NM_SM, y_NM_SM = iD.smote(X_NM, y_NM)
# X_NM_SM, y_NM_SM = shuffle(X_NM_SM, y_NM_SM)
# X_DE, y_DE = iD.DE_synthetic(X, y, int(X.shape[0] / 12), 20)
# X_DE, y_DE = shuffle(X_DE, y_DE)
# X_DE_recall, y_DE_recall = iD.DE_synthetic(X, y, int(X.shape[0] / 12), 20, super='recall')
# X_DE_recall, y_DE_recall = shuffle(X_DE_recall, y_DE_recall)

for clf in clfs1:
    print(clf)

    y_ADASYN_pred = make_clf(X_ADASYN, y_ADASYN, X_test, clf=clf)
    print('ADASYN')
    print('Accuracy: ', accuracy_score(y_test, y_ADASYN_pred))
    print('Recall: ', recall_score(y_test, y_ADASYN_pred, pos_label=0))
    print('Precision: ', precision_score(y_test, y_ADASYN_pred, pos_label=0))
    print('True Negative:', recall_score(y_test, y_ADASYN_pred, pos_label=1))
    print('F1: ', f1_score(y_test, y_ADASYN_pred, pos_label=0))
    print('F_Beta: ', fbeta_score(y_test, y_ADASYN_pred, pos_label=0, beta= 10))
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_ADASYN_pred))


    # y_NM_pred = make_clf(X_NM, y_NM, X_test, clf=clf)
    # print('NearMiss')
    # print('Accuracy: ', accuracy_score(y_test, y_NM_pred))
    # print('Recall: ', recall_score(y_test, y_NM_pred, pos_label=0))
    # print('Precision: ', precision_score(y_test, y_NM_pred, pos_label=0))
    # print('True Negative:', recall_score(y_test, y_NM_pred, pos_label=1))
    # print('F1: ', f1_score(y_test, y_NM_pred, pos_label=0))
    # print('F_Beta: ', fbeta_score(y_test, y_NM_pred, pos_label=0, beta= 10))
    # print('Confusion Matrix: ')
    # print(confusion_matrix(y_test, y_NM_pred))
    #
    # y_SMOTE_pred = make_clf(X_SMOTE, y_SMOTE, X_test, clf=clf)
    # print('SMOTE')
    # print('Accuracy: ', accuracy_score(y_test, y_SMOTE_pred))
    # print('Recall: ', recall_score(y_test, y_SMOTE_pred, pos_label=0))
    # print('Precision: ', precision_score(y_test, y_SMOTE_pred, pos_label=0))
    # print('True Negative:', recall_score(y_test, y_SMOTE_pred, pos_label=1))
    # print('F1: ', f1_score(y_test, y_SMOTE_pred, pos_label=0))
    # print('F_Beta: ', fbeta_score(y_test, y_SMOTE_pred, pos_label=0, beta=10))
    # print('Matrix: ')
    # print(confusion_matrix(y_test, y_SMOTE_pred))
    #
    # y_NM_SM_pred = make_clf(X_NM_SM, y_NM_SM, X_test, clf=clf)
    # print('NearMiss + SMOTE')
    # print('Accuracy: ', accuracy_score(y_test, y_NM_SM_pred))
    # print('Recall: ', recall_score(y_test, y_NM_SM_pred, pos_label=0))
    # print('Precision: ', precision_score(y_test, y_NM_SM_pred, pos_label=0))
    # print('True Negative:', recall_score(y_test, y_NM_SM_pred, pos_label=1))
    # print('F1: ', f1_score(y_test, y_NM_SM_pred, pos_label=0))
    # print('F_Beta: ', fbeta_score(y_test, y_NM_SM_pred, pos_label=0, beta=10))
    # print('Confusion Matrix: ')
    # print(confusion_matrix(y_test, y_NM_SM_pred))
    #
    # y_DE_pred = make_clf(X_DE, y_DE, X_test, clf=clf)
    # print('DE')
    # print('Accuracy: ', accuracy_score(y_test, y_DE_pred))
    # print('Recall: ', recall_score(y_test, y_DE_pred, pos_label=0))
    # print('Precision: ', precision_score(y_test, y_DE_pred, pos_label=0))
    # print('True Negative:', recall_score(y_test, y_DE_pred, pos_label=1))
    # print('F1: ', f1_score(y_test, y_DE_pred, pos_label=0))
    # print('F_Beta: ', fbeta_score(y_test, y_DE_pred, pos_label=0, beta=10))
    # print('Confusion Matrix: ')
    # print(confusion_matrix(y_test, y_DE_pred))
    #
    # y_DE_recall_pred = make_clf(X_DE_recall, y_DE_recall, X_test, clf=clf)
    # print('DE recall-supervised')
    # print('Accuracy: ', accuracy_score(y_test, y_DE_recall_pred))
    # print('Recall: ', recall_score(y_test, y_DE_recall_pred, pos_label=0))
    # print('Precision: ', precision_score(y_test, y_DE_recall_pred, pos_label=0))
    # print('True Negative:', recall_score(y_test, y_DE_recall_pred, pos_label=1))
    # print('F1: ', f1_score(y_test, y_DE_recall_pred, pos_label=0))
    # print('F_Beta: ', fbeta_score(y_test, y_DE_recall_pred, pos_label=0, beta=10))
    # print('Confusion Matrix: ')
    # print(confusion_matrix(y_test, y_DE_recall_pred))