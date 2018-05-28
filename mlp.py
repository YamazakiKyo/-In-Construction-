import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ictaiDefs.DataInput import input_data
from ictaiDefs.DataInput import batch_feed
from ictaiDefs.DataInput import output_data
import ictaiDefs.DataInput as iD


'''
1. Good explain about CrossEntropyLoss in Pytorch: http://sshuair.com/2017/10/21/pytorch-loss/ 
2. 
'''

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
tFloat = torch.cuda.FloatTensor
tLong = torch.cuda.LongTensor
# torch.cuda.manual_seed(42)

batch_size = 1000
iter = 20000

LR = 0.0001
N_INPUT = 29
N_H1 = 35
N_H2 = 20
N_H3 = 10
N_OUTPUT = 2

BIAS = -0.2
BN_MOMEN = 0.5
WC = torch.tensor([1, 1]).cuda().type(tFloat)

ACT1 = F.relu
ACT2 = F.relu



class MLPnet(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_output):
        super(MLPnet, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        nn.init.orthogonal(self.h1.weight)
        self.h2 = nn.Linear(n_h1, n_h2)
        nn.init.orthogonal(self.h2.weight)
        self.h3 = nn.Linear(n_h2, n_h3)
        nn.init.orthogonal(self.h3.weight)
        self.output = nn.Linear(n_h3, n_output)
        nn.init.orthogonal(self.output.weight)

    def forward(self, X):
        X = F.relu(self.h1(X))
        X = F.relu(self.h2(X))
        X = F.relu(self.h3(X))
        X = self.output(X)
        return X

''' 6-layer version '''
class MLPnet_deeper(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_h4,n_h5, n_h6, n_output):
        super(MLPnet_deeper, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        nn.init.orthogonal(self.h1.weight)
        self.h2 = nn.Linear(n_h1, n_h2)
        nn.init.orthogonal(self.h2.weight)
        self.h3 = nn.Linear(n_h2, n_h3)
        nn.init.orthogonal(self.h3.weight)
        self.h4 = nn.Linear(n_h3, n_h4)
        nn.init.orthogonal(self.h4.weight)
        self.h5 = nn.Linear(n_h4, n_h5)
        nn.init.orthogonal(self.h5.weight)
        self.h6 = nn.Linear(n_h5, n_h6)
        nn.init.orthogonal(self.h6.weight)
        self.output = nn.Linear(n_h6, n_output)
        nn.init.orthogonal(self.output.weight)

    def forward(self, X):
        # X = F.relu(self.h1(X))
        X = F.leaky_relu(self.h1(X), negative_slope=0.0001)
        # X = F.relu(self.h2(X))
        X = F.leaky_relu(self.h2(X), negative_slope=0.0001)
        # X = F.relu(self.h3(X))
        X = F.leaky_relu(self.h3(X), negative_slope=0.0001)
        X = F.dropout(X, p=0.2, training=self.training)
        X = F.relu(self.h4(X))
        # X = F.leaky_relu(self.h4(X), negative_slope=0.0001)
        X = F.relu(self.h5(X))
        # X = F.leaky_relu(self.h5(X), negative_slope=0.0001)
        X = F.relu(self.h6(X))
        # X = F.leaky_relu(self.h6(X), negative_slope=0.0001)
        X = F.dropout(X, p=0.2, training=self.training)
        X = self.output(X)
        return X

''' deeperer version '''
class MLPnet_deeperer(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_h4,n_h5, n_h6,n_h7,n_h8, n_h9, n_output):
        super(MLPnet_deeperer, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        nn.init.orthogonal(self.h1.weight)
        self.h2 = nn.Linear(n_h1, n_h2)
        nn.init.orthogonal(self.h2.weight)
        self.h3 = nn.Linear(n_h2, n_h3)
        nn.init.orthogonal(self.h3.weight)
        self.h4 = nn.Linear(n_h3, n_h4)
        nn.init.orthogonal(self.h4.weight)
        self.h5 = nn.Linear(n_h4, n_h5)
        nn.init.orthogonal(self.h5.weight)
        self.h6 = nn.Linear(n_h5, n_h6)
        nn.init.orthogonal(self.h6.weight)
        self.h7 = nn.Linear(n_h6, n_h7)
        nn.init.orthogonal(self.h7.weight)
        self.h8 = nn.Linear(n_h7, n_h8)
        nn.init.orthogonal(self.h8.weight)
        self.h9 = nn.Linear(n_h8, n_h9)
        nn.init.orthogonal(self.h9.weight)
        self.output = nn.Linear(n_h9, n_output)
        nn.init.orthogonal(self.output.weight)

    def forward(self, X):
        X = F.relu(self.h1(X))
        # X = F.leaky_relu(self.h1(X), negative_slope=0.0001)
        X = F.relu(self.h2(X))
        # X = F.leaky_relu(self.h2(X), negative_slope=0.0001)
        X = F.relu(self.h3(X))
        # X = F.leaky_relu(self.h3(X), negative_slope=0.0001)
        X = F.dropout(X, p=0.2, training=self.training)
        X = F.relu(self.h4(X))
        # X = F.leaky_relu(self.h4(X), negative_slope=0.0001)
        X = F.relu(self.h5(X))
        # X = F.leaky_relu(self.h5(X), negative_slope=0.0001)
        X = F.relu(self.h6(X))
        # X = F.leaky_relu(self.h6(X), negative_slope=0.0001)
        X = F.dropout(X, p=0.2, training=self.training)
        X = F.relu(self.h7(X))
        X = F.relu(self.h8(X))
        X = F.relu(self.h9(X))
        X = F.dropout(X, p=0.2, training=self.training)
        X = self.output(X)
        return X

class MLP_BN(nn.Module):
    def __init__(self):
        super(MLP_BN, self).__init__()
        # self.bn_input = nn.BatchNorm1d(N_INPUT, momentum=BN_MOMEN) # for input data
        self.h1 = nn.Linear(N_INPUT, N_H1)
        self._set_init(self.h1)
        self.bn_h1 = nn.BatchNorm1d(N_H1, momentum=BN_MOMEN)
        self.h2 = nn.Linear(N_H1, N_H2)
        self._set_init(self.h2)
        self.bn_h2 = nn.BatchNorm1d(N_H2, momentum=BN_MOMEN)
        self.h3 = nn.Linear(N_H2, N_H3)
        self._set_init(self.h3)
        self.bn_h3 = nn.BatchNorm1d(N_H3, momentum=BN_MOMEN)
        self.output = nn.Linear(N_H3, N_OUTPUT)
        self._set_init(self.output)

    def _set_init(self, layer):
        nn.init.orthogonal(layer.weight)
        nn.init.constant(layer.bias, BIAS)

    def forward(self, X):
        # X = self.bn_input(X)
        X = ACT1(self.h1(X))
        X = self.bn_h1(X)
        X = ACT2(self.h2(X))
        X = self.bn_h2(X)
        X = ACT2(self.h3(X))
        X = self.bn_h3(X)
        X = self.output(X)
        return X


X, y = input_data(2014, n_classes=2) #, preprocessing='smote_2_class'
X = normalize(X, norm='l2', axis=0)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print('Finished input data, resampling...')

# X = normalize(X, norm='l2', axis=0)

# X, y = iD.smote(X, y)
X, y = iD.DE_synthetic(X, y, int(X.shape[0] / 10), 20, super=None) #int(X.shape[0] / 10)
X, y = shuffle(X, y)

X_val_tensor = Variable(torch.from_numpy(X_val).cuda().type(tFloat), requires_grad=False)
y_val_tensor = Variable(torch.from_numpy(y_val).cuda().type(tLong), requires_grad=False)

# mlp = MLPnet(n_feature=27, n_h1=40, n_h2=25, n_h3=10, n_output=2)

mlp = MLP_BN().cuda()

# mlp = MLPnet_deeperer(n_feature=27,
#                       n_h1=40,
#                       n_h2=50,
#                       n_h3=60,
#                       n_h4=50,
#                       n_h5=40,
#                       n_h6=30,
#                       n_h7=20,
#                       n_h8=10,
#                       n_h9=5,
#                       n_output=2)

# optimizer = opt.SGD(mlp.parameters(), lr=0.0001, momentum=0.9) #, weight_decay=0.05
optimizer = opt.Adam(mlp.parameters(), lr=LR, weight_decay=0.05)
cost_function = nn.CrossEntropyLoss(weight=WC).cuda()
iter_set, loss_set, train_acc_set, val_acc_set, val_recall_set = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Iterations')
ax.set_ylabel('% Accuracy')
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
ax2.set_ylim(0, 2)
plt.ion()
plt.show()

for i in range(iter):
    X_batch, y_batch = batch_feed(i, X, y, batch_size=batch_size)
    # X_batch, y_batch = iD.nearmiss(X_batch, y_batch)
    # X_batch, y_batch = iD.smote(X_batch, y_batch)
    X_batch_tensor = Variable(torch.from_numpy(X_batch).cuda().type(tFloat), requires_grad=False)
    y_batch_tensor = Variable(torch.from_numpy(y_batch).cuda().type(tLong), requires_grad=False)

    logit = mlp(X_batch_tensor)
    loss = cost_function(logit, y_batch_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        _, pred = torch.max(F.softmax(logit, dim=1), 1) # softmax with row summation = 1.0
        y_pred = pred.data.cpu().numpy()
        accuracy = sum(y_pred == y_batch) / batch_size
        # accuracy = recall_score(y_batch, y_pred, pos_label=0)

        with torch.no_grad():
            _, pred_val = torch.max(F.softmax(mlp(X_val_tensor), dim=1), 1)
            y_pred_val = pred_val.data.cpu().numpy()
            acc_val = sum(y_pred_val == y_val) / len(y_val)
            recall_val = f1_score(y_val, y_pred_val, pos_label=0)
            val_acc_set = np.append(val_acc_set, acc_val)
            val_recall_set = np.append(val_recall_set, recall_val)

        iter_set = np.append(iter_set, i)
        loss_set = np.append(loss_set, loss.item())
        train_acc_set = np.append(train_acc_set, accuracy)

        try:
            ax.lines.remove(line1)
            ax.lines.remove(line2)
            ax.lines.remove(line4)
            ax2.lines.remove(line3)
        except Exception:
            pass
        line1, = ax.plot(iter_set, train_acc_set, '-r', label='Training_acc')
        line2, = ax.plot(iter_set, val_acc_set, '-b', label='Validation_acc')
        line3, = ax2.plot(iter_set, loss_set, '-k', label='Loss')
        line4, = ax.plot(iter_set, val_recall_set, '-g', label='Validation_F1-score')
        ax.legend(loc=3)
        ax2.legend(loc=8)
        plt.pause(0.1)

print('Training Done!')
print('Confusion Matrix: ')
print(confusion_matrix(y_val, y_pred_val))
print('test accuracy: %.2f' %(accuracy_score(y_val, y_pred_val)))
print('test F1-score: %.2f' %(recall_score(y_val, y_pred_val, pos_label=0)))

# output_data(X_batch_res, y_batch_res)

# torch.save(mlp.state_dict(), 'mlp.pt')

# model.load_state_dict(torch.load(filepath))
# model.eval()