import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score, fbeta_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ictaiDefs.DataInput import input_data
from ictaiDefs.DataInput import batch_feed
from ictaiDefs.DataInput import output_data
import ictaiDefs.DataInput as iD
import ictaiDefs.CustMLP as iC
tFloat = torch.cuda.FloatTensor
tLong = torch.cuda.LongTensor
torch.cuda.manual_seed(42)

'''
1. Good explain about CrossEntropyLoss in Pytorch: http://sshuair.com/2017/10/21/pytorch-loss/ 
'''


'''Hyper Parameters'''
batch_size = 1000
iter = 70000
LR = 0.00001
L2_Reg = 0
DE_batch = 12
acc_ctrl = 0.6
recall_ctrl = 0.8
evo_super = None #(None, 'f1', 'recall')
evo_req = 0.05
whole_Norm = False
weighted_cost = torch.tensor([1, 1]).cuda().type(tFloat)

'''Fetch Data'''
X, y = input_data('2014_v2', n_classes=2) #, preprocessing='smote_2_class'
if whole_Norm:
    X = normalize(X, norm='l2', axis=0)
X, X_val, y, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_val_tensor = Variable(torch.from_numpy(X_val).cuda().type(tFloat), requires_grad=False)
y_val_tensor = Variable(torch.from_numpy(y_val).cuda().type(tLong), requires_grad=False)

'''Hybrid Re-sampling'''
# X, y = iD.nearmiss(X, y)
# X, y = iD.smote(X, y)
X, y = iD.DE_synthetic(X, y, int(X.shape[0] / DE_batch), 20, super=evo_super, req=evo_req)
X, y = shuffle(X, y)
n_batch = math.ceil(X.shape[0] / batch_size)
print('Number of batches: ', n_batch)

'''Choose MLP model'''
mlp = iC.MLP_BN().cuda()

# mlp = iC.MLPnet(n_feature=27, n_h1=40, n_h2=25, n_h3=10, n_output=2).cuda()

# mlp = iC.MLPnet_deeper(n_feature=29, n_h1=35, n_h2=40,
#                     n_h3=30, n_h4=20, n_h5=10,
#                     n_h6=5, n_output=2).cuda()

# mlp = MLPnet_deeperer(n_feature=29, n_h1=35, n_h2=45,
#                       n_h3=55, n_h4=50, n_h5=40,
#                       n_h6=30, n_h7=20, n_h8=10,
#                       n_h9=5, n_output=2).cuda()
'''How to train?'''
optimizer = opt.Adam(mlp.parameters(), lr=LR, weight_decay=L2_Reg)
cost_function = nn.CrossEntropyLoss(weight=weighted_cost).cuda()
'''Plot as Anime'''
iter_set, loss_set, train_acc_set,  = np.array([]), np.array([]), np.array([])
val_acc_set, val_recall_set, val_f1_set =np.array([]), np.array([]), np.array([])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Epoches')
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

    if i % n_batch == 0:
        _, pred = torch.max(F.softmax(logit, dim=1), 1) # softmax with row summation = 1.0
        y_pred = pred.data.cpu().numpy()
        accuracy = sum(y_pred == y_batch) / batch_size

        with torch.no_grad():
            _, pred_val = torch.max(F.softmax(mlp(X_val_tensor), dim=1), 1)
            y_pred_val = pred_val.data.cpu().numpy()
            acc_val = sum(y_pred_val == y_val) / len(y_val)
            f1_val = f1_score(y_val, y_pred_val, pos_label=0)
            recall_val = recall_score(y_val, y_pred_val, pos_label=0)
            val_acc_set = np.append(val_acc_set, acc_val)
            val_recall_set = np.append(val_recall_set, recall_val)
            val_f1_set = np.append(val_f1_set, f1_val)
            if (acc_val > acc_ctrl) and (recall_val > recall_ctrl):
                break

        iter_set = np.append(iter_set, int(i / n_batch))
        loss_set = np.append(loss_set, loss.item())
        train_acc_set = np.append(train_acc_set, accuracy)

        try:
            ax.lines.remove(line1)
            ax.lines.remove(line2)
            ax.lines.remove(line4)
            ax.lines.remove(line5)
            ax2.lines.remove(line3)
        except Exception:
            pass
        line1, = ax.plot(iter_set, train_acc_set, '-r', label='Training_acc')
        line2, = ax.plot(iter_set, val_acc_set, '-b', label='Validation_acc')
        line4, = ax.plot(iter_set, val_f1_set, '-g', label='Validation_F1-score')
        line5, = ax.plot(iter_set, val_recall_set, '-c', label='Validation_recall')
        line3, = ax2.plot(iter_set, loss_set, '-k', label='Loss')
        ax.legend(loc=3)
        ax2.legend(loc=8)
        plt.pause(0.1)

print('Training Done!')
print('Confusion Matrix: ')
print(confusion_matrix(y_val, y_pred_val))
print('test accuracy: %.2f' %(accuracy_score(y_val, y_pred_val)))
print('test F1-score: %.2f' %(f1_score(y_val, y_pred_val, pos_label=0)))
print('test recall: %.2f' %(recall_score(y_val, y_pred_val, pos_label=0)))
print('test precision: %.2f' %(precision_score(y_val, y_pred_val, pos_label=0)))