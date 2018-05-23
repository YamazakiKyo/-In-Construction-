import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ictaiDefs.DataInput import input_data
from ictaiDefs.DataInput import batch_feed

'''
1. Good explain about CrossEntropyLoss in Pytorch: http://sshuair.com/2017/10/21/pytorch-loss/ 
2. 
'''

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
tFloat = torch.cuda.FloatTensor
tLong = torch.cuda.LongTensor

class MLPnet(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_output):
        super(MLPnet, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        self.h2 = nn.Linear(n_h1, n_h2)
        self.h3 = nn.Linear(n_h2, n_h3)
        self.output = nn.Linear(n_h3, n_output)

    def forward(self, X):
        X = F.tanh(self.h1(X))
        X = F.relu(self.h2(X))
        X = F.relu(self.h3(X))
        X = self.output(X)
        return X

X, y = input_data(2006, n_classes=2, preprocessing='smote_2_class')
X_val, y_val = input_data(2007, n_classes=2)

X_val_tensor = Variable(torch.from_numpy(X_val).cuda().type(tFloat), requires_grad=False)
y_val_tensor = Variable(torch.from_numpy(y_val).cuda().type(tLong), requires_grad=False)

mlp = MLPnet(n_feature=27, n_h1=50, n_h2=25, n_h3=10, n_output=2)

batch_size = 500
iter = 200000
WC = torch.tensor([1.25, 1]).cuda().type(tFloat)

mlp = mlp.cuda()
# optimizer = opt.SGD(mlp.parameters(), lr=0.00001, momentum=0.6)
optimizer = opt.Adamax(mlp.parameters(), lr=0.001)
cost_function = nn.CrossEntropyLoss(weight=WC).cuda()

iter_set = np.array([])
loss_set = np.array([])
train_acc_set = np.array([])
val_acc_set = np.array([])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Iterations')
ax.set_ylabel('% Accuracy')
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
plt.ion()
plt.show()

for i in range(iter):
    X_batch, y_batch = batch_feed(i, X, y, batch_size=batch_size)
    X_batch_tensor = Variable(torch.from_numpy(X_batch).cuda().type(tFloat), requires_grad=False)
    y_batch_tensor = Variable(torch.from_numpy(y_batch).cuda().type(tLong), requires_grad=False)

    logit = mlp(X_batch_tensor)
    loss = cost_function(logit, y_batch_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 1000 == 0:
        _, pred = torch.max(F.softmax(logit, dim=1), 1)
        y_pred = pred.data.cpu().numpy()
        accuracy = sum(y_pred == y_batch) / batch_size

        with torch.no_grad():
            _, pred_val = torch.max(F.softmax(mlp(X_val_tensor), dim=1), 1)
            y_pred_val = pred_val.data.cpu().numpy()
            acc_val = sum(y_pred_val == y_val) / len(y_val)
            val_acc_set = np.append(val_acc_set, acc_val)

        iter_set = np.append(iter_set, i)
        loss_set = np.append(loss_set, loss.item())
        train_acc_set = np.append(train_acc_set, accuracy)

        try:
            ax.lines.remove(line1)
            ax.lines.remove(line2)
            ax2.lines.remove(line3)
        except Exception:
            pass
        line1, = ax.plot(iter_set, train_acc_set, '-r', label='Training_acc')
        line2, = ax.plot(iter_set, val_acc_set, '-b', label='Validation_acc')
        line3, = ax2.plot(iter_set, loss_set, '-k', label='Loss')
        ax.legend(loc=3)
        ax2.legend(loc=8)
        plt.pause(0.1)

print('Training Done!')
print('Confusion Matrix: ')
print(confusion_matrix(y_val, y_pred_val))

# torch.save(mlp.state_dict(), 'mlp.pt')

# model.load_state_dict(torch.load(filepath))
# model.eval()
