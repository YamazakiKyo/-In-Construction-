import torch.nn as nn
import torch.nn.functional as F

ACT1 = F.relu
ACT2 = F.relu

N_INPUT = 29
N_H1 = 35
N_H2 = 20
N_H3 = 10
N_OUTPUT = 2
BIAS = 0.01
BN_MOMEN = 0.5

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

class MLPnet_deeper(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_h4,n_h5, n_h6, n_output):
        super(MLPnet_deeper, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        self._set_init(self.h1)
        self.bn_h1 = nn.BatchNorm1d(n_h1, momentum=BN_MOMEN)
        self.h2 = nn.Linear(n_h1, n_h2)
        self._set_init(self.h2)
        self.bn_h2 = nn.BatchNorm1d(n_h2, momentum=BN_MOMEN)
        self.h3 = nn.Linear(n_h2, n_h3)
        self._set_init(self.h3)
        self.bn_h3 = nn.BatchNorm1d(n_h3, momentum=BN_MOMEN)
        self.h4 = nn.Linear(n_h3, n_h4)
        self._set_init(self.h4)
        self.bn_h4 = nn.BatchNorm1d(n_h4, momentum=BN_MOMEN)
        self.h5 = nn.Linear(n_h4, n_h5)
        self._set_init(self.h5)
        self.bn_h5 = nn.BatchNorm1d(n_h5, momentum=BN_MOMEN)
        self.h6 = nn.Linear(n_h5, n_h6)
        self._set_init(self.h6)
        self.bn_h6 = nn.BatchNorm1d(n_h6, momentum=BN_MOMEN)
        self.output = nn.Linear(n_h6, n_output)
        self._set_init(self.output)

    def _set_init(self, layer):
        nn.init.orthogonal(layer.weight)
        nn.init.constant(layer.bias, BIAS)

    def forward(self, X):
        X = F.relu(self.h1(X))
        X = self.bn_h1(X)
        X = F.relu(self.h2(X))
        X = self.bn_h2(X)
        X = F.relu(self.h3(X))
        X = self.bn_h3(X)
        X = F.dropout(X, p=0.5, training=self.training)
        X = F.relu(self.h4(X))
        X = self.bn_h4(X)
        X = F.relu(self.h5(X))
        X = self.bn_h5(X)
        X = F.relu(self.h6(X))
        X = self.bn_h6(X)
        X = F.dropout(X, p=0.5, training=self.training)
        X = self.output(X)
        return X

class MLPnet_deeperer(nn.Module):
    def __init__(self, n_feature, n_h1, n_h2, n_h3, n_h4,n_h5, n_h6,n_h7,n_h8, n_h9, n_output):
        super(MLPnet_deeperer, self).__init__()
        self.h1 = nn.Linear(n_feature, n_h1)
        self._set_init(self.h1)
        self.bn_h1 = nn.BatchNorm1d(n_h1, momentum=BN_MOMEN)
        self.h2 = nn.Linear(n_h1, n_h2)
        self._set_init(self.h2)
        self.bn_h2 = nn.BatchNorm1d(n_h2, momentum=BN_MOMEN)
        self.h3 = nn.Linear(n_h2, n_h3)
        self._set_init(self.h3)
        self.bn_h3 = nn.BatchNorm1d(n_h3, momentum=BN_MOMEN)
        self.h4 = nn.Linear(n_h3, n_h4)
        self._set_init(self.h4)
        self.bn_h4 = nn.BatchNorm1d(n_h4, momentum=BN_MOMEN)
        self.h5 = nn.Linear(n_h4, n_h5)
        self._set_init(self.h5)
        self.bn_h5 = nn.BatchNorm1d(n_h5, momentum=BN_MOMEN)
        self.h6 = nn.Linear(n_h5, n_h6)
        self._set_init(self.h6)
        self.bn_h6 = nn.BatchNorm1d(n_h6, momentum=BN_MOMEN)
        self.h7 = nn.Linear(n_h6, n_h7)
        self._set_init(self.h7)
        self.bn_h7 = nn.BatchNorm1d(n_h7, momentum=BN_MOMEN)
        self.h8 = nn.Linear(n_h7, n_h8)
        self._set_init(self.h8)
        self.bn_h8 = nn.BatchNorm1d(n_h8, momentum=BN_MOMEN)
        self.h9 = nn.Linear(n_h8, n_h9)
        self._set_init(self.h9)
        self.bn_h9 = nn.BatchNorm1d(n_h9, momentum=BN_MOMEN)
        self.output = nn.Linear(n_h9, n_output)
        self._set_init(self.output)

    def _set_init(self, layer):
        nn.init.orthogonal(layer.weight)
        nn.init.constant(layer.bias, BIAS)

    def forward(self, X):
        X = F.relu(self.h1(X))
        # X = F.leaky_relu(self.h1(X), negative_slope=0.0001)
        X = self.bn_h1(X)
        X = F.relu(self.h2(X))
        # X = F.leaky_relu(self.h2(X), negative_slope=0.0001)
        X = self.bn_h2(X)
        X = F.relu(self.h3(X))
        # X = F.leaky_relu(self.h3(X), negative_slope=0.0001)
        X = self.bn_h3(X)
        X = F.dropout(X, p=0.2, training=self.training)
        X = F.relu(self.h4(X))
        # X = F.leaky_relu(self.h4(X), negative_slope=0.0001)
        X = self.bn_h4(X)
        X = F.relu(self.h5(X))
        # X = F.leaky_relu(self.h5(X), negative_slope=0.0001)
        X = self.bn_h5(X)
        X = F.relu(self.h6(X))
        # X = F.leaky_relu(self.h6(X), negative_slope=0.0001)
        X = self.bn_h6(X)
        X = F.dropout(X, p=0.2, training=self.training)
        X = F.relu(self.h7(X))
        X = self.bn_h7(X)
        X = F.relu(self.h8(X))
        X = self.bn_h8(X)
        X = F.relu(self.h9(X))
        X = self.bn_h9(X)
        X = F.dropout(X, p=0.2, training=self.training)
        X = self.output(X)
        return X