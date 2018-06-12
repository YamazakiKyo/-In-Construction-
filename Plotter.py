import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
import numpy as np
import pandas as pd
import math
import ictaiDefs.DataInput as iD



X, y = make_classification(n_classes=2, class_sep=0.8, weights=[0.05, 0.95],
                           n_informative=2, n_redundant=0, flip_y=0.1,
                           n_features=3, n_clusters_per_class=1,
                           n_samples=5000, random_state=42)

# X, y = iD.input_data('2014_v2', n_classes=2)
# X, y = shuffle(X, y)
# X, y = iD.batch_feed(10, X, y, 10000)

min_alpha = 0.8
maj_alpha = 0.3

X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=2)
X_vis = pca.fit_transform(X)

# NearMiss
X_NM, y_NM = iD.nearmiss(X, y)
X_NM_vis = pca.fit_transform(X_NM)

# + SMOTE
# X_SMOTE, y_SMOTE = iD.smote(X_NM, y_NM)
X_SMOTE, y_SMOTE = iD.smote(X, y)
X_SMOTE_vis = pca.fit_transform(X_SMOTE)

# Synthetic (Accuracy Supervised)
X_syn, y_syn = iD.DE_synthetic(X, y, int(X.shape[0] / 5), 20, req=0.1)
X_sym_vis = pca.fit_transform(X_syn)

# PLOT
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=min_alpha, marker='*')
c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=maj_alpha)
ax1.set_title('Original set')

ax2.scatter(X_NM_vis[y_NM == 0, 0], X_NM_vis[y_NM == 0, 1],
            label="Class #0", alpha=min_alpha, marker='*')
ax2.scatter(X_NM_vis[y_NM == 1, 0], X_NM_vis[y_NM == 1, 1],
            label="Class #1", alpha=maj_alpha)
ax2.set_title('NearMiss-1')


ax3.scatter(X_SMOTE_vis[y_SMOTE == 0, 0], X_SMOTE_vis[y_SMOTE == 0, 1], label="Class #0",
                 alpha=min_alpha, marker='*')
ax3.scatter(X_SMOTE_vis[y_SMOTE == 1, 0], X_SMOTE_vis[y_SMOTE == 1, 1], label="Class #1",
                 alpha=maj_alpha)
ax3.set_title('Borderline-SMOTE')

ax4.scatter(X_sym_vis[y_syn == 0, 0], X_sym_vis[y_syn == 0, 1], label="Class #0",
                 alpha=min_alpha, marker='*')
ax4.scatter(X_sym_vis[y_syn == 1, 0], X_sym_vis[y_syn == 1, 1], label="Class #1",
                 alpha=maj_alpha)
ax4.set_title('SDSE')

# make nice plotting
for ax in (ax1, ax2, ax3,ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

f.legend((c0, c1), ('Minority Class', 'Majority Class'), loc='lower center', ncol=2, labelspacing=0., fontsize=15, markerscale=3)
plt.tight_layout(pad=3)
plt.show()


