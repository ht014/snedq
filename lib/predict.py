import os
import pickle as pkl
import numpy as np
import scipy.io
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred

def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def load_label(file, variable_name="group"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
    print(label.shape, type(label), label.min(), label.max())
    return label

def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        # print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        logger.info("micro f1 %f macro f1 %f", mi, ma)
        micro.append(mi)
        macro.append(ma)
    print("%d fold validation, training ratio %f", len(micro), train_ratio)
    print("Average micro %.2f, Average macro %.2f",
            np.mean(micro) * 100,
            np.mean(macro) * 100)
