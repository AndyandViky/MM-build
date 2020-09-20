# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: xgb.py
@time: 2020/9/18 15:49
@desc: xgb.py
'''
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


def print_metrix(y_train, X_train, y_test, X_test, clf):

    score = accuracy_score(y_train, clf.predict(X_train))
    test_score = accuracy_score(y_test, clf.predict(X_test))
    p_score = precision_score(y_test, clf.predict(X_test), average='weighted')
    recall = recall_score(y_test, clf.predict(X_test), average='weighted')
    f1 = f1_score(y_test, clf.predict(X_test), average='weighted')
    print("Training data size: %.d" % len(y_train), "Labeld accuracy: %.4f" % score,
          " , Unlabeled ACC: %.4f" % test_score, 'P-score: %.4f' % p_score,
          'recall: %.4f' % recall, 'F1-score: %.4f' % f1)


def selfTraining(X_train, X_test, y_train, y_test):

    rfc_model = RandomForestClassifier()
    et_model = ExtraTreesClassifier()
    gnb_model = GaussianNB()
    knn_model = KNeighborsClassifier()
    lr_model = LogisticRegression()
    dt_model = DecisionTreeClassifier()
    svc_model = SVC()

    xgbc_model = XGBClassifier(
        learning_rate=0.07,
        n_estimators=90,
        max_depth=5,
        booster='dart',
        min_child_weight=0.7,
        gamma=0.4,
        subsample=1,
        colsample_bytree=1,
        objective='multi:softprob',
        nthread=4,
        scale_pos_weight=1,
    )

    clf = xgbc_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)
    cm = confusion_matrix(y_test, clf.predict(X_test))
    from plot import Q4
    Q4().confusion_matrix(cm)

    # 随机森林
    clf = rfc_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # ET
    clf = et_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # 朴素贝叶斯
    clf = gnb_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # K最近邻
    clf = knn_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # 逻辑回归
    clf = lr_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # 决策树
    clf = dt_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)

    # 支持向量机
    clf = svc_model.fit(X_train, y_train)
    print_metrix(y_train, X_train, y_test, X_test, clf)
