import numpy as np
import scipy as sc
import os
import re
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import RidgeClassifierCV,LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from numpy import sort
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,cross_val_predict
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.feature_selection import (mutual_info_classif, SelectKBest, chi2, SelectPercentile, SelectFromModel, SequentialFeatureSelector, SequentialFeatureSelector, f_regression)
import matplotlib.pyplot as plt

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def f1(precision,recall):
    if(precision==0 or recall==0):
        return 0
    return (recall*precision*2)/(recall+precision)

def f2(precision,recall):
    if(precision==0 or recall==0):
        return 0
    return (recall*precision*5)/(recall+4*precision)

from sklearn.model_selection import GridSearchCV

Precision = []
Recall = []
F_measure = []
feature_importances=[]
ROC = []
ROC_train = []
PRC = []
F1 = []
F1_train = []
F1_best = []
F2 = []
F2_best = []
ACC = []
Name=[[] for i in range(1201)]

for r in range(20):
    df = pd.read_csv('label_citation_data_1797rows.csv')
    df_copy = df.copy()
    y_pred = []
    y_real = []
    y_proba_minority = []
    y_pred_train = []
    y_real_train = []
    y_proba_minority_train = []
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1245)
    df=df.replace("\\N",np.nan)
    df.fillna(-2,inplace=True)
    df=df.replace("Others",0)
    df=df.replace("Extends",1)
    df=shuffle(df) 
    df=shuffle(df) 
    df=shuffle(df)

    df_nontestData=df
    y = df_nontestData['cite_function'].values
    X = df_nontestData.drop(columns=['citingpaperID','citedpaperID','cite_function'])
    # X = X.iloc[:,1:10]
    # print(X)
    # pd.set_option('display.max_columns',None)

    # X = (X-X.mean())/X.std()
    X = X.values
    # if r==0:
    #     parameters = { 
    #     'n_estimators': [20, 50],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [4,5,6,7,8],
    #     'criterion' :['gini', 'entropy']
    #     }
    #     model = ExtraTreesClassifier()
    #     grid_search = GridSearchCV(
    #             estimator=model,
    #             param_grid=parameters,
    #             scoring = 'f1',
    #             n_jobs = 10,
    #             cv = 10,
    #             verbose=True
    #         )
    #     grid_search.fit(X, y)
    #     print(grid_search.best_params_)
    #     clf=grid_search.best_estimator_

    for i, (train_index, test_index) in enumerate(k_fold.split(X,y)):

        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        # clf=svm.SVC(class_weight={1: 2},probability=True)
        # clf = LogisticRegression(penalty='l1', solver='saga', max_iter=200, class_weight={1: 2},random_state=1)
        # clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(64,), random_state=1)
        # clf = XGBClassifier(learning_rate=0.05, n_estimators=60, objective='binary:logistic', silent=True, max_depth=2, nthread=4, scale_pos_weight=2)
        # clf = tree.DecisionTreeClassifier(class_weight={0:0.5,1:1})
        # clf = RandomForestClassifier(n_estimators=20,class_weight={0:0.5,1:1})
        clf = ExtraTreesClassifier(n_estimators=800, n_jobs=-1,
                               class_weight={1: 1.2, 0: 1},max_depth=3+r)
        clf.fit(Xtrain, ytrain)
        # feature_importances.append(clf.feature_importances_)
        # print(type(sort(clf.feature_importances_)))

        pred = clf.predict(Xtest)
        pred_train = clf.predict(Xtrain)
        # pred_proba = grid_search.best_estimator_.predict_proba(Xtest)
        pred_proba = clf.predict_proba(Xtest)
        pred_proba_train = clf.predict_proba(Xtrain)
        y_real.append(ytest)
        y_real_train.append(ytrain)
        y_pred.append(pred)
        y_pred_train.append(pred_train)
        y_proba_minority.append(pred_proba)
        y_proba_minority_train.append(pred_proba_train)
    y_pred = np.concatenate(y_pred)
    y_real = np.concatenate(y_real)
    y_proba_minority = np.concatenate(y_proba_minority)
    y_pred_train = np.concatenate(y_pred_train)
    y_real_train = np.concatenate(y_real_train)
    y_proba_minority_train = np.concatenate(y_proba_minority_train)

    roc = roc_auc_score(y_real, y_proba_minority[:,1])
    f1_ = f1_score(y_real, y_pred)
    ROC.append(roc)
    F1.append(f1_)

    roc = roc_auc_score(y_real_train, y_proba_minority_train[:,1])
    f1_ = f1_score(y_real_train, y_pred_train)
    ROC_train.append(roc)
    F1_train.append(f1_)
print(F1_train,F1)
print(ROC_train,ROC)
plt.plot(range(3,23), F1_train, "o-", color="r", label="Training score")
plt.plot(range(3,23), F1, "o-", color="g", label="test score")
plt.plot(range(3,23), ROC_train, "--", color="r", label="Training score")
plt.plot(range(3,23), ROC, "--", color="g", label="test score")
plt.show()


