import numpy as np
import scipy as sc
import os
import re
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV,LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from numpy import sort
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,cross_val_predict
from xgboost import XGBClassifier
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
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
df_source = pd.read_csv('non_graph.csv')
Precision = []
Recall = []
F_measure = []
feature_importances=[]
ROC = []
PRC = []
F1_best = []
F2 = []
F2_best = []
ACC = []
Name=[[] for i in range(1201)]

for r in range(100):
    df = df_source.copy()
    y_pred = []
    y_real = []
    y_proba_minority = []
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1245)
    
    df=shuffle(df) 
    df=shuffle(df) 
    df=shuffle(df)
    df=shuffle(df)

    df_nontestData=df
    y = df_nontestData['graph_labels'].values
    name = df_nontestData['name'].values
    X = df_nontestData.drop(columns=['graph_labels','name'])
    # pd.set_option('display.max_columns',None)

    X = (X-X.mean())/X.std()
    X = X.values
    if r==0:
        parameters = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
        model = LogisticRegression(class_weight={1: 2},random_state=1,solver='liblinear')
        grid_search = GridSearchCV(
                estimator=model,
                param_grid=parameters,
                scoring = 'f1',
                n_jobs = 10,
                cv = 10,
                verbose=True
            )
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        clf=grid_search.best_estimator_
    for i, (train_index, test_index) in enumerate(k_fold.split(X,y)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        # print(ytest)
        NameXtrain, NameXtest = name[train_index], name[test_index]
        # clf=svm.SVC(class_weight={1: 2},probability=True)
        # clf = LogisticRegression(penalty='l1', class_weight={1: 2},random_state=1)
        # clf = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(64,), random_state=1)
        # clf = XGBClassifier(learning_rate=0.05, n_estimators=60, objective='binary:logistic', silent=True, max_depth=2, nthread=4, scale_pos_weight=2)
        # clf = tree.DecisionTreeClassifier(class_weight={0:0.5,1:1})
        # clf = RandomForestClassifier(n_estimators=20,class_weight={0:0.5,1:1})
        clf.fit(Xtrain, ytrain)
        # feature_importances.append(clf.feature_importances_)
        # print(type(sort(clf.feature_importances_)))

        pred = clf.predict(Xtest)

        for idx in range(NameXtest.shape[0]):
            # print(name[idx],preds[idx])
            Name[NameXtest[idx]].append(pred[idx])
        
        # pred_proba = grid_search.best_estimator_.predict_proba(Xtest)
        pred_proba = clf.predict_proba(Xtest)
        y_real.append(ytest)
        y_pred.append(pred)
        y_proba_minority.append(pred_proba)
    y_pred = np.concatenate(y_pred)
    y_real = np.concatenate(y_real)
    y_proba_minority = np.concatenate(y_proba_minority)

    roc = roc_auc_score(y_real, y_proba_minority[:,1])
    prc = average_precision_score(
        y_real,
        y_proba_minority[:,1],
        pos_label=1,
    )
    precision, recall, thresholds = precision_recall_curve(
        y_real,
        y_proba_minority[:,1],
        pos_label=1,
    )
    recall_=0
    precision_=0
    Thresholds=0
    f1_best=0
    for pre_i in range(len(precision)):
        f=f1(precision[pre_i],recall[pre_i])
        if(f>f1_best):
            f1_best=f
            Thresholds=thresholds[pre_i]
            recall_=recall[pre_i]
            precision_=precision[pre_i]

    F1_best.append(f1_best)
    y_pred = (y_proba_minority[:,1] >= Thresholds).astype(int)
    acc=accuracy_score(y_real,y_pred)

    for label_index, label in enumerate([1]):
        print(
            "Overall Precision, Recall, ROC, PRC, ACC for Class",
            label,
            ": ",
            precision_,
            ", ",
            recall_,
            ", ",
            f1_best,
            ", ",
            roc,
            ", ",
            prc,
            ", ",
            acc
        )
    
    Precision.append(precision_)
    Recall.append(recall_)
    ROC.append(roc)
    PRC.append(prc)
    ACC.append(acc)

# print(Name)
Stability=[]
for id_name in Name:
    if(id_name!=[]):
        id_name = np.array(id_name)
        counts = np.bincount(id_name)
        stability = np.max(counts)/id_name.shape[0]
        Stability.append(stability)
Stability=np.array(Stability)
print("Stability:",np.mean(Stability))
def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")
print_to_file("record.txt","lg 10 cv Precision:%.3f Recall:%.3f F1_best:%.3f F1_best(std): %.3f ROC:%.3f ROC(std): %.3f PRC:%.3f ACC:%.3f for Class Fellow" % (np.mean(Precision),np.mean(Recall),np.mean(F1_best),np.std(F1_best),np.mean(ROC),np.std(ROC),np.mean(PRC),np.mean(ACC)))

print("100 cv Precision, Recall, F1_best, ROC, PRC, ACC for Class:",np.mean(Precision),np.mean(Recall),np.mean(F1_best),np.mean(ROC),np.mean(PRC),np.mean(ACC))


