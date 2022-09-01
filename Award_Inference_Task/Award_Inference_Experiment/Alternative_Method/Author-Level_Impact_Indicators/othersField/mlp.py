import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import numpy as np
import scipy as sc
import os
import re
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV,LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from numpy import sort
from sklearn.model_selection import cross_val_score,cross_val_predict
from xgboost import XGBClassifier
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()

        self.enhance = enhance

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x.cuda())
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = F.log_softmax(self.fc3(x))
        return x

class cDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

import numpy as np
import sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader, random_split


def f1(precision,recall):
    if(precision==0 or recall==0):
        return 0
    return (recall*precision*2)/(recall+precision)

def f2(precision,recall):
    if(precision==0 or recall==0):
        return 0
    return (recall*precision*5)/(recall+4*precision)

Precision = []
Recall = []
F_measure = []
feature_importances=[]
ROC = []
PRC = []
ACC = []
F1_best = []
F2 = []
F2_best = []
Name=[[] for i in range(1201)]

for r in range(100):
    df = pd.read_csv('non_graph.csv')
    df_copy = df.copy()
    y_pred = []
    y_real = []
    y_proba_minority = []
    k_fold = KFold(n_splits=10, shuffle=True, random_state=1245)
    
    df=shuffle(df) 
    df=shuffle(df) 
    df=shuffle(df)
    df=shuffle(df)

    df_nontestData=df
    # featureImportances=['year','ave_citation','ave_authorOrder','ave_year_sub_first','ave_papers_key','authorCitationNLP','authorHIndexNLP','sum_paperCitation','avg_paperCitation','sum_avgOutDegreeCitation','sum_paperCitation-avgOutDegreeYearspan','sum_paperCitation-maxOutDegreeYearspan','nodeSize','edgeSize','componentSize','hIndexGraph','hIndexhComponent']
    Y = df_nontestData['graph_labels'].values
    name = df_nontestData['name'].values
    X = df_nontestData.drop(columns=['graph_labels','name'])

    # pd.set_option('display.max_columns',None)
    # print(X.corr())
    # s()
    
    X = (X-X.mean())/X.std()
    X = X.values
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = Y[train_index], Y[test_index]
        NameXtrain, NameXtest = name[train_index], name[test_index]
        device='cuda:0'
        train_dataset = cDataset(Xtrain, ytrain)
        test_dataset = cDataset(Xtest, ytest)
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        model = MLP(7, 64, 2).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001) 

        n_epochs, best_loss, step, early_stop_count = 10, math.inf, 0, 0

        for epoch in range(n_epochs):
            model.train() # Set your model to train mode.
            
            # tqdm is a package to visualize your training progress.
            train_pbar = tqdm(train_loader, position=0, leave=True)

            for x, y in train_pbar:
                optimizer.zero_grad()               # Set gradient to zero.
                x, y = x.to(device), y.to(device)   # Move your data to device. 
                pred = model(x)       
                weights = np.array([1,2])
                weights = torch.FloatTensor(weights).to(device)
                y=y.long()
                loss = F.nll_loss(pred, y, weight=weights)

                loss.backward()                     # Compute gradient(backpropagation).
                optimizer.step()                    # Update parameters.
                step += 1
                
                # print(model.state_dict())
                # Display current epoch number and loss on tqdm progress bar.
                train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})
        model.eval()
        for data,y in test_loader:
            out = model(data)
            pred = out.max(dim=1)[1]
            preds=pred.cpu().long().numpy()
            labels=torch.squeeze(y).cpu()
            labels=labels.long().numpy()
            y_proba=out.cpu().detach().numpy()

        for idx in range(NameXtest.shape[0]):
            # print(name[idx],preds[idx])
            Name[NameXtest[idx]].append(preds[idx])

        y_real.append(labels)
        y_pred.append(preds)
        y_proba_minority.append(y_proba)
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

print_to_file("record.txt","mlp 10 cv Precision:%.3f Recall:%.3f F1_best:%.3f F1_best(std): %.3f ROC:%.3f ROC(std): %.3f PRC:%.3f ACC:%.3f for Class Fellow" % (np.mean(Precision),np.mean(Recall),np.mean(F1_best),np.std(F1_best),np.mean(ROC),np.std(ROC),np.mean(PRC),np.mean(ACC)))
 
print("100 cv Precision, Recall, F1_best, ROC, PRC, ACC for Class:",np.mean(Precision),np.mean(Recall),np.mean(F1_best),np.mean(ROC),np.mean(PRC),np.mean(ACC))

