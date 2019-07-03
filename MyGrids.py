import click, os, time
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,cross_val_score,GroupKFold
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator,ClassifierMixin

from xgboost import XGBClassifier


class LgbmHelper():
    def __init__(self,params):
        self.estimator = None
        self.parametros = params
        
    def fit(self,X,Y):
        d_train = lgb.Dataset(X, label = Y)
        self.estimator = lgb.train(self.parametros, d_train, 100)
        return self

    def predict_proba(self,X):
        return self.estimator.predict(X)
    


class MyGridRF():
    def __init__(self,model,era):
        self.estimator = model
        self.parametros = {'n_estimators': [40],
                            'max_depth':[2],
                            'max_features':[5],
                            'min_samples_split':[2],
                            'min_samples_leaf': [300]}
        self.eras = era
    
    def fit(self,X,Y):
        gkf = GroupKFold(n_splits=5)
        kfold_split = gkf.split(X, Y, groups=self.eras)
        grid = GridSearchCV(estimator=self.estimator, param_grid=self.parametros, cv=kfold_split, scoring='neg_log_loss',n_jobs=1, verbose=3)
        grid_result = grid.fit(X.values, Y.values)
        self.estimator = grid_result.best_estimator_
        return self
    
    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)[:, 1]
        return proba
    


class MyGridLG():
    def __init__(self,model,era):
        self.estimator = model
        self.parametros = {'C':[0.001],
                            'penalty':['l2']}
        self.eras = era
    
    def fit(self,X,Y):
        gkf = GroupKFold(n_splits=5)
        kfold_split = gkf.split(X, Y, groups=self.eras)
        grid = GridSearchCV(estimator=self.estimator, param_grid=self.parametros, cv=kfold_split, scoring='neg_log_loss',n_jobs=1, verbose=3)
        grid_result = grid.fit(X.values, Y.values)
        self.estimator = grid_result.best_estimator_
        return self
    
    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)[:, 1]
        return proba

class LgAndPcaHelper():
    def __init__(self,model):
        self.estimator = model
        self.pca_estimator = PCA(n_components=2,random_state=123)
        
    def Pca_dataset(self,X,y=None,dataset = None):
        if dataset == 'train':
            self.pca_estimator = self.pca_estimator.fit(X,Y)
            
        components = self.pca_estimator.transform(X)
        return pd.concat([X,pd.DataFrame(components,columns=['feature_PCA_1','feature_PCA_2'])],axis=1)
        
        
    def fit(self,X,Y):
        dataset = self.Pca_dataset(X,Y,dataset='train')
        self.estimator = self.estimator.fit(dataset,Y)
        return self
    
    def predict_proba(self, X):
        X = self.Pca_dataset(X)
        proba = self.estimator.predict_proba(X)[:, 1]
        return proba