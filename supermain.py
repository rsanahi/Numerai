import click, os, time
import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from MyGrids import MyGridLG, MyGridRF, LgbmHelper

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
np.random.seed(42)


def print_statistics(loss, auc, c, dataset="val"):
    s = f"Error in {dataset} -> (loss, auc, consistency): ({loss:.6f},{auc:0.6f}, {c:.6f})"
    print(s)

def choice(size_test,train):
    tst = []
    while len(tst) < size_test:
        x = np.random.choice(train.era)
        if not x in tst:
            tst.append(x)
    return tst

def split_v2(train, size_test=40):
    #tst = np.random.choice(train.era,size_test)
    tst = choice(size_test,train)
    tr = [x for x in train.era.unique() if not x in tst]
    #print("Eras test: ",tst)
    #print("Eras train ", tr)
    test = train[train.era.isin(tst)]
    trainn = train[train.era.isin(tr)]
    return trainn,test

def compute_consistency(val, target, yhat, upper_bound=0.693):
    errors = []
    auc = []
    new_val = val
    val_eras = new_val.era.unique().tolist()
    y_val = val[target].values
    for era in val_eras:
        idx = (new_val.era == era).values
        errors.append(metrics.log_loss(y_val[idx], yhat[idx]))
        auc.append(metrics.roc_auc_score(y_val[idx], yhat[idx]))

    errors = pd.DataFrame({'era': val_eras, 'error': errors,'auc':auc})
    errors['upper'] = upper_bound
    errors['consistency'] = errors.error < errors.upper
    return errors

def compute_statistics(df, target, yhat):
    """ compute logloss, auc, and consistency between variables"""
    y = df[target].values
    loss = metrics.log_loss(y, yhat)
    auc = metrics.roc_auc_score(y, yhat)
    errors = compute_consistency(df, target, yhat)
    consistency = errors.consistency.mean()
    # returning errors in case you want to plot
    return loss, auc, consistency, errors

def make_plot(plot_info,targets,PATH):
    fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, figsize = (15,10))
    fig.autofmt_xdate(rotation = 45)
    names = [t.split('target_')[1] for t in targets]
    x = [x+1 for x in range(len(plot_info[4][0]))]
    
    print('Making logloss plot')
    p1 = ax1.plot(names,plot_info[0],color='y',marker='o',label='train')
    p2 = ax1.plot(names,plot_info[1],color='b',marker='o',label='valid')
    ax1.legend()
    ax1.set_title('- Targets Logloss')

    print('Making consistency plot')
    
    p4 = ax2.plot(names,plot_info[2],color='b',marker='o')
    ax2.set_title('- Targets Consistency')

    fig.savefig(f'{PATH}/submission_results')
    fig.show()

    print('Making logloss for era plot')

    fig, ((ax3)) = plt.subplots(nrows=1, ncols=1, figsize = (18,13))
    fig.autofmt_xdate(rotation = 45)
    eras = ['era121','era122','era123','era124','era125','era126','era127','era128','era129','era130','era131','era132']

    p5 = ax3.plot(eras,[0.501 for x in range(len(eras))],color='y')
    p6 = ax3.plot(eras,plot_info[4][0].auc,marker='o',label=names[0])
    p7 = ax3.plot(eras,plot_info[4][1].auc,marker='o',label=names[1])
    p8 = ax3.plot(eras,plot_info[4][2].auc,marker='o',label=names[2])
    p9 = ax3.plot(eras,plot_info[4][3].auc,marker='o',label=names[3])
    p10 = ax3.plot(eras,plot_info[4][4].auc,marker='o',label=names[4])
    p11 = ax3.plot(eras,plot_info[4][5].auc,marker='o',label=names[5])
    p12 = ax3.plot(eras,plot_info[4][6].auc,marker='o',label=names[6])
    p13 = ax3.plot(eras,[0.51 for x in range(len(eras))],color='r')

    fig.legend()
    fig.savefig(f'{PATH}/submission_results_era')
    fig.show()
    
    return True

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers
        self.vote= vote
        self.weights = weights
        
    def fit(self, X,Y):
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clf.fit(X,Y)
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self,X):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
        return maj_vote
    
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas,axis=0, weights=self.weights)
        
        return avg_proba


def make_predictions(n_round,PATH):
    """
        lt = []
        ltt = []
        lv = []
        cv = []
        ceras = []
    """
    print(f"Cargando train.csv round {n_round}")
    train = pd.read_csv(f'../../raw_data/round {n_round}/numerai_training_data.csv', header=0)
    print(f'Cargando tournament.csv round {n_round}')
    tournament = pd.read_csv(f'../../raw_data/round {n_round}/numerai_tournament_data.csv',header = 0)
    print(train.shape,tournament.shape)
    
    plot_info = [[],[],[],[],[]]
    submission_info = {}
    
    ids = tournament['id']
    targets = [f for f in list(train) if 'target' in f]
    validation = tournament[tournament.data_type == 'validation']
    features = [f for f in list(train) if 'feature' in f]
    x_prediction = tournament[features]
    x_valid = validation[features]
    X = train[features]
    params = {'bagginf_freq':5,
            'bagging_fraction':0.9,
            'boost_from_average':'false',
            'colsample_bytree':0.12,
            'boost':'gbdt',
            'learning_rate': 0.05,
            'min_data_in_leaf':5,
            'min_sum_hessian_in_leaf':1.0,
            'num_leaves': 9,
            'num_threads':4,
            'metric':'auc',
            'feature_fraction': 1.0,
            'max_depth': -1,
            'lambda_l2':0.001,
            'objective': 'binary',
            'verbosity': 0}
    
    train_, test_ = split_v2(train)
    m1 = LogisticRegression(n_jobs=1,random_state=42)
    lg =  MyGridLG(m1,train.era)
    m2 = RandomForestClassifier(n_jobs=-1,random_state=42)
    rf = MyGridRF(m2,train.era)
    lb = LgbmHelper(params)
    mv_clf = MajorityVoteClassifier(classifiers=[lg,rf,lb],weights=[0.20,0.40,0.40])


    for target in targets:
        print(f'- {target.upper()}')
        train_features,train_labels = train_[features],train_[target]
        test_features,test_labels = test_[features], test_[target]
        y_valid = validation[target]
        Y=train[target]

        print('# Training')
        mv_clf.fit(X,Y)

        print('# Computing logloss')
        error_train, auc_train, consistency_train, _ = compute_statistics(train,'target_bernie',mv_clf.predict_proba(train[features]))
        error_valid, auc_vali, consistency_valid, _va = compute_statistics(validation,'target_bernie',mv_clf.predict_proba(validation[features]))

        plot_info[0].append(error_train)
        plot_info[1].append(error_valid)
        plot_info[2].append(consistency_valid)

        plot_info[3].append(_va)

        print_statistics(error_valid, auc_vali, consistency_valid, dataset="valid")

        print("# Predicting")
        y_prediction = mv_clf.predict_proba(x_prediction)
        #results = y_prediction[:, 1]

        print("# Creating submission...")
        name  = target.split('target_')[1]
        results_df = pd.DataFrame(data={f'probability_{name}':y_prediction})
        joined = pd.DataFrame(ids).join(results_df)
        
        print(f"# Writing predictions to {target}_submissions.csv...")
        joined.to_csv(f"{PATH}/{name}_submission.csv", index=False)
        
        submission_info.setdefault(target, {'status': 'waiting', 'root': name, 'account': 'LAGERTHA'
                                                           , 'resumen': [] })

    np.save(f'{PATH}/resumen_info.npy',submission_info)
    
    mp = make_plot(plot_info, targets, PATH)
    assert mp == True, 'Plots Done'
    
    return 'Submissions Done!'

@click.command()
@click.option("--n_round", help="Number of round.")

def tournament_(n_round):
    i = time.time()
    print(f'Making Submission of round {n_round}')
    PATH = f'../submission/round {n_round}/SE'
    os.makedirs(exist_ok=True, name=PATH)
    result = make_predictions(n_round,PATH)
    f = time.time()
    print(f'{result}, time: {(f-i)/60} min')

if __name__ == '__main__':
        tournament_()