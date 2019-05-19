import click, os, time
import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss, roc_auc_score
from sklearn import metrics

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
np.random.seed(42)


class MYLGBM():

    def __init__(self):
        self.params = {
            'bagginf_freq':5,
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
    
    def get_parametros(self):
        return self.params

    def dataset(self, train_features,train_labels):
        return lgbm.Dataset(train_features, label = train_labels)
    
    def fit(self,params,d_train):
        return lgbm.train(params, d_train, 100)
    

def choice(train,size_test):
    tst = []
    while len(tst) < size_test:
        x = np.random.choice(train.era)
        if not x in tst:
            tst.append(x)
    return tst

def split_v2(train, size_test=40):
    #tst = np.random.choice(train.era,size_test)
    tst = choice(train,size_test)
    print(tst)
    tr = [x for x in train.era.unique() if not x in tst]
    #print("Eras test: ",tst)
    #print("Eras train ", tr)
    test = train[train.era.isin(tst)]
    trainn = train[train.era.isin(tr)]
    return trainn,test

def check_consistency(model,valid_data,featu,target):
    loglossssssss = []
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        current_valid_data = valid_data[valid_data.era==era]
        X_valid = current_valid_data[featu]
        Y_valid = current_valid_data[target]
        y_prediction = model.predict(X_valid)
        #probabilities = y_prediction[:, 1]
        loss = log_loss(Y_valid, y_prediction)
        loglossssssss.append(loss)
        if (loss < 0.693):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        #print(str(era),": loss - "+str(loss), "consistent: "+str(consistent))
    print ("Consistency: "+str(count_consistent/count))
    return loglossssssss,(count_consistent/count)

def compute_consistency(val, target, yhat, upper_bound=0.693):
    """ compute the logloss and auc for eras """
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
    """ compute logloss, auc, and consistency between variables """
    y = df[target].values
    loss = log_loss(y, yhat)
    auc = roc_auc_score(y, yhat)
    errors = compute_consistency(df, target, yhat)
    consistency = errors.consistency.mean()
    # returning errors in case you want to plot
    return loss, auc, consistency, errors

def print_statistics(loss, auc, c, dataset="val"):
    s = f"Error in {dataset} -> (loss, auc, consistency): ({loss:.6f},{auc:0.6f}, {c:.6f})"
    print(s)

def make_plot(plot_info,targets,PATH):
    fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, figsize = (15,10))
    fig.autofmt_xdate(rotation = 45)
    names = [t.split('target_')[1] for t in targets]
    x = [x+1 for x in range(len(plot_info[4][0]))]
    
    print('Making logloss plot')
    p1 = ax1.plot(names,plot_info[0],color='y',marker='o',label='train')
    p2 = ax1.plot(names,plot_info[1],color='b',marker='o',label='test')
    p3 = ax1.plot(names,plot_info[2],color='g',marker='o',label='valid')
    ax1.legend()
    ax1.set_title('- Targets Logloss')

    print('Making consistency plot')
    
    p4 = ax2.plot(names,plot_info[3],color='b',marker='o')
    ax2.set_title('- Targets Consistency')

    fig.savefig(f'{PATH}/submission_results')
    print('Making logloss for era plot')

    fig, ((ax3)) = plt.subplots(nrows=1, ncols=1, figsize = (18,13))
    fig.autofmt_xdate(rotation = 45)
    eras = ['era121','era122','era123','era124','era125','era126','era127','era128','era129','era130','era131','era132']

    p5 = ax3.plot(eras,[0.6935 for x in range(len(eras))],color='y')
    p6 = ax3.plot(eras,plot_info[4][0],marker='o',label=names[0])
    p7 = ax3.plot(eras,plot_info[4][1],marker='o',label=names[1])
    p8 = ax3.plot(eras,plot_info[4][2],marker='o',label=names[2])
    p9 = ax3.plot(eras,plot_info[4][3],marker='o',label=names[3])
    p10 = ax3.plot(eras,plot_info[4][4],marker='o',label=names[4])
    p11 = ax3.plot(eras,plot_info[4][5],marker='o',label=names[5])
    p12 = ax3.plot(eras,plot_info[4][6],marker='o',label=names[6])
    p13 = ax3.plot(eras,[0.6930 for x in range(len(eras))],color='r')

    fig.legend()
    fig.savefig(f'{PATH}/submission_results_era')
    fig.show()
    
    return True


def make_predictions(roundd,PATH):
    """
        lt = []
        ltt = []
        lv = []
        cv = []
        ceras = []
    """
    print(f"Cargando train.csv round {roundd}")
    train = pd.read_csv(f'../../raw_data/round {roundd}/numerai_training_data.csv', header=0)
    print(f'Cargando tournament.csv round {roundd}')
    tournament = pd.read_csv(f'../../raw_data/round {roundd}/numerai_tournament_data.csv',header = 0)
    print(train.shape,tournament.shape)
    
    plot_info = [[],[],[],[],[]]
    submission_info = {}
    
    ids = tournament['id']
    targets = [f for f in list(train) if 'target' in f]
    validation = tournament[tournament.data_type == 'validation']
    features = [f for f in list(train) if 'feature' in f]
    x_prediction = tournament[features]
    X = train[features]
    
    train_, test_ = split_v2(train)
    lgb = MYLGBM() 
    
    for target in targets:
        print(f'- {target.upper()}')
        train_features,train_labels = train_[features],train_[target]
        test_features,test_labels = test_[features], test_[target]
        y_valid = validation[target]
        Y=train[target]
        
        params = lgb.get_parametros()
        d_train = lgb.dataset(X,Y)
        
        print('# Training')
        clf = lgb.fit(params, d_train)
        
        print('Computing logloss')
        lg_train = log_loss(train_labels,clf.predict(train_features))
        lg_test = log_loss(test_labels,clf.predict(test_features))
        lg_valid = log_loss(y_valid, clf.predict(validation[features]))

        error_valid, auc_valid, consistency_valid, _ = compute_statistics(
            validation,target,clf.predict(validation[features]))
        
        print_statistics(error_valid, auc_valid, consistency_valid, dataset="valid")
        
        print('# Checking consistency')
        ceras,cv= check_consistency(clf,validation,features,target)
        
        plot_info[0].append(lg_train)
        plot_info[1].append(lg_test)
        plot_info[2].append(lg_valid)
        plot_info[3].append(cv)
        plot_info[4].append(ceras)
        
        print("# Predicting")
        y_prediction = clf.predict(x_prediction)
        #results = y_prediction[:, 1]
        
        print("# Creating submission...")
        name  = target.split('target_')[1]
        results_df = pd.DataFrame(data={f'probability_{name}':y_prediction})
        joined = pd.DataFrame(ids).join(results_df)
        
        print(f"# Writing predictions to {target}_submissions.csv...")
        joined.to_csv(f"{PATH}/{name}_submission_lgbm.csv", index=False)
        
        submission_info.setdefault(target, {'status': 'waiting', 'root': name, 'account': 'LAGERTHA'
                                                           , 'resumen': [] })
    
    np.save(f'{PATH}/resumen_info.npy',submission_info)
    
    mp = make_plot(plot_info, targets, PATH)
    assert mp == True, 'Plots Done'
    
    return 'Submissions Done!'

@click.command()
@click.option("--n_round", help="Number of round.")

def tournament_(n_round):
        print(f'hola {n_round}')
        PATH = f'../submission/round {n_round}/LGBM'
        os.makedirs(exist_ok=True, name=PATH)
        result = make_predictions(n_round,PATH)
        print(result)

if __name__ == '__main__':
        tournament_()