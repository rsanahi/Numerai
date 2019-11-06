import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import tensorflow
from pylab import rcParams
from sklearn.decomposition import PCA, non_negative_factorization
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import KFold

tensorflow.random.set_seed(1234)

def load_data(n_round, save=True, with_preprosessing=False):
    print("Reading normal data")
    data_train_name = "numerai_training_data.csv"
    data_tournament_name = "numerai_tournament_data.csv"

    if save: 
        print('Loading train data')
        df_train = pd.read_csv(f'../../../raw_data/round {n_round}/{data_train_name}', header=0)
        features = [c for c in df_train if c.startswith("feature")]
        save_memo(df_train, features)
        print('Loading tournament data')
        df_test = pd.read_csv(f'../../../raw_data/round {n_round}/{data_tournament_name}',header = 0)
        save_memo(df_test, features)
    return df_train, df_test, features

def load_proses_data(n_round, feather=True, preprocessing=False):
    data_train_name = "numerai_training_preprosessing_data.csv"
    data_tournament_name = "numerai_tournament_preprosessing_data.csv"
    if feather:
        print("Reading Normal(feather) Data")
        df = pd.read_feather(f"../../../raw_data/round {n_round}/train-tmp")
        tournament = pd.read_feather(f"../../../raw_data/round {n_round}/tournament-tmp")
        features = [c for c in df if c.startswith("feature")]
        return df, tournament, features
    
    elif preprocessing:
        print("Reading Proses data")
        df = pd.read_csv(f'../../../raw_data/round {n_round}/{data_train_name}', header=0)
        tournament = pd.read_csv(f'../../../raw_data/round {n_round}/{data_tournament_name}',header = 0)
        features = [c for c in df if c.startswith("feature")]
        return df, tournament, features

def save_memo(data,features):
    for column in features:
        data[column] = data[column].astype(np.dtype('float32'))
    return data

def choice(size_test,eras):
    tst = []
    while len(tst) < size_test:
        x = np.random.choice(eras)
        if not x in tst:
            tst.append(x)
    return tst

def split_v2(train, verbose, size_test=40):
    #tst = np.random.choice(train.era,size_test)
    tst = choice(size_test,train.era.unique())
    tr = [x for x in train.era.unique() if not x in tst]
    if verbose:
        print("Eras test: ",tst)
        print("Eras train ", tr)
    test = train[train.era.isin(tst)]
    trainn = train[train.era.isin(tr)]
    return trainn,test,tst,tr

def plot_history(history, keys=['loss'], title='',path='/'):
    n = len(keys)
    epochs = [i for i in range(len(history['loss']))]
    plt.figure(figsize=(15, 6))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.plot(epochs, history[keys[i]], label=keys[i])
        plt.plot(epochs, history[f'val_{keys[i]}'], label=f'val_{keys[i]}')
        plt.legend()
        ax.set_title(f'{keys[i]}')
    plt.savefig(f'{path}/{title}history')
    plt.show()

def numerai_score(y_true, y_pred, eras):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def score(y_true,y_pred):
    # method="first" breaks ties based on order in array
    return np.corrcoef(y_true,y_pred.rank(pct=True, method="first"))[0,1]

def basic_plot(x,xlabel='x',ylabel='y',title='basic plot', margin=[0.02],save=False,path='/'):
    y = [n for n in range(len(x))]
    
    for m in margin:
        plt.plot(y, [m for x in y])
    plt.plot(y,x,marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(f'{path}/{title[0]}_history.png')
    plt.show()

def check_correlation_consistency(model,valid_data, metric, features, target, verbose=0):
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    metric_val = []
    for era in eras:
        count += 1
        current_valid_data = valid_data[valid_data.era==era]
        X_valid = current_valid_data[features]
        Y_valid = current_valid_data[target]
        y_prediction = model.predict(X_valid)
        probabilities = y_prediction
        m = metric(Y_valid, probabilities)
        metric_val.append(m)
        if (m > 0.02):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        if verbose:
            print(str(era),": loss - "+str(m), "consistent: "+str(consistent))
    print (f"Consistency {count_consistent}/{count}: {count_consistent/count}")
    print(f'{count_consistent} de {count}')
    return metric_val,(count_consistent/count)

def PCA_preprosessing(X, feature_groups, features, pca=None):
    if pca:
        pca_ = pca
        data = pca_.transform(X[features])
    else:
        pca_ = PCA(n_components=2)
        pca_ = pca_.fit(X[features])
        data = pca_.transform(X[features])
    all_components = pd.DataFrame(data)
    all_components = all_components.add_prefix('feature_PCA_')
    for group in feature_groups:
        pca = PCA(n_components=2)
        print(group)
        components = pd.DataFrame(pca.fit_transform(X[feature_groups[group]]))
        components = components.add_prefix(f'{group}_')
        all_components = pd.concat([all_components, components],axis=1)
    
    return all_components, pca_
