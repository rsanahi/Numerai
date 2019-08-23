import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)


def load_data(n_round, save=True):
    print("Reading normal data")
    if save: 
        print('Loading train data')
        df_train = pd.read_csv(f'../../../raw_data/round {n_round}/numerai_training_data.csv', header=0)
        features = [c for c in df_train if c.startswith("feature")]
        save_memo(df_train, features)
        print('Loading tournament data')
        df_test = pd.read_csv(f'../../../raw_data/round {n_round}/numerai_tournament_data.csv',header = 0)
        save_memo(df_test, features)
    return df_train, df_test, features

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
