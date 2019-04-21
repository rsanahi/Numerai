import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from model_selection import KFoldEra

import time, click, warnings

plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}
warnings.filterwarnings('ignore')

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

def load_data( n_round):
    print("Reading normal data")
    df_train = pd.read_csv(f'../../raw_data/round {n_round}/numerai_training_data.csv', header=0)
    df_test = pd.read_csv(f'../../raw_data/round {n_round}/numerai_tournament_data.csv',header = 0)
    return df_train, df_test

def train_predict(PATH,n_round):
    """ Make predictions for target """
    print(f"Working with round: {n_round}")
    raw_train, raw_test = load_data(n_round)
    validation = raw_test[raw_test.data_type == 'validation']
    ids = raw_test['id']
    targets = [f for f in list(raw_train) if 'target' in f]
    features = [f for f in list(raw_train) if 'feature' in f]
    model = LogisticRegression(C=0.001, penalty='l2')

    pca = PCA(n_components=2,random_state=123)

    for target in targets:
        components = pca.fit_transform(raw_train[features],raw_train[target])
        components_valid = pca.transform(validation[features])
        components = pd.DataFrame(components,columns=['feature_PCA_1','feature_PCA_2'])
        components_valid = pd.DataFrame(components_valid,columns=['feature_PCA_1','feature_PCA_2'])

        df = pd.concat([raw_train[features],components],axis=1)
        df_valid = pd.concat([validation,components_valid],axis=1)

        features2 = [f for f in list(df) if 'feature' in f]

        print('Fitting single model...')
        start_time = time.time()
        model.fit(df[features2],raw_train[target])
        print('Fit: {}s'.format(time.time() - start_time))

        compute_statistics(df_valid,target,model.predict_proba(df_valid[features2])[:,1])

        components_tournament = pca.fit_transform(raw_test[features],raw_test[target])
        components_tournament = pd.DataFrame(components_tournament,columns=['feature_PCA_1','feature_PCA_2'])
        tournament = pd.concat([raw_test,components_tournament], axis=1)

        yhat = model.predict_proba(tournament[features2])[:,1]

        print("# Creating submission...")
        name  = target.split('target_')[1]
        results_df = pd.DataFrame(data={f'probability_{name}':yhat})
        joined = pd.DataFrame(ids).join(results_df)

        print(f"# Writing predictions to {target}_submissions.csv...")
        joined.to_csv(f"{PATH}/{name}_submission.csv", index=False)

    return 'Submissions Done!'

@click.command()
@click.argument("n_round")

def main(n_round):
    PATH = f'../submission/round {round}/PCA'
    result = train_predict(PATH,n_round)


if __name__ == "__main__":
    main()