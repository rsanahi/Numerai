# bag_base_models.py
from sklearn import decomposition
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import click
import datetime
import numpy as np
import os
import pandas as pd
import time
import round_targets
from utils.model_selection import KFoldEra

# defining function to compute consistency
def compute_consistency(val, target, yhat, upper_bound=0.693):
    errors = []
    new_val = val
    val_eras = new_val.era.unique().tolist()
    y_val = val[target].values
    for era in val_eras:
        idx = (new_val.era == era).values
        errors.append(metrics.log_loss(y_val[idx], yhat[idx]))

    errors = pd.DataFrame({'era': val_eras, 'error': errors})
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


def print_statistics(loss, auc, c, dataset="val"):
    s = f"Error in {dataset} -> (loss, auc, consistency): ({loss:.6f},{auc:0.6f}, {c:.6f})"
    print(s)

def load_data(path, n_round):
    """ read data """
    train_filename = os.path.join(
        path, "numerai_datasets", "numerai_training_data.csv")
    test_filename = os.path.join(
        path, "numerai_datasets", "numerai_training_data.csv")
    print("Reading normal data")
    df_train = pd.read_csv(train_filename)
    df_test = pd.read_csv(test_filename)
    return df_train, df_test


def train_predict(n_round, target, inv_lambda=1e-2):
    # load data
    # title used to identify the prediction of the output file
    print(f"Working with round:{n_round} and predicting target:{target}")
    PATH = f"../raw_data/round{n_round}/"

    raw_train, raw_test = load_data(PATH, n_round)

    continous_features = [
        f for f in raw_train.columns.tolist() if f.startswith("feature")]
    # selecting training, val, and test sets.
    train = raw_train.copy()
    # Selecting here target to predict.
    features = continous_features

    kf = KFoldEra(n_splits=4, shuffle=False, seed=123)
    folds = kf.split(train)
    for k, (train_index, val_index) in enumerate(folds):
        print(f"Fitting model on fold:{k}")
        X_train, y_train = train.iloc[train_index][features], train.iloc[train_index][target]
        X_val, y_val = train.iloc[val_index][features], train.iloc[val_index][target]
        model = LogisticRegression(
            penalty="l2", C=inv_lambda, n_jobs=-1, verbose=0)
        print('Fitting single model...')
        start_time = time.time()
        model.fit(X_train, y_train)
        print('Fit: {}s'.format(time.time() - start_time))

        yhat_train = model.predict_proba(X_train)
        yhat_val = model.predict_proba(X_val)

        n_eras_train = train.iloc[train_index]["era"].unique().shape[0]
        n_era_val = train.iloc[val_index]["era"].unique().shape[0]
        print(f"Training in:{n_eras_train} eras, validatiing in:{n_era_val} eras")
        error_train, auc_train, consistency_train, _ = compute_statistics(
            train.iloc[train_index, :], target, yhat_train[:, 1])
        error_val, auc_val, consistency_val, _ = compute_statistics(
            train.iloc[val_index, :], target, yhat_val[:, 1])
        print_statistics(error_train, auc_train,
                        consistency_train, dataset="train")
        print_statistics(error_val, auc_val, consistency_val, dataset="val")


@click.command()
@click.argument("n_round")
def main(n_round):
    # add here the command line arguments.
    #ltarget = round_targets.ltarget
    ltarget = ["target_bernie"]
    for target in ltarget:
        train_predict(n_round, target)


if __name__ == "__main__":
    main()