"""
Utilities function to split data set into eras. 
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold


def train_test_split(data: pd.DataFrame, test_size: float = 0.2, seed: int = 123):
    """ 
    Split arrays or matrices into random train and test subsets
    """
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(n_samples)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    return data.iloc[train_indices, :], data.iloc[test_indices, :]


def test_train_test_split_era():
    """ testing train_test_split_era """
    data_path = "../../raw_data/round152/numerai_datasets/"
    filename = "numerai_training_data.csv"
    path_filename = os.path.join(data_path, filename)
    df = pd.read_csv(path_filename)

    n_eras = df["era"].unique().shape[0]
    test_size = 0.3
    train, test = train_test_split_era(df, test_size=test_size)
    eras_train = train["era"].unique()
    eras_test = test["era"].unique()
    # testing intersection and eras' number
    assert set(eras_train) & set(eras_test) == set()
    assert len(eras_test) == int(n_eras * test_size)
    assert len(eras_train) == int(n_eras * (1-test_size))

    # testing col invariance
    assert train.shape[1] == df.shape[1]
    assert test.shape[1] == df.shape[1]


def train_test_split_era(data: pd.DataFrame, test_size: float = 0.2, seed: int = 123):
    """
    Split arrays or matrices into random train and test subsets
    taking into account the number of eras
    """
    unique_eras = data["era"].unique()
    n_samples = len(unique_eras)
    n_test = int(n_samples * test_size)
    np.random.seed(seed)
    shuffled_eras = np.random.permutation(unique_eras)
    test_eras = shuffled_eras[:n_test]
    train_eras = shuffled_eras[n_test:]
    train_indices = data["era"].isin(train_eras)
    return data.loc[train_indices, :], data.loc[~train_indices, :]


def test_KFoldEra():
    """ testing KFoldEra """
    data_path = "../../raw_data/round152/numerai_datasets/"
    filename = "numerai_training_data.csv"
    path_filename = os.path.join(data_path, filename)
    df = pd.read_csv(path_filename)

    n_eras = df["era"].unique().shape[0]
    n_splits = 10
    kf = KFoldEra(n_splits=n_splits)
    folds = kf.split(df)

    # testing number of folds
    assert len(folds) == n_splits

    # for every fold test conditions
    for train_index, test_index in folds:
        train, test = df.iloc[train_index], df.iloc[test_index]
        eras_train = train["era"].unique()
        eras_test = test["era"].unique()
        # testing intersection and eras' number
        assert set(eras_train) & set(eras_test) == set()
        assert len(eras_test) == int(n_eras // n_splits)
        assert len(eras_train) == int(n_eras // n_splits) * (n_splits - 1)
        # testing col invariance
        assert train.shape[1] == df.shape[1]
        assert test.shape[1] == df.shape[1]


class KFoldEra():
    """
    K-Folds cross-validator
    Provides train/test indices to split data in train/test sets using era
    Parameters
    ----------
    n_splits : int, default=3.
        Number of folds.
    shuffle : boolean, default=False
        Wheather to shuffle the data before splitting into batches. 
    seed : int, default=123
        seed used by the random number generator
    """

    def __init__(self, n_splits: int = 3, shuffle: bool = False, seed: int = 123):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, df, y=None, groups=None):
        """
        re-using KFold implementation from sklearn
        """
        unique_eras = df["era"].unique()
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                   random_state=self.seed)
        folds = []
        for train_index_era, test_index_era in kf.split(unique_eras):
            train_era = unique_eras[train_index_era]
            test_era = unique_eras[test_index_era]
            # finding indices in the data set
            train_index = df["era"].isin(train_era)
            test_index = df["era"].isin(test_era)
            train_idx = np.flatnonzero(train_index)
            test_idx = np.flatnonzero(test_index)
            folds.append((train_idx, test_idx))
        return folds
