import pandas as pd
import numpy as np


def load_ecg_data() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    mitbih_train = pd.read_csv('./datasets/mitbih_train.csv', header=None).to_numpy()[:, :-1]
    mitbih_test = pd.read_csv('./datasets/mitbih_test.csv', header=None).to_numpy()[:, :-1]
    ptbdb_normal = pd.read_csv('./datasets/ptbdb_normal.csv', header=None).to_numpy()[:, :-1]
    ptbdb_abnormal = pd.read_csv('./datasets/ptbdb_abnormal.csv', header=None).to_numpy()[:, :-1]
    return mitbih_train, mitbih_test, ptbdb_normal, ptbdb_abnormal


