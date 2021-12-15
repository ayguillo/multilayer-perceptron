import numpy as np

def train_test_split(X, y, split_size=0.33) :
    X_train = X.sample(frac=split_size)
    y_train = y.loc[X_train.index]
    X_test = X.drop(labels=X_train.index)
    y_test = y.drop(labels=y_train.index)
    return(X_train, y_train, X_test, y_test)

def tranform_pandas_to_numpy(x_train, y_train, x_test, y_test):
    x_train_numpy = x_train.to_numpy()
    y_train_numpy = y_train.to_numpy()
    x_test_numpy = x_test.to_numpy()
    y_test_numpy = y_test.to_numpy()
    return(x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy)

def initialize(X) :
    W = np.zeros((X.shape[1], 1))
    b = np.random.randn(1)
    return (W, b)