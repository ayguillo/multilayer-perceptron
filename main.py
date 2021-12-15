import pandas as pd
from tools.Preprocessing import train_test_split, tranform_pandas_to_numpy, initialize
from model.model import ArtificialNeuron

    
# def train_test_split(X, y) :
#     train_df =

if __name__ == "__main__" :
    df = pd.read_csv("data.csv", header=None)
    X = df.loc[:, df.columns.drop(1)]
    y = df[df.columns[1]]
    y = y.map(dict(M=1, B=0))
    print(X, y)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train, y_train, X_test, y_test = tranform_pandas_to_numpy(X_train, y_train, X_test, y_test)
    neural = ArtificialNeuron(X_train, y_train)
    neural.fit()
    neural.plot_loss()