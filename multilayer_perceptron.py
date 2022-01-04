import pandas as pd
from tools.Preprocessing import train_test_split, tranform_pandas_to_numpy, initialize
from model.artificialNeuron import ArtificialNeuron, Layer
# from sklearn.metrics import accuracy_score
import argparse
import os, shutil

def args_parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="define your file", type = str)
    parser.add_argument("-v","--visu", help="plot loss", action="store_true")
    args = parser.parse_args()
    return(args)

def main() :
    args = args_parse()
    df = pd.read_csv(args.file, header=None)
    X = df.loc[:, df.columns.drop([0,1])]
    y = df[df.columns[1]]
    y = y.map(dict(M=1, B=0))
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    if os.path.isdir('data') is False :
        os.mkdir('data')
    else :
        shutil.rmtree('data/')
    X_train, y_train, X_test, y_test = tranform_pandas_to_numpy(X_train, y_train, X_test, y_test)
    neural = ArtificialNeuron(X_train,
                              y_train,
                              X_test,
                              y_test,
                              layers=[Layer(X_train.shape[1], 20),
                                     Layer(20, 10),
                                     Layer(10, 5),
                                     Layer(5, 2)])
    # neural.fit()
    # if args.visu :
    #     neural.plot_loss()
    # y_pred = neural.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    
if __name__ == "__main__" :
    main()