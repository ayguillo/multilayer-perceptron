import numpy as np
import matplotlib.pyplot as plt

class ArtificialNeuron :
    def __init__(self, X_train, y_train, learning_rate = 0.01, n_iter = 1000) :
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.loss = []
        
    def __initialize(self) :
        W = np.zeros((self.X_train.shape[1], 1))
        b = np.random.randn(1)
        return (W, b)
    
    def __model(self, W, b, X) :
        Z = X.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def __log_loss(self, A) :
        return(1/len(self.y_train) * np.sum(-self.y_train * np.log(A) - (1 - self.y_train) * np.log(1 - A)))

    def __gradients(self, A) :
        dW = 1 / len(self.y_train) * np.dot(self.X_train.T, A - self.y_train)
        db = 1 / len(self.y_train) * np.sum(A - self.y_train)
        return (dW, db)
    
    def __update(self, dW, db, W, b) :
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        return(W, b)
    
    def fit(self) :
        self.X_train = self.X_train / self.X_train.max(axis=0)
        W, b = self.__initialize()
        for i in range(self.n_iter) :
            activation = self.__model(W, b, self.X_train)
            loss = self.__log_loss(activation)
            self.loss.append(loss)
            dW, db = self.__gradients(activation)
            W, b = self.__update(dW, db, W,b)
            
    def plot_loss(self) :
        plt.plot(self.loss)
        plt.show()
        
    def predict(self) :
        activation = self.__model()