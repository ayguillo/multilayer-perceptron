import numpy as np
import matplotlib.pyplot as plt

class ArtificialNeuron :
    def __init__(self, X_train, y_train, X_test, y_test, learning_rate = 0.01, n_iter = 5000) :
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.reshape((y_train.shape[0], 1))
        self.y_test = y_test.reshape((y_test.shape[0], 1))
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.train_loss = []
        self.test_loss = []
        self.mean = np.mean(self.X_train , axis=0)
        self.std = np.std(self.X_train, axis=0)
        self.max = self.X_train.max(axis=0)
        self.min = self.X_train.min(axis=0)
        
    def __initialize(self) :
        W = np.random.randn(self.X_train.shape[1], 1)
        b = np.random.randn(1)
        return (W, b)
    
    def __model(self, W, b, X) :
        Z = X.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def __log_loss(self, A, y) :
        epsilon = 1e-15 #Eviter le 0 dans le log
        return((1/len(y)) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))

    def __gradients(self, A) :
        dW = 1 / len(self.y_train) * np.dot(self.X_train.T, A - self.y_train)
        db = 1 / len(self.y_train) * np.sum(A - self.y_train)
        return (dW, db)
    
    def __update(self, dW, db, W, b) :
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        return(W, b)
    
    def fit(self) :
        W, b = self.__initialize()
        self.X_train  = (self.X_train  - self.mean) / self.std
        self.X_test  = (self.X_test  - self.mean) / self.std
        # self.X_train = (self.X_train - self.min) / (self.max -self.min)
        activation = self.__model(W, b, self.X_train)
        for i in range(self.n_iter) :
            #Train loop
            activation = self.__model(W, b, self.X_train)
            if i % 10 == 0 :
                # loss in train set
                loss = self.__log_loss(activation, self.y_train)
                self.train_loss.append(loss)
                #loss in test set
                activation_test = self.__model(W, b, self.X_test)
                loss_test = self.__log_loss(activation_test, self.y_test)
                self.test_loss.append(loss_test)
                
            dW, db = self.__gradients(activation)
            W, b = self.__update(dW, db, W,b)
        self.W, self.b = W, b
            
    def plot_loss(self) :
        plt.plot(self.train_loss, label="train loss")
        plt.plot(self.test_loss, label="test loss")
        plt.legend()
        plt.show()
        
    def predict(self, X_test) :
        X_test  = (X_test  - self.mean) / self.std
        activation = self.__model(self.W, self.b, X_test)
        return activation >= 0.5