import numpy as np
import matplotlib.pyplot as plt

class Layer :
    def __init__(self, input_size, output_size, activation="sigmoid") :
        self.activation_str = activation
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size)
        self.a = 1
        self.Z = 1
        self.dW = 1
        self.db = 1

    def activation(self, Z) :
        epsilon = 1e-15
        if self.activation_str == "sigmoid" :
            return(1 / (1 + np.exp(-Z)))
        
    def activation_back(self, dA, Z) :
        if self.activation_str == "sigmoid" :
            sig = 1 / (1 + np.exp(-Z))
            return(dA * sig * (1 - sig))
    
class ArtificialNeuron :
    def __init__(self, X_train, y_train, X_test, y_test, layers, learning_rate = 0.01, n_iter = 3000) :
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
        if len(layers) <= 3 :
            raise ValueError("Len layer <= 3")
        self.layers = layers
        
    def __initialize(self) :
        W = np.random.randn(self.X_train.shape[1], 1)
        b = np.random.randn(1)
        return (W, b)
    
    def __activation(self, W, b, X) :
        print(X.shape, W.shape)
        print(X.dot(W).shape, b)
        Z = X.dot(W) + b

        return(1 / (1 + np.exp(-Z)))
        # if self.activation == "sigmoid" :
        #     return(1 / (1 + np.exp(-Z)))
        # elif self.activation == "softmax" :
        #     e_z = np.exp(X)
        #     return(e_z/np.sum(e_z))
    
    def __log_loss(self, A, y) :
        epsilon = 1e-15 #Eviter le 0 dans le log
        return((1/len(y)) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)))

    def __gradients(self, A) :
        dW = 1 / len(self.y_train) * np.dot(self.X_train.T, A - self.y_train)
        db = 1 / len(self.y_train) * np.sum(A - self.y_train)
        return (dW, db)
    
    def __forward_prop(self, x) :
        current_a = x
        for layer in range(len(self.layers)) :
            prev_a = current_a
            W = self.layers[layer].W
            b = self.layers[layer].b
            z = prev_a.dot(W) + b
            activation = self.layers[layer].activation(z)
            current_a = activation
            self.layers[layer-1].a = prev_a
            self.layers[layer].Z = z
        return(activation)
    
    def __back_prop(self, activation) :
        m = self.y_train.shape[1]
        y = self.y_train.reshape(activation.shape)
        dA_prev = - (y/activation - (1-y/1-activation))
        for layer in reversed(range(len(self.layers)-1)) :
            dA_curr = dA_prev
            a = self.layers[layer-1].a
            Z = self.layers[layer].Z
            W = self.layers[layer].W
            b = self.layers[layer].b
            m = a.shape[1]
            dz = self.layers[layer].activation_back(dA_curr, Z)
            dW = np.dot(dz.T, a) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            dA_prev = np.dot(W, dz.T)
            self.layers[layer].dW = dW
            self.layers[layer].db = db
    
    def __update(self) :
        for layer in self.layers :
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
    
    def train(self) :
        self.X_train  = (self.X_train  - self.mean) / self.std
        self.X_test  = (self.X_test  - self.mean) / self.std
        for i in range(self.n_iter) :
            a = self.__forward_prop(self.X_train)
            self.__back_prop(a)
            self.__update()
        
    def __update(self, dW, db, W, b) :
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        return(W, b)
    
    def fit(self) :
        W, b = self.__initialize()
        self.X_train  = (self.X_train  - self.mean) / self.std
        self.X_test  = (self.X_test  - self.mean) / self.std
        for epoch in range(self.n_iter) :
            #Train loop
            activation = self.__activation(W, b, self.X_train)
            # loss in train set
            loss = self.__log_loss(activation, self.y_train)
            self.train_loss.append(loss)
            #loss in test set
            activation_test = self.__activation(W, b, self.X_test)
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
        activation = self.__activation(self.W, self.b, X_test)
        return activation >= 0.5