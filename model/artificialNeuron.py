import numpy as np
import matplotlib.pyplot as plt

class Layer :
    def __init__(self, input_size, output_size, activation="sigmoid") :
        self.x = x
        self.y = y
        self.activation = activation
        self.layer = self.init_network()
        self.W = np.random.randn(self.input_size.shape[1], 1)
        self.b = np.random.randn(self.output_size.shape[1], 1)
        self.a = 1
        self.Z = 1
        self.dW = 1
        self.db = 1
    
    def init_network(self) :
        l =[]
        for i in range(self.x * self.y):
            l.append(np.random.randn())
        return(np.array(l).reshape(self.x, self.y))

    def activation(self, Z, b) :
        if self.activation == "sigmoid" :
            return(1 / (1 + np.exp(-Z)))
        
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
        for layer in range(len(self.layers)-1) :
            prev_a = current_a
            W = self.layers[layer+1].W
            b = self.layers[layer+1].b
            z = prev.dot(W) + b
            activation = self.layers[layer].activation(z)
            activation = current_a
            self.layers[layer].a = prev_a
            self.layers[layer+1].Z = z
        return(a)
    
    def __back_prop(self, activation) :
        m = self.y_train.shape[1]
        y = self.y_train.reshape(activation.shape)
        dA_prev = - (y/activation - (1-y/1-activation))
        for layer in reversed(range(len(self.layers)-1)) :
            dA_curr = dA_prev
            a = layers[layer].a
            Z = layers[layer+1].Z
            W = layers[layer+1].W
            b = layers[layer+1].b
            m = a.shape[1]
            dz = layers[layer].activation(Z)
            dW = np.dot(dz, a.T) / m
            db = np.sum(dz, axis+1, keepdims=True) / m
            da = np.dot(W, dz)
            layers[layer+1].dW = dW
            layers[layer+1].db = db
    
    def __update(self) :
        for layer in self.layers :
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
    
    # def __update(self, dW, db, W, b) :
    #     W = W - self.learning_rate * dW
    #     b = b - self.learning_rate * db
    #     return(W, b)
    
    # def fit(self) :
    #     W, b = self.__initialize()
    #     self.X_train  = (self.X_train  - self.mean) / self.std
    #     self.X_test  = (self.X_test  - self.mean) / self.std
    #     for epoch in range(self.n_iter) :
    #         #Train loop
    #         activation = self.__activation(W, b, self.X_train)
    #         # loss in train set
    #         loss = self.__log_loss(activation, self.y_train)
    #         self.train_loss.append(loss)
    #         #loss in test set
    #         activation_test = self.__activation(W, b, self.X_test)
    #         loss_test = self.__log_loss(activation_test, self.y_test)
    #         self.test_loss.append(loss_test)
                
    #         dW, db = self.__gradients(activation)
    #         W, b = self.__update(dW, db, W,b)
    #     self.W, self.b = W, b
            
    # def plot_loss(self) :
    #     plt.plot(self.train_loss, label="train loss")
    #     plt.plot(self.test_loss, label="test loss")
    #     plt.legend()
    #     plt.show()
        
    def predict(self, X_test) :
        X_test  = (X_test  - self.mean) / self.std
        activation = self.__activation(self.W, self.b, X_test)
        return activation >= 0.5