import  matplotlib.pyplot as plt
from Network import NeuronLayer
from Network import *
import numpy as np

import keras
def encode1shot(labels,nclasses):
    m =len(labels)
    y=np.zeros((m,nclasses))
    y[np.arange(m),labels]=1
    return y
    
def getData():
    (X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype(float)/255
    X_test = X_test.astype(float)/255
    X_train  = X_train.reshape(X_train.shape[0],-1)
    X_test  = X_test.reshape(X_test.shape[0],-1)
    return(X_train,y_train),(X_test, y_test)

def test():
    loss=[]
    np.random.seed(0)
    nclasses=10
    nhidden1=128
    nhidden2=64
    (X,y),(X_test,y_test)=getData()
    #print(y)
    X_train=X
    y_train = encode1shot(y,nclasses)
    #y_test = encode1shot(y_test,nclasses)
    
    n= X.shape[1]

    network = Network()

    #Single Layer Neural Network 
    network.add(NeuronLayer("Neuron Layer",n,nclasses))
    network.add(tanhLayer("tanh Layer"))

    """
    #Two Layer Neural Network with sigmoid and tanh
    network.add(NeuronLayer("First Neuron Layer",n,nhidden1))
    network.add(ReLULayer("ReLU Layer"))
    network.add(NeuronLayer("Second Neuron Layer",nhidden1,nclasses))
    network.add(tanhLayer("tanh Layer"))
    #Three Layer Neural Network with sigmoid and tanh
    network.add(NeuronLayer("First Neuron Layer",n,nhidden1))
    network.add(ReLULayer("sigmoid Layer"))
    network.add(NeuronLayer("Second Neuron Layer",nhidden1,nhidden2))
    network.add(tanhLayer("tanh Layer"))
    network.add(NeuronLayer("Third Neuron Layer",nhidden2,nclasses))
    network.add(ReLULayer("ReLU Layer"))
    """
    
    
    loss=network.trainNetwork(X_train,y_train,100)
    yhat = network.predict(X_test)
    print("Prediction Success", np.mean(yhat == y_test))

    predictions=network.getPredictions()
    print(predictions)
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax2.set_ylim([0.0,1.0])
    ax1.plot(loss)
    ax2.plot(predictions)
    plt.show()
if __name__ == "__main__":
    test()
