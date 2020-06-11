import  matplotlib.pyplot as plt
from Network import NeuronLayer
from Network import *
import numpy as np

import keras
def plotActivation(a):
    fig, (ax1, ax2) = plt.subplots(2)
    X,Y = a.plotFunction()
    ax1.plot(X,Y)
    X,Y = a.plotDerivative()
    ax2.plot(X,Y)
    ax1.set_title("Function")
    ax2.set_title("Derivative")
    #plt.show()
def plotPredictions(loss,predictions):
    fig, (ax1, ax2) = plt.subplots(2)
    ax2.set_ylim([0.0,1.0])
    ax1.plot(loss)
    ax2.plot(predictions)
    #plt.show()
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
    nclasses=10
    nhidden1=128
    nhidden2=64
    (X,y),(X_test,y_test)=getData()
    X_train=X
    y_train = encode1shot(y,nclasses)
    
    n= X.shape[1]

    np.random.seed(0)
    network = Network()
    network.add(NeuronLayer("Neuron Layer",n,nclasses))
    lrelu=leakyReLULayer("leakyReLU Layer",0.01)
    print("Using Activation",lrelu.layerName)
    #asinh=arcSinHLayer("arcSinH Layer")
    #print("Using Activation",asinh.layerName)
    network.add(lrelu)

    loss=network.trainNetwork(X_train,y_train,100)
    yhat = network.predict(X_test)
    pSuccess = np.mean(yhat == y_test)
    print("Prediction Success",pSuccess)
    plotActivation(lrelu)
    plotPredictions(network.getLoss(),network.getPredictions())
    plt.show()

    #Single Layer Neural Network 
    #network.add(tanhLayer("tanh Layer"))
    #network.add(arcTanLayer("arcTan Layer"))
    #network.add(arcSinHLayer("arcSInH Layer"))
    #network.add(softSignLayer("softSign Layer"))
    #network.add(leakyReLULayer("leakyReLU Layer",0.01))
    #network.add(sinusoidLayer("sunusoid Layer"))
    #network.add(gaussianLayer("gaussian Layer"))
    

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
    
    
if __name__ == "__main__":
    test()
