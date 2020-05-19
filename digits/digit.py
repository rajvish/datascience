import matplotlib.pyplot as plt
import numpy as np
from NNLayer import NetworkLayer
from NNLayer import ReLULayer
from NNLayer import sigmoidLayer
from NNLayer import compute_loss_ReLU
from NNLayer import compute_loss_gradients_ReLU
#from NNLayer import compute_loss_gradients_sigmoid

from mninst_data import mninst_data

    
def buildNetwork(X,layers):
    """
    layers is the output length for each layer.
    It contains atleast one element, the number of classes
    For example: a three layer Neural network with a hidden layer of 64 units and 10 output classes 
    will be layers=[64,10]
    """
    assert(len(layers) >=1)
    network=[]
    m=X.shape[1]
    for i in range(len(layers)):
        l = layers[i]
        network.append(NetworkLayer("Wb layer "+str(i),m,l))
        network.append(ReLULayer("ReLU layer"+str(i)))
        #network.append(sigmoidLayer("sigmoid layer"+str(i)))
        m=l
    return network

def forwardPropagation(network,X):
    activations=[]
    input=X
    for n in network:
        a=n.forwardPropagation(input)
        activations.append(a)
        input=a
    return activations
        
def backwardPropagation(network,input,iGradients):
    oGradients=[]
    grad=iGradients
    for n in network:
        o=n.backwardPropagation(input,grad)
        oGradients.append(o)
        grad = o
    return  oGradients
def trainNetwork(network,X,y):

    #Forward prop
    activation= forwardPropagation(network,X)
    network_inputs = [X]+activation  #Input for each layer
    yhat=activation[-1]

    loss = compute_loss_ReLU(yhat,y)
    loss_gradients = compute_loss_gradients_ReLU(yhat,y)

    #backward prop
    for l in range(len(network))[::-1]:
        layer=network[l]
        #print("Starting",layer.layerName,network_inputs[l].shape,loss_gradients.shape)
        loss_gradients=layer.backwardPropagation(network_inputs[l],loss_gradients)
    loss = np.mean(loss)
    #print(loss)
    return loss
def predict(network,X):
    activation=forwardPropagation(network,X)[-1]
    digits=activation.argmax(axis=-1)
    return digits
def main():
    np.random.seed(0)
    iFile = "train-images-idx3-ubyte"
    lFile = "train-labels-idx1-ubyte"
    data=mninst_data(iFile,lFile)
    images,labels = data.getData()
    X= images/255.0
    y=labels

    #print(type(X))
    X_train=X[:50000]
    y_train=y[:50000]
    X_val=X[50000:]
    y_val=y[50000:]
    print(X_train.shape)
    layers=[10]
    network=buildNetwork(X_train,layers)
    
    loss=[]
    for i in range(100):
        print(i)
        loss.append(trainNetwork(network,X_train,y_train))
        #print(np.mean(predictNetwork(network,X_train) ==y_train))
    plt.plot(loss)
    plt.show()
    yhat=predict(network,X_train)
    print(np.mean(yhat == y_train))

    yhat=predict(network,X_val)
    print(np.mean(yhat == y_val))

if __name__ == "__main__":
    main()
