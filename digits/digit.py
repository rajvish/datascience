import matplotlib.pyplot as plt
import numpy as np
from NNLayer import NetworkLayer
from NNLayer import ReLULayer
from NNLayer import sigmoidLayer
from NNLayer import tanhLayer

from NNLayer import compute_loss
from NNLayer import compute_loss_gradients


from mnist_digits import mnist_digits

    
def buildNetwork(X,layers):
    """
    layers is the output length for each layer.
    It contains atleast one element, the number of classes
    For example: a three layer Neural network with a hidden layer of 64 units and 10 output classes 
    will be layers=[64,10]

    Softmax layer is added implicitly at the end
    """
    assert(len(layers) >=1)
    network=[]
    n=X.shape[1]
    # Neuron Layer
    network.append(NetworkLayer("Wb layer "+str(0),n,layers[0]))
    # Activation Layer
    network.append(ReLULayer("ReLU layer"+str(0)))
    #network.append(sigmoidLayer("sigmoid layer"+str(0)))
    #network.append(tanhLayer("tanh layer"+str(0)))

    #More Layers if needed
    #network.append(NetworkLayer("Wb layer "+str(0),layers[0])
    #network.append(NetworkLayer("Wb layer "+str(1),layers[0],layers[1]))
    #network.append(NetworkLayer("Softmax"+str(1))
    return network

def forwardPropagation(network,X):
    activations=[]
    input=X
    for n in network:
        a=n.forwardPropagation(input)
        activations.append(a)
        input=a
    return activations
        
def backwardPropagation(network,input,iGradientLoss):
    oGradientLoss=[]
    grad=iGradientLoss
    for n in network:
        grad=n.backwardPropagation(input,grad)
        oGradientLoss.append(grad)
    return  oGradientLoss
def encode1shot(labels,nclasses):
    m =len(labels)
    y=np.zeros((m,nclasses))
    y[np.arange(m),labels]=1
    return y
    
def trainNetwork(network,X,y):
    #Forward propagation
    activation= forwardPropagation(network,X)
    #print(activation)
    network_inputs = [X]+activation  #Input for each layer
    yhat=activation[-1]

    #print("Activation",yhat)
    #Compute the top layer loss and the gradients
    loss = np.mean(compute_loss(yhat,y))
    loss_gradients = compute_loss_gradients(yhat,y)

    #backward propagation
    for l in range(len(network))[::-1]:
        layer=network[l]
        loss_gradients=layer.backwardPropagation(network_inputs[l],loss_gradients)
    loss = np.mean(loss)
    return loss
def predict(network,X):
    activation=forwardPropagation(network,X)[-1]
    digits=activation.argmax(axis=-1)
    return digits
def main():
    np.random.seed(0)
    iFile = "train-images-idx3-ubyte"
    lFile = "train-labels-idx1-ubyte"
    data=mnist_digits(iFile,lFile)
    images,labels = data.getData()
    nclasses=10
    X= images/255.0
    y=encode1shot(labels,nclasses)

    #print(type(X))
    X_train=X[:50000]
    y_train=y[:50000]
    ylabels=labels[:50000]
    X_val=X[50000:]
    y_val=labels[50000:]
    print(X_train.shape)
    layers=[10]
    predictions=[]
    fig, (ax1, ax2) = plt.subplots(2)
    network=buildNetwork(X_train,layers)
    loss=[]
    for i in range(100):
        o= trainNetwork(network,X_train,y_train)
        loss.append(o)
        p= np.mean(predict(network,X_train) ==ylabels)
        predictions.append(p)
        if i%10 ==0:
            print("Iteration i",i, "Loss =",o, "Predictions= ",p)
    ax1.plot(loss)
    ax2.plot(predictions)
    yhat=predict(network,X_val)
    p=np.mean(yhat == y_val)
    print("Prediction",p)
    plt.show()
if __name__ == "__main__":
    main()
