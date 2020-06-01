import numpy as np
class Network:
    def __init__(self):
        self.network=[]
        self.activations=[]
        self.loss=[]
        self.predictions=[]
    def add(self,layer):
        self.network.append(layer)
        return
    def remove(self,layer):
        self.network.remove(layer)
        return
    def printNetwork(self):
        if len(network) ==0:
            print("Empty Network")
            return
        for n in self.network:
            print(n.layerName)
            return
    def _forwardPropagation(self,X):
        activations=[X]
        input = X
        for n in self.network:
            #print("Forward :",n.layerName)
            a=n.forwardPropagation(input)
            activations.append(a)
            input=a
        #print(len(activations))
        #print((activations))
        return activations
    def _backwardPropagation(self,activations,iGradientLoss):
        gloss=iGradientLoss
        for l in range(len(self.network))[::-1]:
            layer=self.network[l]
            #print("Backward :",layer.layerName)
            gloss=layer.backwardPropagation(activations[l],gloss)
        return 
    def predict(self,X):
        activation=self._forwardPropagation(X)[-1]
        prediction=activation.argmax(axis=-1)
        return prediction
    def computeLoss(self,yhat,y):
        m =yhat.shape[0]
        #y is onehot encoded of the label value 'l'.
        # pick up the lth value from yhat
        predictionProbability=np.max(yhat*y,axis=1) 
        loss=(1./m)*(-predictionProbability+np.log(np.sum(np.exp(yhat),axis=-1)))
        return loss
    def computeGradientloss(self,yhat,y):
        m=yhat.shape[0]
        softmax=np.exp(yhat)/np.exp(yhat).sum(axis=1,keepdims=True)
        gloss=(1./m)*(-y+softmax)
        return gloss
    def _singleStep(self,X,y):
        activations=self._forwardPropagation(X)
        yhat=activations[-1]
        y1=np.argmax(y*yhat,axis=1)
        p=np.mean(y1 == np.argmax(y))
        self.predictions.append(p)
        loss=self.computeLoss(yhat,y)
        gloss=self.computeGradientloss(yhat,y)
        self._backwardPropagation(activations,gloss)
        return np.mean(loss)
        
    def trainNetwork(self,X,y,steps):
        for i in range(steps):
            self.loss.append(self._singleStep(X,y))
        return self.loss
    def getPredictions(self):
        return self.predictions
    
class NeuronLayer:
    def __init__(self,layerName,nInput,nOutput,lrate=0.5,factor=0.01):
        self.layerName = layerName
        self.nInput= nInput
        self.Output=nOutput
        self.lrate=lrate
        self.factor = factor
        self.W=np.random.randn(nInput,nOutput)*self.factor
        print("W Shape",self.W.shape)
        self.b=np.zeros(nOutput)
        print("b Shape",self.b.shape)
    def forwardPropagation(self,input):
        retval= np.matmul(input,self.W)+self.b
        return retval
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss = np.matmul(iGradientLoss,self.W.T)
        dW= np.matmul(input.T,iGradientLoss)
        db=np.sum(iGradientLoss,axis=0)
        self.W = self.W - self.lrate*dW
        self.b = self.b - self.lrate*db
        return oGradientLoss
class ReLULayer:
    def __init__(self,layerName):
        print("Initing Layer",layerName)
        self.layerName = layerName
    def dZ(self,input):
        return input>0
    def forwardPropagation(self,input):
        return np.maximum(0,input)
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss = iGradientLoss*self.dZ(input)
        return oGradientLoss

class sigmoidLayer:
    def __init__(self,layerName):
        print("Initing Layer",layerName)
        self.layerName = layerName
    def dZ(self,input):
        return self.sigmoid(input)*(1-self.sigmoid(input))
    def sigmoid(self,z):
        return 1./(1.+np.exp(-z))
    def forwardPropagation(self,input):
        sigmoidInput = self.sigmoid(input)
        return sigmoidInput
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss = iGradientLoss*self.dZ(input)
        return  oGradientLoss
class tanhLayer:
    def __init__(self,layerName):
        print("Initing Layer",layerName)
        self.layerName = layerName
    def tanh(self,z):
        #t=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        t=np.tanh(z)
        return t
    def dZ(self,z):
        t=self.tanh(z)
        return 1-t*t
    def forwardPropagation(self,input):
        return self.tanh(input)
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss= iGradientLoss*self.dZ(input)
        return  oGradientLoss


