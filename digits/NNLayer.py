import numpy as np
class NetworkLayer:
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
        #print("Forward Propagation",self.layerName)
        retval= np.matmul(input,self.W)+self.b
        #print("Returning",retval.shape)
        return retval
    def backwardPropagation(self,input,iGradientLoss):
        #print("Back Propagation",self.layerName)
        oGradientLoss = np.matmul(iGradientLoss,self.W.T)
        dW= np.matmul(input.T,iGradientLoss)
        db=np.sum(iGradientLoss,axis=0)
        self.W = self.W - self.lrate*dW
        self.b = self.b - self.lrate*db
        return oGradientLoss
class ReLULayer:
    def __init__(self,layerName):
        self.layerName = layerName
    def dZ(self,input):
        return input>0
    def forwardPropagation(self,input):
        return np.maximum(0,input)
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss = iGradientLoss*self.dZ(input)
        return oGradientLoss

class sigmoidLayer:
    def sigmoid(self,z):                                                                           
        return 1./(1.+np.exp(-z))
    def __init__(self,layerName):
        self.layerName = layerName
    def dZ(self,input):
        return self.sigmoid(input)*(1-self.sigmoid(input))
    def forwardPropagation(self,input):
        sigmoidInput = self.sigmoid(input)
        return sigmoidInput
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss = iGradientLoss*self.dZ(input)
        return  oGradientLoss
class tanhLayer:
    def __init__(self,layerName):
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


#Compute the Loss. yhat is the output of some activation layer. Could be Sigmoid or ReLU or TanH
# y is the one hot encoding of the putput
def compute_loss(yhat,y):
    m = yhat.shape[0]
    lfora = np.max(yhat*y,axis=1)  #Pick the estimated activation for each element
    loss=(1./m)* (-lfora+np.log(np.sum(np.exp(yhat),axis=-1)))
    return loss
def compute_loss_gradients(yhat,y):
    softmax = np.exp(yhat) / np.exp(yhat).sum(axis=-1,keepdims=True)
    loss_gradients = (-y+softmax)/yhat.shape[0]
    return loss_gradients
