import numpy as np
class NetworkLayer:
    def __init__(self,layerName,nInput,nOutput,lrate=0.5):
        self.layerName = layerName
        self.nInput= nInput
        self.Output=nOutput
        self.lrate=lrate
        self.W=np.random.randn(nInput,nOutput)*0.05
        print("W Shape",self.W.shape)
        self.b=np.zeros(nOutput)
        print("b Shape",self.b.shape)
    def forwardPropagation(self,input):
        #print("Forward Propagation",self.layerName)
        retval= np.matmul(input,self.W)+self.b
        #print("Returning",retval.shape)
        return retval
    def backwardPropagation(self,input,iGradients):
        #print("Back Propagation",self.layerName)
        oGgradients = np.matmul(iGradients,self.W.T)
        dW= np.matmul(input.T,iGradients)
        db=np.mean(iGradients,axis=0)*input.shape[0]
        self.W = self.W - self.lrate*dW
        self.b = self.b - self.lrate*db
        return oGgradients 
    def getWb(self):
        return W,b
        
class ReLULayer:
    def __init__(self,layerName):
        self.layerName = layerName
    def dZ(self,input):
        return input>0
    def forwardPropagation(self,input):
        #print("Forward Propagation",self.layerName)
        #print("Input Shape",input.shape)
        return np.maximum(0,input)
    def backwardPropagation(self,input,prevgradients):
        #print("Back Propagation",self.layerName)
        #print("Input shape",input.shape,"gradient Output.shape",gradientOutput.shape)
        #print("relugradient",reluGradient.shape)
        return prevgradients*(input > 0)

class sigmoidLayer:
    def __init__(self,layerName):
        self.layerName = layerName
    def dZ(self,input):
        return input*(1-input)
    def forwardPropagation(self,input):
        eZ=np.exp(input)
        sum=np.sum(eZ,axis=1)
        s1=(eZ.T/sum).T
        print("Forward Propgation",s1.shape)
        return s1
    def backwardPropagation(self,input,prevgradients):
        return  prevgradients*self.dZ(input)


def compute_loss_ReLU(yhat,y):
    lfora=yhat[np.arange(len(yhat)),y] #dimension  = len(y)
    loss= -lfora+np.log(np.sum(np.exp(yhat),axis=-1))
    print(loss.shape)
    return loss

def compute_loss_gradients_ReLU(yhat,y):
    onesfora=np.zeros_like(yhat)
    onesfora[np.arange(len(yhat)),y]=1
    #softmax=np.exp(yhat)/np.exp(yhat).sum(axis=-1,keepdims=True)
    softmax = np.exp(yhat) / np.exp(yhat).sum(axis=-1,keepdims=True)
    #print("Softmamx shape",softmax.shape)
    #print("************")
    loss_gradients = (-onesfora+softmax)/yhat.shape[0]
    print("loss gradients =",loss_gradients.shape)
    return loss_gradients
