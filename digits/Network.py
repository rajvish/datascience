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
        y1=yhat.argmax(axis=-1)
        
        p=np.mean(y1 == np.argmax(y,axis=1))
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
    def getLoss(self):
        return self.loss
    
class NeuronLayer:
    def __init__(self,layerName,nInput,nOutput,lrate=0.5,factor=0.01):
        self.layerName = layerName
        self.nInput= nInput
        self.Output=nOutput
        self.lrate=lrate
        self.factor = factor
        self.W=np.random.randn(nInput,nOutput)*self.factor
        #print("W Shape",self.W.shape)
        self.b=np.zeros(nOutput)
        #print("b Shape",self.b.shape)
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
class ActivationLayer:
    def __init__(self,layerName):
        self.layerName = __name__+layerName
    def function(self,input):
        return self.layerName+"Function Not Implemented"
    def plotFunction(self,min=-10.,max=10.):
        X=np.linspace(min,max,num=500)
        Y=self.function(X)
        return X,Y
    def plotDerivative(self,min=-10.,max=10.):
        X=np.linspace(min,max,num=500)
        Y=self.derivative(X)
        return X,Y
    def derivative(self, input):
        return self.layerName+"Function Not Implemented"
    def forwardPropagation(self,input):
      return self.function(input)
    def backwardPropagation(self,input,iGradientLoss):
        oGradientLoss= iGradientLoss*self.derivative(input)
        return  oGradientLoss
class ReLULayer(ActivationLayer):
    def derivative(self,input):
        return input>0
    def function(self,input):
        return np.maximum(0,input)
class sigmoidLayer(ActivationLayer):
    def funcction(self,z):
        return 1./(1.+np.exp(-z))
    def derivative(self,input):
        t= self.function(input)
        return t*(1-t)
class tanhLayer(ActivationLayer):
    def function(self,z):
        return np.tanh(z)
    def derivative(self,z):
        t=self.function(z)
        return 1-t*t


class identityLayer(ActivationLayer):
    def function(self,input):
      return input
    def derivative(self,input):
      return 1

class arcTanLayer(ActivationLayer):
    def function(self,input):
      return np.arctan(input)
    def derivative(self,input):
      return 1./((input*input)+1)
class arcSinHLayer(ActivationLayer):
    def function(self,input):
      return np.sinh(input)
    def derivative(self,input):
      return 1./np.sqrt(((input*input)+1))
      
class softSignLayer(ActivationLayer):
    def function(self,input):
        return input/(1+np.abs(input))
    def derivative(self,input):
        t=self.function(input)
        return 1./t*t
class leakyReLULayer(ActivationLayer):
    def __init__(self,layerName,alpha):
        super().__init__(layerName)
        #self.layerName = layerName
        self.alpha =alpha
    def function(self,input):
        return np.where(input > 0,input, input*self.alpha)
    def derivative(self,input):
        return np.where(input > 0,1.0, self.alpha)
        return self.alpha if input < 0 else 1.0
class sinusoidLayer(ActivationLayer):
    def function(self,input):
        return np.sin(input)
    def derivative(self,input):
        return np.cos(input)
class gaussianLayer(ActivationLayer):
    def function(self,input):
        return np.exp(-input*input)
    def derivative(self,input):
        return -2.*input*self.function(input)
