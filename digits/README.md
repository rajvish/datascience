Numpy and Python based Digit recognizer

1. Python and Numpy based
2. Data is downloaded from the web
3. Can be used to create multiple layers
4. Different activation functions

Create a Network:
    - Create a NeuronLayer
    - Create an Activation Layer
    For example, to create a two layer network with leaky ReLU activation
        network = Network()
        network.add(NeuronLayer("Neuron Layer",n,nclasses))
        lrelu=leakyReLULayer("leakyReLU Layer",0.01)
        network.add(lrelu)

Train:
    This will train the network for 100 steps
        network.trainNetwork(X_train,y_train,100)
Predict from the trained Network:
        yhat = network.predict(X_test)
Loss:
    network.getLoss() 
    will get the loss at each step
Predictions:
    network.getPredictions()
    will get the Predictions at each step
Activations Graphs:
    lrelu.plotFunction() - Will plot the Activation function 
    lrelu.plotDerivative() - Will plot the derivative for the Activation Function
Making sure this goes
