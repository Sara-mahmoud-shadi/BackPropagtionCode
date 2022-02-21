from Activations import *
from Constants import *
import numpy as np

def getSignalValues(layerNum, neural_network):
    layer = neural_network.Layers[layerNum]
    outputVector = []
    for neuron in layer.neurons:
        outputVector.append(neuron.signal_error)
    outputVector = np.array(outputVector)
    outputVector = outputVector.reshape([layer.number_of_neurons, 1])
    return outputVector


def getOtherTerm(layerNum, neural_network):
    signalVector = getSignalValues(layerNum+1, neural_network)
    weightVector = neural_network.weights[layerNum+1]
    result = signalVector * weightVector
    result = result.sum(axis =0)
    return result


def getChangeInWeight(i, neural_network, eta , dw_cols , dw_rows):
    Dw = np.zeros([dw_rows , dw_cols])
    for x in range(dw_cols) :
        if i == 0:
           input = (neural_network.inputLayer.Inputs)[x]
        else:
           input = getInputList(neural_network.Layers[i - 1], neural_network.useBias)[x]
        signal_Vector = getSignalValues(i, neural_network)
        dw = signal_Vector * input * eta
        dw = dw.reshape(dw_rows , 1)
        Dw[:,x] = dw[:,0]
    return Dw


def getInputList(layer, useBias):
    inputs = []
    if useBias == 1:
        inputs.append(1)
    for i in range(layer.number_of_neurons):
        inputs.append(layer.neurons[i].output)
    inputs = np.array(inputs)
    inputs = inputs.reshape([layer.number_of_neurons+useBias, 1])
    return inputs

#----------------------------------------------#

def forward(neural_network):
    for i in range(neural_network.number_of_hidden_Layers+1):
        weightVector = neural_network.weights[i]
        if i == 0:
            inputVector = neural_network.inputLayer.Inputs
        else:
            inputVector = getInputList(neural_network.Layers[i-1], neural_network.useBias)
        result = np.dot(weightVector, inputVector)
        num_rows, num_cols = result.shape
        for x in range(neural_network.Layers[i].number_of_neurons):
            neural_network.Layers[i].neurons[x].netInput = result[x][0]
        if neural_network.Activation == SIGMOID_FUN:
            for x in range(num_rows):
                result[x] = sigmoid(result[x])
        else:
            for x in range(result.num_rows):
                result[x] = HYPERBOLIC_TANGET_FUN(result[x])
        for x in range(neural_network.Layers[i].number_of_neurons):
            neural_network.Layers[i].neurons[x].output = result[x][0]


def backward(neural_network):
    for i in range(neural_network.number_of_hidden_Layers, -1, -1):
        layer = neural_network.Layers[i]
        if i == neural_network.number_of_hidden_Layers:
            # output layer
            for neuron in layer.neurons:
                if neural_network.Activation == SIGMOID_FUN:
                    neuron.signal_error = (neuron.target - neuron.output) * gradient_sigmoid(neuron.netInput)
                else:
                    neuron.signal_error = (neuron.target - neuron.output) * gradient_Hyperbolic_Tangent(neuron.netInput)
        else:
            # hidden Layer
            for index, neuron in enumerate(layer.neurons):
                if neural_network.Activation == SIGMOID_FUN:
                    neuron.signal_error = gradient_sigmoid(neuron.netInput) * getOtherTerm(i, neural_network)[index]
                else:
                    neuron.signal_error = gradient_Hyperbolic_Tangent(neuron.netInput) * getOtherTerm(i, neural_network)[index]


def updateWeights(neural_network, eta):
    for i in range(neural_network.number_of_hidden_Layers+1):
        oldWeight = neural_network.weights[i]
        row , col = oldWeight.shape
        dw = getChangeInWeight(i, neural_network, eta ,col ,row)
        newWeight = oldWeight + dw
        neural_network.weights[i] = newWeight

#----------------------------------------------#


