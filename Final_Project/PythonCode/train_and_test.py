import numpy as np
from classifier import Classifier
import load_data
import time


def runNeuralNetwork(train, test, hLayer=None, mode=None, momentumFactor=0.0):
    print ""
    print "Neural Network =============================="
    print " - number of hidden layer nodes:",
    if hLayer is not None:
        print hLayer
    else:
        print " default (one hidden layer with node number = 2 * feature number)"

    print " - weight initialization mode:",
    if mode is not None:
        print mode
    else:
        print "default"

    print " - momentum factor", momentumFactor

    nn = Classifier("neural_network", hidden_layer=hLayer, weightInitMode=mode, momentumFactor=momentumFactor)
    nn.train(train.copy(), test.copy())
    nn.test(test.copy(), "test")


if __name__ == "__main__":

    trainNum = 30000
    testNum = 10000
    trainingData = load_data.load_digit2("data/train.csv", 1, trainNum + 1)
    testData = load_data.load_digit2("data/train.csv", trainNum + 1, trainNum + testNum + 1)

    momentumFactor = 0.8

    # ======================
    weightInitMode = "shallow"
    hidden_layer   = [1000]
    runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # # ======================
    # weightInitMode = "shallow"
    # hidden_layer   = [16]
    # runNeuralNetwork(trainingData, testData, hidden_layer, weightInitMode, momentumFactor)
    # # ======================
    # weightInitMode = "deep"
    # hidden_layer   = [1000, 1000]
    # runNeuralNetwork(traininData, testData, hidden_layer, weightInitMode, momentumFactor)