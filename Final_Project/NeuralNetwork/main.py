from classifier import Classifier
import data_loader


def runNeuralNetwork(train, test, batchSize, classNum, hLayer=None, mode=None, momentumFactor=0.0):
    """
    A function that call the the classifier to train a learning model.
    Args:
        train: training examples (numpy)
        test: testing examples (numpy)
        batchSize: the number of training example for each iteration
        classNum: the number of classes
        hLayer: number of the hidden layer nodes (list)
        mode: weight initializing mode
        momentumFactor: momentum factor
    """
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
    nn.train(train, test, classNum, batchSize)
    nn.test(test, "test")


if __name__ == "__main__":

    trainNum = 300
    testNum = 100
    trainingData = data_loader.load_csv("data/train.csv", 1, trainNum + 1) # skip the first row, which is a title
    testData = data_loader.load_csv("data/train.csv", trainNum + 1, trainNum + testNum + 1)

    print ""
    print "training data", trainingData.shape
    print "testing data", testData.shape

    momentumFactor = 0.8
    batchSize = 1
    classNum = 10

    # ======================
    # weight initializing mode support default, "shallow", and "deep"
    weightInitMode = "shallow"
    hidden_layer   = [1000]
    runNeuralNetwork(trainingData, testData, batchSize, classNum, hidden_layer, weightInitMode, momentumFactor)
