from classifier import Classifier
import numpy as np
import math
import time

class NeuralNetwork(Classifier):
    """
    A neural network classifier with feed forward and back-propagation.
    """

    def __init__(self, training_data, nodeNum, weightInitMode=None, momentumFactor=0.0):
        # number of total training iteration
        self.ITERATION = 1000

        # for randomly initializing theta
        self.EPISLON = 0.1

        # learning rate
        self.ALPHA = 0.00001

        # decay rate for momentum
        self.momentumFactor = momentumFactor

        # dictionary for momentum
        self.momentum = {}

        # number of classes for the classification
        self.classNum = nodeNum[-1]

        # number of nodes in each layers
        self.nodeNum = {}
        i = 1
        for nn in nodeNum:
            self.nodeNum[i] = nn
            i += 1

        # number of total layers including the input and output layers
        self.layerNum = len(nodeNum)

        # the weight (theta) initialization mode
        self.weightInitMode = weightInitMode

        self.theta = {}
        self.delta = {}
        self.z = {}
        self.a = {}
        self.djdw = {}

        # get labels and nodes for the first layer (input layer)
        self.labels, self.x = self.splitData(training_data)

    def splitData(self, training_data):
        """
        Split the labels and features from the input training_data.
        Args:
            training_data:
        Returns:
            labels: a (exampleNum x classNum) matrix that contains label for each training example
            examples: a (exampleNum x featureNum ) matrix for the input training example
        """
        if self.classNum > 1:
            labels = np.zeros((len(training_data), self.classNum))
            for i, label in enumerate(training_data[:, 0].reshape(len(training_data), 1)):
                labels[i][int(label)] = 1.0
            labels = labels.T
        else:
            labels = training_data.copy()[:, 0]              # get the first column (label) from the matrix
            labels = labels.reshape(labels.shape[0], 1).T

        examples = training_data.copy()[:, 1:]               # get the rest columns (features of examples) from the matrix
        onesCol = np.ones((examples.shape[0], 1))
        examples = np.hstack((examples, onesCol))            # add one column (all ones) for bias
        return labels, examples

    def train(self):
        """
        Train this neural network with feed-forward and back-propagation.
        It will first initialize the weight according to different initialization
        methods (default, shallow, deep). The training process will be run

        When training is finished, the theta can be used for prediction.
        """
        startTime = time.time()
        self.initTheta()

        iter = 0
        cost = 1
        while cost > 0.01 and iter <= self.ITERATION:
            iter += 1
            yHat = self.feedForward(self.x)
            cost = self.cost(self.labels, yHat)
            if iter < 10:
                print cost
            self.backpropagation()
            self.updateTheta()

            if iter % 10 == 0:
                print "cost = ", cost, " (", iter, " iteration)"

        # print "cost = ", cost
        print "Training time", time.time() - startTime

    def initTheta(self):
        """
        initialize each theta by
        theta = random matrix * 2 epislon - epislon
        so each element will be in the range [-epislon, epislon]
        Returns:

        """
        # print "Initialize theta"
        initW = {}
        for i in range(1, self.layerNum):
            if self.weightInitMode is None:
                # Use the default value EPISLON
                # the initial value of theta will be within [-EPISLON, EPISLON]
                initW[i] = self.EPISLON
            elif self.weightInitMode == "shallow":
                # Use number of node in previous layer (n)
                # w = 1 / math.sqrt(n)
                # the initial value of theta will be within [-w, w]
                initW[i] = 1 / float(math.sqrt(self.nodeNum[i]))
            elif self.weightInitMode == "deep":
                # Use the number of nodes in previous and next layers
                # w = math.sqrt(6) / (math.sqrt(preNode + nextNode))
                # the initial value of theta will be within [-w, w]
                initW[i] = math.sqrt(6) / math.sqrt(self.nodeNum[i] + self.nodeNum[i+1])

        for i in range(1, self.layerNum):  # layer number starts from 1,  add one bias
            # theta[previous layer, next layer]
            th = np.random.rand(self.nodeNum[i] + 1, self.nodeNum[i+1]) * 2 * initW[i] - initW[i]
            self.theta[i] = th

    def feedForward(self, x):
        """
        Compute all the x in each layer except the first input layer.
        """
        self.a[1] = x.T  # transform the example to a column-based matrix (just for convention)

        for i in range(1, self.layerNum):
            self.z[i+1] = np.dot(self.theta[i].T, self.a[i])
            self.a[i+1] = sigmoid(self.z[i+1])

            # add bias column to each layer except the output layer
            if i + 1 < self.layerNum:
                conesRow = np.ones((1, self.a[i+1].shape[1]))
                self.a[i+1] = np.vstack((self.a[i+1], conesRow))
        return self.a[self.layerNum]

    def backpropagation(self):
        """
        Compute all the delta for each layer except the first input layer.
        Compute the derivative for each layer except the last output layer.
        """
        # compute the delta for the output layer delta = (y - a) .* a .* (1 - a)
        self.delta[self.layerNum] = -(self.labels - self.a[self.layerNum]) * sigmoidDe(self.z[self.layerNum])

        # compute the delta for other layers (from layerNum - 1 to 2)
        # compute the derivative (d cost / d weight) for layerNum -1 to 1
        # no delta for input layer
        for l in range(self.layerNum - 1, 1, -1):
            preDelta = self.delta[l+1]
            self.djdw[l] = np.dot(self.a[l], preDelta.T)
            self.delta[l] = np.dot(self.theta[l][:-1, :], preDelta) * sigmoidDe(self.z[l])
        self.djdw[1] = np.dot(self.a[1], self.delta[2].T)

    def updateTheta(self):
        """
        Update every theta in network using deltas. Also implemented momentum to
        speed up the weight update.
        for each layer of theta:
            theta_ij = theta_ij - (self.ALPHA * a_i * delta_j + momentum)
        """
        for l in range(self.layerNum - 1, 0, -1):
            thisMomentum = self.ALPHA * np.dot(self.a[l], self.delta[l+1].T)
            if l in self.momentum:
                lastMomentum = self.momentum[l]
            else:
                lastMomentum = 0
            self.theta[l] -= (thisMomentum + self.momentumFactor * lastMomentum)
            self.momentum[l] = thisMomentum

    def cost(self, labels, yHat):
        """
        Args:
            labels: the true labels of the data
            yHat: the predicted of the data
        Returns:
            cost: the cost of the prediction
            costMat: the
        """
        costMat = labels - yHat
        cost = sum(sum(costMat * costMat)) / 2.0
        return cost

    def predict(self, data):
        data = data.reshape(1, data.shape[0])
        label, feature = self.splitData(data)

        yHat = self.feedForward(feature)

        # if self.classNum > 1:
        maxV = -1
        maxIdx = None
        i = 0
        for y in yHat:
            if y > maxV:
                maxV = y
                maxIdx = i
            i += 1
        return maxIdx


# ===================================================================
# Helper functions
# ===================================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    # return 1 / (1 + expit(-z))

def sigmoidDe(z):
    return sigmoid(z) * (1 - sigmoid(z))