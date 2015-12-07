"""
Class for a classification algorithm.
"""

import numpy as np
from collections import Counter

class Classifier:

    def __init__(self, classifier_type, **kwargs):
        """
        Initializer. Classifier_type should be a string which refers
        to the specific algorithm the current classifier is using.
        Use keyword arguments to store parameters
        specific to the algorithm being used. E.g. if you were
        making a neural net with 30 input nodes, hidden layer with
        10 units, and 3 output nodes your initalization might look
        something like this:

        neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

        Here I have the weight matrices being stored in a list called weights (initially empty).
        """
        self.classifier_type = classifier_type
        self.params = kwargs
        """
        The kwargs you inputted just becomes a dictionary, so we can save
        that dictionary to be used in other methods.
        """
        self.clf = None

    def train(self, training_data):
        """
        Data should be nx(m+1) numpy matrix where n is the
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type. E.g. if you had a module called
        neural_nets:

        if self.classifier_type == 'neural_net':
            import neural_nets
            neural_nets.train_neural_net(self.params, training_data)

        Note that your training algorithms should be modifying the parameters
        so make sure that your methods are actually modifying self.params

        You should print the accuracy, precision, and recall on the training data.
        """
        from NeuralNetwork import NeuralNetwork
        # find the numbers for feature and label
        featureNum = training_data.shape[1] - 1
        # labelNum = len(np.unique(training_data[:, :1]))
        labelNum = 10

        # get the number of nodes for each layer
        if "hidden_layer" in self.params and self.params["hidden_layer"] is not None:
            nodeNum = [featureNum] + self.params["hidden_layer"] + [labelNum]
        else:
            nodeNum = [featureNum, featureNum * 2, labelNum]

        # get the mode for initializing the weight
        if "weightInitMode" in self.params and self.params["weightInitMode"] is not None:
            weightInitMode = self.params["weightInitMode"]
        else:
            weightInitMode = None

        # get the momentum factor
        if "momentumFactor" in self.params:
            momentumFactor = self.params["momentumFactor"]
        else:
            momentumFactor = 0.0

        self.clf = NeuralNetwork(training_data, nodeNum, weightInitMode, momentumFactor)
        self.clf.train()
        self.test(training_data, "training")


    def predict(self, data):
        """
        Predict class of a single data vector
        Data should be 1x(m+1) numpy matrix where m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type.

        This method should return the predicted label.
        """
        return self.clf.predict(data)

    def test(self, test_data, mode):
        """
        Data should be nx(m+1) numpy matrix where n is the
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        You should print the accuracy, precision, and recall on the test data.
        """
        correct = 0
        countPrediction = {}
        countCorrect = {}
        countTotal = Counter(list(test_data[:, 0]))
        allPrediction = {}

        labels = np.unique(test_data[:, 0])
        for label in labels:
            countCorrect[label] = 0
            countPrediction[label] = 0
            allPrediction[label] = 0

        for e in test_data:
            label = e[0]
            pred_label = self.predict(e)
            if label == pred_label:
                correct += 1
                if e[0] in countCorrect:
                    countCorrect[e[0]] += 1
                else:
                    countCorrect[e[0]] = 1
            if pred_label in allPrediction:
                allPrediction[pred_label] += 1
            else:
                allPrediction[pred_label] = 1

            if pred_label in countPrediction:
                countPrediction[pred_label] += 1
            else:
                countPrediction[pred_label] = 1

        print "count correct", countCorrect
        print "all predictions", allPrediction
        accuracy = float(correct) / len(test_data)
        print "The accuracy for", mode, "is", accuracy



        # for key in countPrediction.keys():
        #     if countPrediction[key] == 0:
        #         print "precision for key:", key, " = ", 0.0
        #     else:
        #         print "precision for key:", key, " = ", countCorrect[key] / float(countPrediction[key])
        #     print "recall for key:", key, "    = ", countCorrect[key] / float(countTotal[key])
        # print

    def getAttrValue(self, ex):
        """
        Find the attribute values for each attribute.
        Args:
            ex: given examples
        Returns: a dictionary where the keys are the attribute indices and the values are the attribute values.
        """
        attrValue = {}
        for i in range(len(ex[0])):
            attrValue[i] = list(set([v for v in ex[:, i]]))
        return attrValue