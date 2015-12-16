import numpy as np
import csv


def load_digit(filename, startIdx, endIdx):
    """
    Load the hand written digit features into a matrix
    Args:
        data_ratio: the ratio of examples that go into the training set,
                    and the rest of the examples will be the validation set.
                    For the testing data set, this ratio should be set to 1.
    Returns:
        a np.array that contains the features of examples
    """
    training = None

    with open(filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        for row in csvreader:
            i += 1
            if i <= startIdx:
                continue
            rawData = [int(d) for d in row[0].split(",")]
            rawData = [rawData[0]] + [d/255.0 for d in rawData[1:]]

            rawData = np.array(rawData)

            if i <= endIdx + 1:
                if training is None:
                    training = rawData
                else:
                    training = np.vstack((training, rawData))
            else:
                break
    return training

def load_digit2(filename, startIdx, endIdx):
    training = None
    i = 0
    print ""
    print "loading data",
    for x in open("data/train.csv").readlines()[startIdx:endIdx]:
        i += 1
        if i % 500 == 0:
            print ".",
        rawData = [int(d) for d in x.split(",")]
        rawData = [rawData[0]] + [d/255.0 for d in rawData[1:]]

        rawData = np.array(rawData)

        if training is None:
            training = rawData
        else:
            training = np.vstack((training, rawData))

    return training