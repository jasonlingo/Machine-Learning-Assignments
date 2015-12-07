"""
Methods for loading the data sets as numpy matrices.
"""

import numpy as np
import random
import csv



def load_digit(filename, startIdx, endIdx):
    """
    Load the hand written digit features into a 7
    Args:
        data_ratio: the ratio of examples that go into the training set,
                    and the rest of the examples will be the validation set.
                    For the testing data set, this ratio should be set to 1.
    Returns:
        a tuple of numpy matrices, the first in the tuple is the training
        data set, the second is the validation data set
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
            rawData = [rawData[0]] + [d/100.0 for d in rawData[1:]]

            rawData = np.array(rawData)

            if i <= endIdx + 1:
                if training is None:
                    training = rawData
                else:
                    training = np.vstack((training, rawData))
            else:
                break
    return training
