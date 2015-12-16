import numpy as np
import csv


def load_csv(filename, startIdx, endIdx):
    """
    Load the data from csv file into a numpy format.
    Args:
        filename: the input csv file
        startIdx: the starting index of the data to be loaded
        endIdx: the ending index of the data to be loaded
    Returns:
        A numpy matrix containing examples. The dimension of the matrix is (examples x features)
    """
    print ""
    print "loading data",
    data = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        for row in spamreader:
            if i < startIdx:
                i += 1
                continue
            if i >= endIdx:
                break

            if i % 500 == 0:
                print ".",
            rawData = [int(d) for d in row[0].split(",")]
            # feature scaling
            rawData = [rawData[0]] + [d/255.0 for d in rawData[1:]]
            data.append(rawData)
            i += 1

    return np.array(data)
