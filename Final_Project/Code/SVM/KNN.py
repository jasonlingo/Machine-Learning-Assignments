from sklearn import svm
import csv
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.neighbors import NearestNeighbors


data = []
label = []
#load training data
with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0 
    for row in spamreader:
        if i == 0:
            i += 1
            continue
        d = [a for a in row[0].split(",")]
        label.append(d.pop(0))
        data.append(d)

num = 1000
print "start training"
print "data: ", len(data)
print "train data: ", num
# training
trainingData = np.array(data[0:num], 'float64')
trainingLabel = np.array(label[0:num], 'int')

# for d in trainingData:
#     print d

# clf = svm.SVC()
clf = NearestNeighbors()
# clf = LinearSVC()
# clf = OneVsRestClassifier(LinearSVC())
# clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(trainingData, trainingLabel)
print "end training"

# testData = []
# testLabel = []
# with open('test2.csv', 'rb') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     i = 0 
#     for row in spamreader:
#         if i == 0:
#             i += 1
#             continue
#         d = [a for a in row[0].split(",")]
#         testLabel.append(d.pop(0))
#         testData.append(d)



# predicting
print "start predicting"
testData = data[num:num+1000]
testLabel = label[num:num+1000]
correct = 0
for i in range(len(testData)):
    if i % 100 == 0:
        print ".",
    # print testLabel[i], clf.predict(np.array(testData[i], 'float64'))
    if np.array(testLabel[i], 'int') == clf.predict(np.array(testData[i], 'float64')):
        correct += 1

print "accuracy = ", correct / float(len(testData))
