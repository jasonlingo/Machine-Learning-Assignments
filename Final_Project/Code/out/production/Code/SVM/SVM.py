from sklearn import svm
import csv
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

data = []
label = []
num = 10000
testNum = 100
#load training data
with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0 
    for row in spamreader:
        if i == 0:
            i += 1
            continue
        if i >= (num + testNum) * 2:
            break
        else:
            i += 1
        d = [float(a) for a in row[0].split(",")]
        label.append(int(d.pop(0)))
        data.append(d)


print "start training"
print "data: ", len(data)
print "train data: ", num
# training
trainingData = np.array(data[0:num])
# trainingData = np.array(data[0:num], 'float64')
# trainingData = data[0:num]
trainingLabel = label[0:num]
# trainingLabel = np.array(label[0:num], 'int')
# trainingLabel = label[0:num]

# for d in trainingData:
#     print d
"""
C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma=gamma_range, C=C_range)

grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=StratifiedKFold(y=trainingLabel))

grid.fit(trainingData, trainingLabel)
"""

clf = svm.SVC(kernel="linear")
# clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, \
#     gamma=1.0000000000000001e-05, kernel='rbf', max_iter=-1, \
#     probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
# clf = LinearSVC()
# clf = OneVsRestClassifier(LinearSVC())
# clf = OneVsRestClassifier(LinearSVC(random_state=0))
clf.fit(trainingData, trainingLabel)
# print("The best classifier is: ", grid.best_estimator_)

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
testData = data[num:num+testNum]
testLabel = label[num:num+testNum]
correct = 0
for i in range(len(testData)):
    if i % 100 == 0:
        print ".",

    print testLabel[i], clf.predict([testData[i]])[0]
    # if np.array(testLabel[i], 'int') == clf.predict(np.array(testData[i], 'float64'))[0]:
    if testLabel[i] == clf.predict(np.array(testData[i]))[0]:
        correct += 1

print "accuracy = ", correct / float(len(testData))
