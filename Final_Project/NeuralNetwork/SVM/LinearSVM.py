import numpy as np
import csv
import sklearn
from sklearn.svm import SVC


data = []
label = []
num = 30000
testNum = 10000
#load training data
with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        if i == 0:
            i += 1
            continue
        if i >= (num + testNum + 1):
            break
        else:
            i += 1

        rawData = row[0].split(",")

        label.append(int(rawData[0]))
        data.append([])
        for d in rawData[1:]:
            data[-1].append(int(d))


print "start training"
print "data: ", len(data)
print "train data: ", num
# training

X = np.array(data[0:num])
# trainingData = np.array(data[0:num], 'float64')
# trainingData = data[0:num]
y = np.array(label[0:num])


clf = sklearn.svm.LinearSVC()
clf.fit(X, y)

correct = 0

for i in range(0, num):
    # print label[i+1], (clf.predict(np.array([data[i+1]])))[0]
    if label[i] == (clf.predict(np.array([data[i]])))[0]:
        correct += 1
print "training accuracy"
print correct / float(num)

correct = 0

for i in range(num, num+testNum):
    # print label[i+1], (clf.predict(np.array([data[i+1]])))[0]
    if label[i] == (clf.predict(np.array([data[i]])))[0]:
        correct += 1
print "testing accuracy"
print correct / float(testNum)



# prettyPicture

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# def submitAccuracy():
#     return acc





# # clf = SVC(kernel='linear') #0.93
# # clf = SVC(kernel='poly') #0.98
# # clf = SVC()
# clf = sklearn.svm.LinearSVC()
# clf.fit(X, y) 


# correct = 0
# for i in range(num, num+testNum):
#     print label[i+1], (clf.predict(np.array([data[i+1]])))[0]
#     if label[i+1] == (clf.predict(np.array([data[i+1]])))[0]:
#         correct += 1

# print correct / float(testNum)



