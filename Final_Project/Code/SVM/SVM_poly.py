import numpy as np
import csv

data = []
label = []
num = 1000
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

        rawData = row[0].split(",")

        label.append(int(rawData[0]))
        # d = [float(a) for a in row[0].split(",")]
        # label.append(int(d.pop(0)))
        # data.append(d)
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

print X
print y

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(kernel='poly')
clf.fit(X, y) 


correct = 0
for i in range(num, num+testNum):
    print label[i+1], (clf.predict(np.array([data[i+1]])))[0]
    if label[i+1] == (clf.predict(np.array([data[i+1]])))[0]:
        correct += 1

print correct / float(testNum)



# prettyPicture

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# def submitAccuracy():
#     return acc

