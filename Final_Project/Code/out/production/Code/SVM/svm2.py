# Standard scientific Python imports
import matplotlib.pyplot as plt
import csv

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
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

print "start training"
print "data: ", len(data)
# training
trainingData = data[0:5000]
trainingLabel = label[0:5000]
clf = svm.SVC()
clf.fit(trainingData, trainingLabel) 

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.

# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(trainingData, trainingLabel)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(data[5000:5100])

for i in predicted:
    print i

# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)

# plt.show()