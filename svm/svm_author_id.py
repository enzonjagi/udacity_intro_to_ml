#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

from itertools import count
from sklearn import svm
from time import time
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
def check_for_accuracy(pred):
    '''Calculating the accuracy'''
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    return accuracy

# create the classifier
clf = svm.SVC(kernel='rbf', C=10000)
# train the classifier
clf.fit(features_train, labels_train)
# make predictions based on the trained model
pred = clf.predict(features_test)
def count_predictions(pred, num):
    '''Counts the predictions of a class'''
    count = 0 
    for i in pred:
        if i == num:
            count += 1
    return count
chris = count_predictions(pred=pred, num=1)
print(f'Chris: {chris}')
# check accuracy of the model
accuracy = check_for_accuracy(pred)
print(f'Full data set Accuracy: {accuracy}')

#########################################################
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''
'''
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
# create the classifier
clf = svm.SVC(kernel='rbf', C=10000)
# train the classifier
clf.fit(features_train, labels_train)
# make predictions based on the trained model
pred = clf.predict(features_test)
# Datapoints
# 10
answer10 = pred[10]
print(f'Class for 10: {answer10}')
# 26
answer26 = pred[26]
print(f'Class for 26: {answer26}')
# 50
answer50 = pred[50]
print(f'Class for 50: {answer50}')

# check accuracy of the model
accuracy = check_for_accuracy(pred)
print(f'1% data set Accuracy: {accuracy}')

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
'''
#########################################################
