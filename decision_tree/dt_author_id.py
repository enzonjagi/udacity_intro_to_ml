#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
def set_up_classifier(min_sample_split):
    from sklearn import tree

    # setup the classify
    clf = tree.DecisionTreeClassifier(min_samples_split=min_sample_split)
    return clf

# calculate accuracy
def check_for_accuracy(pred):
    '''Calculating the accuracy'''
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, pred)
    return accuracy
# classifier
clf = set_up_classifier(40)
# fit classifier
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# calculate accuracy
acc = check_for_accuracy(pred)
print(f'Model Accuracy: {acc}')
# features
number_of_features = len(features_train[0])
print(f'Number of Features: {number_of_features}')

#########################################################


