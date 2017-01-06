print(__doc__)

import sys, os, imtools
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.svm import OneClassSVM
from os.path import basename
from decimal import *
import sift

typeFeats='dsift'
# dirTrain='features/train'
# dirTest='features/test'
dirTrain = '/home/joshua/Desktop/SingleClass/train/' # add by hyejun
dirTest = '/home/joshua/Desktop/SingleClass/test/' # add by hyejun
featlist = []

def read_feature_labels(path):
  featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.'+typeFeats)]
  features = []
  for featfile in featlist:
    l,d = sift.read_features_from_file(featfile)
    features.append(d.flatten())
  features = array(features)

  labels = [featfile.split('/')[-1][0] for featfile in featlist]

  return features,array(labels),featlist


def get_fileNameList(path):
  featlist = [f for f in os.listdir(path) if f.endswith('.'+typeFeats)]
  labels = [featfile for featfile in featlist]
  return array(labels)


# y are labels - not used for one-class
X_train, y_train, trainFN = read_feature_labels(dirTrain)
X_test, y_test, testFN = read_feature_labels(dirTest)
#print y_train
#print y_test
# print ("X_train shape: %s" % repr(X_train.shape))
# print ("X_test shape: %s" % repr(X_test.shape))
# kernel can be rbf, linear, sigmoid, precomputed
# nu 0.2 and linear are working best for me so far
svm = OneClassSVM(nu=0.2, kernel="linear", gamma=0.1)
svm.fit(X_train);
y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

print " "
print "Training"
errors = 0
valid = 0
total = 0
for i in range(len(y_pred_train)):
  total = total + 1
  #print(trainFN[i] + " " + y_train[i] +" %d" % y_pred_train[i])
  if ((y_train[i] == "1") and (y_pred_train[i] == 1) or (y_train[i] == "0") and (y_pred_train[i] == -1)):
      valid = valid + 1
  else:
      errors = errors + 1

print "Total %s" % total
print "Valid %s" % valid
print "Errors %s" % errors
error = errors / float(total)
print "Error: %s" % error

print " "
print "Testing"
errors = 0
valid = 0
total = 0
for i in range(len(y_pred_test)):
  total = total + 1
  #print(testFN[i] +  " + y_test[i] + " %d" % y_pred_test[i])
  if ((y_test[i] == "1") and (y_pred_test[i] == 1) or (y_test[i] == "0") and (y_pred_test[i] == -1)):
        valid = valid + 1
  else:
        errors = errors + 1

print "Total %s" % total
print "Valid %s" % valid
print "Errors %s" % errors
error = errors / float(total)
print "Error: %s" % error
