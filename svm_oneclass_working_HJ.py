print(__doc__)

import sys, os, imtools, shutil, pickle
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
import sift

typeFeats='dsift'
# dirTrain='features/train'
# dirTest='features/test'
dirTrain = '/home/joshua/Desktop/SingleClass/train/' # add by hyejun
dirTest = '/home/joshua/Desktop/SingleClass/test/' # add by hyejun
featlist = []

def read_feature_labels(path):
  featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
  features = []
  for featfile in featlist:
    l,d = sift.read_features_from_file(featfile)
    features.append(d.flatten())
  features = array(features)

  labels = [featfile.split('/')[-1][0] for featfile in featlist]

  return features,array(labels),featlist


def get_fileNameList_typeFeat(path):
  featlist = [f for f in os.listdir(path) if f.endswith('.'+typeFeats)]
  labels = [featfile for featfile in featlist]
  return array(labels)


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)

    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)

    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


# y are labels - not used for one-class
X_train, y_train, trainFN = read_feature_labels(dirTrain)
X_test, y_test, testFN = read_feature_labels(dirTest)
print y_train
print y_test

# print ("X_train shape: %s" % repr(X_train.shape))
# print ("X_test shape: %s" % repr(X_test.shape))
# kernel can be rbf, linear, sigmoid, precomputed
# nu 0.2 and linear are working best for me so far

""" svm = OneClassSVM(nu=0.2, kernel="linear", gamma=0.1)
    nu = 0.1 (best)
    traindata result -> Valid 138, Errors 23
    testdata result -> Valid 61, Errors 30, Never Errors 9
    
    nu = 0.2
    traindata result -> Valid 123, Errors 38
    testdata result -> Valid 60, Errors 25, Never Errors 15

    nu = 0.3
    traindata result -> Valid 107, Errors 54
    testdata result -> Valid 61, Errors 19, Never Errors 20

    nu = 0.4
    traindata result -> Valid 102, Errors 59
    testdata result -> Valid 60, Errors 17, Never Errors 23
    
    nu = 0.5
    traindata result -> Valid 83, Errors 78
    testdata result -> Valid 58, Errors 15, Never Errors 27
    
    nu = 0.6
    traindata result -> Valid 63, Errors 98
    testdata result -> Valid 57, Errors 9, Never Errors 34
    
    nu = 0.7
    traindata result -> Valid 47, Errors 114
    testdata result -> Valid 52, Errors 6, Never Errors 42
    
    nu = 0.8
    traindata result -> Valid 37, Errors 124
    testdata result -> Valid 49, Errors 2, Never Errors 49

    nu = 0.9
    traindata result -> Valid 17, Errors 144
    testdata result -> Valid 44, Errors 2, Never Errors 54

    nu = 1.0
    traindata result -> Valid 0, Errors 161
    testdata result -> Valid 46, Errors 0, Never Errors 54 """

""" svm = OneClassSVM(nu=0.2, kernel="linear", gamma=0.1)
    rbf -> X(All testdata display -1), poly -> X (Don't run), sigmoid -> X (All traindata and testdata display -1)
    precomputed -> X (error) """

""" svm = OneClassSVM(nu=0.2, kernel="linear", gamma=0.1)
    input gamma option 0.1 ~ 1.0, but same result """

""" svm = OneClassSVM(nu=0.1, kernel="poly", gamma=0.1, degree=3)
    no run """

""" svm = OneClassSVM(nu=0.1, kernel="poly", gamma=0.1, coef0=0.0)
    no run """

""" svm = OneClassSVM(nu=0.1, kernel="sigmoid", gamma=0.1, coef0=0.0)
    coef0=0.0
    traindata result -> Valid 0, Errors 161
    testdata result -> Valid 46, Errors 0, Never Errors 54 """


load_model = ''    # limit 24 character (include .pkl) 
save_model = 'CEM.pkl'    # limit 24 character (include .pkl)

# load model to file
if load_model:
  svm = pickle.load(open(load_model, "rb"))
  y_pred_train = svm.predict(X_train)
  y_pred_test = svm.predict(X_test)
  n_error_train = y_pred_train[y_pred_train == -1].size
  n_error_test = y_pred_test[y_pred_test == -1].size

# train model
else:
  svm = OneClassSVM(nu=0.1, kernel="linear", gamma=0.1, coef0=0.0, shrinking=True, tol=0.001,
                    cache_size=200, verbose=False, max_iter=-1, random_state=None)  # I think best working
  svm.fit(X_train)
  y_pred_train = svm.predict(X_train)
  y_pred_test = svm.predict(X_test)
  n_error_train = y_pred_train[y_pred_train == -1].size
  n_error_test = y_pred_test[y_pred_test == -1].size

  # save model to file
  if save_model:
    pickle.dump(svm, open(save_model, "wb"))


print "Training"
errors = 0
valid = 0
for i in range(len(y_pred_train)):
  print(trainFN[i] + " " + y_train[i] +" %d" % y_pred_train[i])
  if ((y_train[i] == "C") and (y_pred_train[i] == 1) or (y_train[i] == "H") and (y_pred_train[i] == -1)):
      valid = valid + 1
  else:
      errors = errors + 1

print "Valid %s" % valid
print "Errors %s" % errors

print " "
print "Testing"
errors = 0
valid = 0
nerrors = 0   # nerrors is veryvery bad classify count

for i in range(len(y_pred_test)):
  print(testFN[i] + " " + y_test[i] + " %d" % y_pred_test[i])
  if ((y_test[i] == "C") and (y_pred_test[i] == 1) or (y_test[i] == "H") and (y_pred_test[i] == -1)):
        valid = valid + 1

        #resultfolder = 'classify/O'
        #src = testFN[i][:-6] + '.jpg'

        #copyFile(src, resultfolder)

  elif ((y_test[i] == "C")  and (y_pred_test[i] == -1)):
        nerrors = nerrors + 1

        #resultfolder = 'classify/X'
        #src = testFN[i][:-6] + '.jpg'

        #copyFile(src, resultfolder)

  else:
        errors = errors + 1

        #resultfolder = 'classify/X'
        #src = testFN[i][:-6] + '.jpg'

        #copyFile(src, resultfolder)

  ''' if (y_test[i] == "C") and (y_pred_test[i] == 1):      // copy each folder code
    resultfolder = 'classify/Cat pass'
    src = testFN[i][:-6] + '.jpg'

    copyFile(src, resultfolder)

  elif (y_test[i] == "H") and (y_pred_test[i] == -1):
    resultfolder = 'classify/Human pass'
    src = testFN[i][:-6] + '.jpg'

    copyFile(src, resultfolder)

  elif ((y_test[i] == "C")  and (y_pred_test[i] == -1)):
    resultfolder = 'classify/Bad error'
    src = testFN[i][:-6] + '.jpg'

    copyFile(src, resultfolder)

  else:
    resultfolder = 'classify/Error'
    src = testFN[i][:-6] + '.jpg'

    copyFile(src, resultfolder) '''


print "Valid %s" % valid
print "Errors %s" % errors
print "Never Errors %s" % nerrors

print "End"
