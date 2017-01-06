from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
from pylab import *
from PIL import Image
import sys, os, imtools, shutil, pickle, getopt, sift, dsift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

# dirTrain = ''
dirTest = ''
load_model = ''    # limit 24 character (include .pkl) 
# save_model = ''    # limit 24 character (include .pkl)


def print_error():
  print("Usage: %s -m load_model name -e test_dir" % sys.argv[0])
  print("    -m : load model name")
  # print("    -r : Directory of training features")
  print("    -e : Directory of test features")
  sys.exit(1)


def read_feature_labels(path):
  featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]
  features = []
  for featfile in featlist:
    l, d = sift.read_features_from_file(featfile)
    features.append(d.flatten())

  features = array(features)

  return features, featlist


def get_fileNameList(path):
  featlist = [os.path.join(path, f) for f in os.listdir(path)]

  return array(featlist)


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)

    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)

    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


# Read command line args
opts, args = getopt.getopt(sys.argv[1:], "m:e:")

###############################
# o == option
# a == argument passed to the o
###############################

for o, a in opts:
    if o == '-m':
        load_model = a

    # elif o == '-r':
        # dirTrain = a

    elif o == '-e':
        dirTest = a

    else:
        print_error()


# load model to file
if load_model:

  if not load_model or not os.path.isfile(load_model):
    print "No/invalid load model specified"
    print_error()

  if not dirTest or not os.path.isdir(dirTest):
    print "No/invalid test directory specified"
    print_error()


  # png, bmp, gif changed to jpg (Test)
  filelist = get_fileNameList(dirTest)
  
  for infile in filelist: 
      outfile = os.path.splitext(infile)[0] + ".jpg"

      if infile != outfile:
          try:
              Image.open(infile).save(outfile)

          except IOError:
              if os.path.isfile(outfile):
                os.remove(outfile)
   
            
  # create dsift file (Test)
  imlist = [os.path.join(dirTest, f) for f in os.listdir(dirTest) if f.endswith('.jpg')]
  imnbr = len(imlist)

  for filename in imlist:
      filename = filename.replace(" ", "")
      featfile = filename[:-4] + '.dsift'
      dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100, 100)) # process images at fixed size (100,100)

  
  svm = pickle.load(open(load_model, "rb"))
  svm_test_features, filelist = read_feature_labels(dirTest)
  svm_test_result = svm.predict(svm_test_features)

else:

  print "Please input load_model name"
  print_error()

  ''' if not dirTrain or not os.path.isdir(dirTrain):
    print "No/invalid training directory specified"
    print_error()

  if not dirTest or not os.path.isdir(dirTest):
    print "No/invalid test directory specified"
    print_error()
  

  # png, bmp, gif changed to jpg (Train)
  filelist = get_fileNameList(dirTrain)

  for infile in filelist: 
      outfile = os.path.splitext(infile)[0] + ".jpg"

      if infile != outfile:
          try:
              Image.open(infile).save(outfile)

          except IOError:
              if os.path.isfile(outfile):
                os.remove(outfile)


  # png, bmp, gif changed to jpg (Test)
  filelist = get_fileNameList(dirTest)

  for infile in filelist: 
      outfile = os.path.splitext(infile)[0] + ".jpg"

      if infile != outfile:
          try:
              Image.open(infile).save(outfile)

          except IOError:
              if os.path.isfile(outfile):
                os.remove(outfile)


  # create dsift file (Train)
  imlist = [os.path.join(dirTrain, f) for f in os.listdir(dirTrain) if f.endswith('.jpg')]
  imnbr = len(imlist)

  for filename in imlist:
      filename = filename.replace(" ", "")
      featfile = filename[:-4] + '.dsift'
      dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100, 100)) # process images at fixed size (100,100)


  # create dsift file (Test)
  imlist = [os.path.join(dirTest, f) for f in os.listdir(dirTest) if f.endswith('.jpg')]
  imnbr = len(imlist)

  for filename in imlist:
      filename = filename.replace(" ", "")
      featfile = filename[:-4] + '.dsift'
      dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100, 100)) # process images at fixed size (100,100)


  # y are labels - not used for one-class
  svm_train_features = read_feature_labels(dirTrain)
  svm_test_features, filelist = read_feature_labels(dirTest)
  svm = OneClassSVM(nu=0.1, kernel="linear", gamma=0.1, coef0=0.0, shrinking=True, tol=0.001,
                    cache_size=200, verbose=False, max_iter=-1, random_state=None)
  svm.fit(svm_train_features)
  svm_test_result = svm.predict(svm_test_features)

  if save_model:
    pickle.dump(svm, open(save_model, "wb")) '''
    

# Make Directory
dirname = 'C:\Users\user \Desktop\Check'

valid = 0
errors = 0

if not os.path.isdir(dirname):
    os.mkdir(dirname)


for i in range(len(svm_test_result)):
  if (svm_test_result[i] == 1):
        valid = valid + 1
        src = filelist[i][:-6] + '.jpg'

        copyFile(src, dirname)

  else:
        errors = errors + 1
        

print "Valid %s" % valid
print "Errors %s" % errors

print "End"
