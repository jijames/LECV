from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
from pylab import *
from PIL import Image
import sys, os, imtools, shutil, pickle, getopt, sift, dsift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

dirTest = ''
load_model = ''    # limit 24 character (include .pkl)


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


filelist = get_fileNameList(dirTest)

# png, bmp, gif changed to jpg (Test)
for infile in filelist: 
    outfile = os.path.splitext(infile)[0] + ".jpg"

    if infile != outfile:
        try:
            Image.open(infile).save(outfile)

        except IOError:
            print "Can't convert", infile

        except Exception:
            pass


# create dsift file (Test)
imlist = [os.path.join(dirTest, f) for f in os.listdir(dirTest) if f.endswith('.jpg')]

for filename in imlist:
    featfile = filename[:-4] + '.dsift'
    featfile = featfile.replace(" ", "")

    try:
      if not os.path.isfile(featfile):
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100,100)) # process images at fixed size (100,100)

    except Exception:
      pass

svm_train, featlist = read_feature_labels(dirTest)


# Make Directory
dirname = os.path.join('C:\Users', os.getenv('USERNAME'), 'Desktop\Result')

if not os.path.isdir(dirname):
    os.mkdir(dirname)
    

# load model to file
if load_model:
  svm = pickle.load(open(load_model, "rb"))
  svm_test = svm.predict(svm_train)


valid = 0
errors = 0


for i in range(len(svm_test)):
  if (svm_test[i] == 1):
        valid = valid + 1
        src = featlist[i][:-6] + '.jpg'

        copyFile(src, dirname)

  else:
        errors = errors + 1


print "Valid %s" % valid
print "Errors %s" % errors
print "%d Images founded." % valid

now = time.localtime()

f = open(os.path.join(dirname, 'Report.txt'), 'a')
result = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec) + '  ' + 'Result : ' + str(valid) + '\n'
f.write(result)

f.close()

print "End"
