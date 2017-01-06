from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from os.path import basename
from pylab import *
from PIL import Image
import sys, os, imtools, shutil, pickle, getopt, sift, dsift, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

dirTest = ''

def print_error():
  print("Usage: %s -e test_dir" % sys.argv[0])
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
opts, args = getopt.getopt(sys.argv[1:], "e:")

###############################
# o == option
# a == argument passed to the o
###############################

for o, a in opts:
    if o == '-e':
        dirTest = a

    else:
        print_error()


# load model to file

load_model = raw_input("Input load model name : ")

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
              if os.path.isfile(outfile) and infile[-6:] != '.dsift':
                os.remove(outfile)

          except Exception:
              pass    #unknown exception handling
            
             
  # create dsift file (Test)
  imlist = [os.path.join(dirTest, f) for f in os.listdir(dirTest) if f.endswith('.jpg')]
  
  for filename in imlist:
    featfile = filename[:-4] + '.dsift'
    featfile = featfile.replace(" ", "")

    try:  
      if not os.path.isfile(featfile):
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(100, 100)) # process images at fixed size (100,100)

    except Exception:
      pass
    
  
  svm = pickle.load(open(load_model, "rb"))
  svm_test_features, filelist = read_feature_labels(dirTest)
  svm_test_result = svm.predict(svm_test_features)


else:
  print "Please input load_model name"
  print_error()
    

# Make Directory
dirname = os.path.join('C:\Users', os.getenv('USERNAME'), 'Desktop\Result')

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
print "%d Images founded." % valid

now = time.localtime()

f = open(os.path.join(dirname, 'Report.txt'), 'a')
result = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec) + '  ' + 'Result : ' + str(valid) + '\n'
f.write(result)

f.close()

print "End"
End = raw_input("Please Press Enter Key")
