from PIL import Image
from pylab import *
from numpy import *
from numpy.random import randn
from svmutil import *
import imtools, os, pickle
import dsift, sift

# svm classifier test

def read_gesture_features_labels(path):

	# create list of all files ending in .dsift
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	# read the features
	features = []
	for featfile in featlist:
		l,d = sift.read_features_from_file(featfile)
		features.append(d.flatten())
	features = array(features)

	# create labels
	labels = [featfile for featfile in featlist]
	return features, array(labels)

def convert_labels(keys,dict):
  	""" keys should be an array of keys """
  	newKeys = []
  	for index, key in enumerate(keys):
  		newKeys.append(dict[key])

  	return newKeys

def get_fileNameList(path):
	featlist = [f for f in os.listdir(path) if f.endswith('.' + 'dsift')]
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

 
def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)

    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)

    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)


trainfolder = 'features/train/'
testfolder = 'features/test/'
resultfolder = 'pass'

features, labels = read_gesture_features_labels(trainfolder)
test_features, test_labels = read_gesture_features_labels(testfolder)

features = map(list, features)
test_features = map(list, test_features)

classnames = unique(labels)

load_model = ''
save_model = 'output_svm.pkl'

# create conversion function for the labels
transl = {}
for i,c in enumerate(classnames):
    transl[c], transl[i] = i, c


# load model to file
if load_model:
	m = svm_load_model(load_model)
	print m

# train model
else:
	prob = svm_problem(convert_labels(labels, transl), features)
	param = svm_parameter('-t 0')
	m = svm_train(prob, param)

	# save model to file
	if save_model:
		svm_save_model(save_model, m)


res = svm_predict(convert_labels(labels, transl), features, m)

noLabel_tmpList = []
for i in range(len(test_labels)):
	noLabel_tmpList.append(0)
	input_labels = noLabel_tmpList

res = svm_predict(input_labels, test_features, m)[0]

fileNameList = get_fileNameList(testfolder)

for i in range(len(res)):
	print fileNameList[i], " : ", res[i]
	if(res[i] < 5):
		# im = Image.open(test_labels[i][:-6] + '.jpg').save(test_labels[i][:-6] + '.png')

		#cmmd = str('move ' + testfolder[:-1] + '\\' + test_labels[i].split('/')[-1][:-6] + '.jpg ' + resultfolder)
		#os.system(cmmd)

		print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder


print 'End'
