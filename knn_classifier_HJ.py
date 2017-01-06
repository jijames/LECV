from PIL import Image
from pylab import *
from numpy import *
import dsift, sift, knn
import os, pickle, shutil, scipy.misc

def read_gesture_features_labels(path):

	# create list of all files ending in .dsift
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	# read the features
	features = []
	for featfile in featlist:
		l,d = sift.read_features_from_file(featfile)
		features.append(d.flatten())

	features = array(features)

	labels = [featfile for featfile in featlist]
    #labels = [featfile.split('/')[-1][0] for featfile in featlist]

	return features, array(labels)


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


n = 2 #input("number of train fodler : ")

choicefolder = ['classify/Cat/', 'classify/Human/']
trainfolder = 'classify/Cat and Human and Dog train/'
testfolder = 'classify/Cat and Human and Dog test/'
resultfolder = 'pass'

# copy choicefolder to trainfolder
''' for i in range(n):
	copyDirectory(choicefolder[i][:-1], trainfolder[:-1]) '''

features, labels = read_gesture_features_labels(trainfolder)
test_features, test_labels = read_gesture_features_labels(testfolder)

load_model = ''
save_model = ''# 'output_knn_2014.11.09_13_00.pkl'

# k is nearest neighbors
# try k up to 10. 2, 3, --- 10.
k = 3

""" Cat and Human (Class 2)

	k = 2 
    Cat pass result : Cat -> 33, Human -> 5, Other -> 0
	Human pass result : Cat -> 21, Human -> 41, Other -> 0

	k = 3
	Cat pass result : Cat -> 24, Human -> 2, Other -> 0
	Human pass result : Cat -> 30, Human -> 44, Other -> 0

	k = 4
	Cat pass result : Cat -> 16, Human -> 1, Other -> 0
	Human pass result : Cat -> 38, Human -> 45, Other -> 0

	k = 5
	Cat pass result : Cat -> 10, Human -> 1, Other -> 0
	Human pass result : Cat -> 44, Human -> 45, Other -> 0

	k = 6
	Cat pass result : Cat -> 6, Human -> 1, Other -> 0
	Human pass result : Cat -> 48, Human -> 45, Other -> 0

	k = 7
	Cat pass result : Cat -> 4, Human -> 1, Other -> 0
	Human pass result : Cat -> 50, Human -> 45, Other -> 0

	k = 8
	Cat pass result : Cat -> 2, Human -> 1, Other -> 0
	Human pass result : Cat -> 52, Human -> 45, Other -> 0

	k = 9
	Cat pass result : Cat -> 1, Human -> 1, Other -> 0
	Human pass result : Cat -> 53, Human -> 45, Other -> 0

	k = 10
	Cat pass result : Cat -> 0, Human -> 1, Other -> 0
	Human pass result : Cat -> 54, Human -> 45, Other -> 0 """


""" Cat and Human and Dog (Class 3)

	k = 3
	Cat pass result : Cat -> 17, Human -> 0, Dog -> 8, Other -> 0
	Human pass result : Cat -> 22, Human -> 42, Dog -> 16, Other -> 0
	Dog pass result : Cat -> 15, Human -> 4, Dog -> 15, Other -> 0

	k = 4
	Cat pass result : Cat -> 10, Human -> 0, Dog -> 4, Other -> 0
	Human pass result : Cat -> 33, Human -> 44, Dog -> 22, Other -> 0
	Dog pass result : Cat -> 11, Human -> 2, Dog -> 13. Other -> 0

	k = 5
	Cat pass result : Cat -> 7, Human -> 0, Dog -> 2, Other -> 0
	Human pass result : Cat -> 40, Human -> 45, Dog -> 25, Other -> 0
	Dog pass result : Cat -> 7, Human -> 1, Dog -> 12, Other -> 0

	k = 6
	Cat pass result : Cat -> 2, Human -> 0, Dog -> 0, Other -> 0
	Human pass result : Cat -> 43, Human -> 45, Dog -> 29, Other -> 0
	Dog pass result : Cat -> 9, Human -> 1, Dog -> 10, Other -> 0

	k = 7
	Cat pass result : Cat -> 1, Human -> 0, Dog -> 0, Other -> 0
	Human pass result : Cat -> 46, Human -> 45, Dog -> 31, Other -> 0
	Dog pass result : Cat -> 7, Human -> 1, Dog -> 8, Other -> 0

	k = 8
	Cat pass result : Cat -> 1, Human -> 0, Dog -> 0, Other -> 0
	Human pass result : Cat -> 47, Human -> 45, Dog -> 33, Other -> 0
	Dog pass result : Cat -> 6, Human -> 1, Dog -> 6, Other -> 0

	k = 9
	Cat pass result : Cat -> 1, Human -> 0, Dog -> 0, Other -> 0
	Human pass result : Cat -> 51, Human -> 45, Dog -> 33, Other -> 0
	Dog pass result : Cat -> 2, Human -> 1, Dog -> 6, Other -> 0

	k = 10
	Cat pass result : Cat -> 0, Human -> 0, Dog -> 0, Other -> 0
	Human pass result : Cat -> 54, Human -> 45, Dog -> 34, Other -> 0
	Dog pass result : Cat -> 0, Human -> 1, Dog -> 5, Other -> 0

"""

# load model to file
if load_model:
	with open(load_model, 'rb') as f:
		knn_classifier = pickle.load(f)

	print knn_classifier

# train model
else:
	''' for i in range(n):
		features, labels = read_gesture_features_labels(choicefolder[i])
		test_features, test_labels = read_gesture_features_labels(testfolder)

		knn_classifier = knn.KnnClassifier(labels, features) '''

	knn_classifier = knn.KnnClassifier(labels, features)

	print knn_classifier

	# save model to file
	if save_model:
		with open(save_model, 'wb') as f:
			pickle.dump(knn_classifier, f)

Cat_c = 0
Cat_h = 0
Cat_d = 0
Cat_o = 0
Human_c = 0
Human_h = 0
Human_d = 0
Human_o = 0
Dog_c = 0
Dog_h = 0
Dog_d = 0
Dog_o = 0

for i in range(len(test_labels)):
	test = knn_classifier.classify(test_features[i], k)
	print test 

	if test.split('/')[-1][0] == 'C' or test.split('/')[-1][0] == 'c':
		resultfolder = 'classify/Cat pass'
		src = testfolder + test_labels[i].split('/')[-1][:-6] + '.jpg'

		copyFile(src, resultfolder)

		if test_labels[i].split('/')[-1][0] == 'C' or test_labels[i].split('/')[-1][0] == 'c':
			Cat_c = Cat_c + 1

		elif test_labels[i].split('/')[-1][0] == 'H' or test_labels[i].split('/')[-1][0] == 'h':
			Cat_h = Cat_h + 1

		elif test_labels[i].split('/')[-1][0] == 'D' or test_labels[i].split('/')[-1][0] == 'd':
			Cat_d = Cat_d + 1

		else:
			Cat_o = Cat_o + 1

		print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder

	elif test.split('/')[-1][0] == 'H' or test.split('/')[-1][0] == 'h':
		resultfolder = 'classify/Human pass'
		src = testfolder + test_labels[i].split('/')[-1][:-6] + '.jpg'

		copyFile(src, resultfolder)

		if test_labels[i].split('/')[-1][0] == 'C' or test_labels[i].split('/')[-1][0] == 'c':
			Human_c = Human_c + 1

		elif test_labels[i].split('/')[-1][0] == 'H' or test_labels[i].split('/')[-1][0] == 'h':
			Human_h = Human_h + 1

		elif test_labels[i].split('/')[-1][0] == 'D' or test_labels[i].split('/')[-1][0] == 'd':
			Human_d = Human_d + 1

		else:
			Human_o = Human_o + 1

		print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder

	elif test.split('/')[-1][0] == 'D' or test.split('/')[-1][0] == 'd':
		resultfolder = 'classify/Dog pass'
		src = testfolder + test_labels[i].split('/')[-1][:-6] + '.jpg'

		copyFile(src, resultfolder)

		if test_labels[i].split('/')[-1][0] == 'C' or test_labels[i].split('/')[-1][0] == 'c':
			Dog_c = Dog_c + 1

		elif test_labels[i].split('/')[-1][0] == 'H' or test_labels[i].split('/')[-1][0] == 'h':
			Dog_h = Dog_h + 1

		elif test_labels[i].split('/')[-1][0] == 'D' or test_labels[i].split('/')[-1][0] == 'd':
			Dog_d = Dog_d + 1

		else:
			Dog_o = Dog_o + 1

		print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder


print 'Cat pass result : '
print "Cat -> %s" % Cat_c
print "Human -> %s" % Cat_h
print "Dog -> %s" % Cat_d
print "Other -> %s" % Cat_o

print 'Human pass result : '
print "Cat -> %s" % Human_c
print "Human -> %s" % Human_h
print "Dog -> %s" % Human_d
print "Other -> %s" % Human_o

print 'Dog pass result : '
print "Cat -> %s" % Dog_c
print "Human -> %s" % Dog_h
print "Dog -> %s" % Dog_d
print "Other -> %s" % Dog_o

''' res = array([knn_classifier.classify(test_features[i], k) for i in range(len(test_labels))])

for i in range(len(test_labels)):
	print res[i],
	print test_labels[i]
	# if(res[i].split('/')[-1][0] == test_labels[i].split('/')[-1][0]):
	for j in range(len(labels)):
		if(res[i] == labels[j]):
			""" im = Image.open(test_labels[i][:-6] + '.jpg')
			path = 'C:/Users/user/Desktop/BoB/Project/Source' """

			# im.save('C:/Users/user/Desktop/BoB/Project/Source/pass/')

			# im = Image.open(test_labels[i][:-6] + '.jpg').save(test_labels[i][:-6] + '.png')

			resultfolder = 'C ' + resultfolder
			cmmd = str('move ' + testfolder[:-1] + '\\' + test_labels[i].split('/')[-1][:-6] + '.jpg ' + resultfolder)
			os.system(cmmd)

			print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder '''

print 'End'
