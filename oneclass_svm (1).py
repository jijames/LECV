print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

from numpy.random import randn
from PIL import Image
from numpy import *
from pylab import *
from os.path import basename
from scipy.cluster.vq import *
import sys, getopt, os, bayes, pca
import sift, dsift, imtools
import pickle, knn


##convert file extensions
'''
path = 'test_AB'
filelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.ppm')]

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print "can't convert", infile

##create dsift files

# create a list of images
path = 'test_AB'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))


# create a list of images
path = 'train_AB'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))
'''


def read_features_labels(path):
	# create list of all files ending in .dsift
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	# read the features
	X = []
	for featfile in featlist:
		l,d = sift.read_features_from_file(featfile)
		X.append(d.flatten())
	X = array(X)

	# create labels
	labels = [featfile.split('/')[-1][0] for featfile in featlist]

	return X,array(labels)


##labeling
X,labels = read_features_labels('train_hello/')
X_train = np.r_[X + 2, X - 2]
X,test_labels = read_features_labels('test_hello/')
X_test = np.r_[X + 2, X - 2]
classnames = unique(labels)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

'''
X = 0.3 * np.random.randn(100, 2)
X.shape[1] = 10368
'''

'''
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
'''

plt.title("Novelty Detection")
plt.contourf(xx, yy, levels=np.linspace(-100, 0, 7), cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, levels=[0], linewidths=2, colors='red')
plt.contourf(xx, yy, levels=[0, 100], colors='orange')

'''
plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
'''

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2],
           ["learned frontier", "training image",
            "test image"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))

plt.xlabel(
    "error train: %d/20 ; errors novel regular: %d/4 ; "
    % (n_error_train, n_error_test))
plt.show()
