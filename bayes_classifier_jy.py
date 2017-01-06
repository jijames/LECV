from numpy.random import randn
from PIL import Image
from numpy import *
from pylab import *
from os.path import basename
from scipy.cluster.vq import *
import sys, getopt, os, bayes, pca
import sift, dsift, imtools
import pickle, knn


##convert .png to .jpg files

path = 'test_hello'
filelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print "can't convert", infile

##create dsift files
# create a list of images
path = 'test_hello'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))


def read_features_labels(path):
	# create list of all files ending in .dsift
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	# read the features
	features = []
	for featfile in featlist:
		l,d = sift.read_features_from_file(featfile)
		features.append(d.flatten())
	features = array(features)

	# create labels
	labels = [featfile.split('/')[-1][0] for featfile in featlist]

	return features,array(labels)


##labeling
features,labels = read_features_labels('train_hello/')

test_features,test_labels = read_features_labels('test_hello/')

classnames = unique(labels)

V,S,m = pca.pca(features)


# keep most important dimensions
V = V[:50]
features = array([dot(V,f-m) for f in features])
test_features = array([dot(V,f-m) for f in test_features])


# test Bayes
bc = bayes.BayesClassifier()
blist = [features[where(labels==c)[0]] for c in classnames]

bc.train(blist,classnames)
res = bc.classify(test_features)[0]

acc = sum(1.0*(res==test_labels)) / len(test_labels)
print 'Accuracy:', acc


'''
def print_confusion(res,labels,classnames):

	n = len(classnames)
	
	# confusion matrix
	class_ind = dict([(classnames[i],i) for i in range(n)])

	confuse = zeros((n,n))
	for i in range(len(test_labels)):
		confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1

	print 'Confusion matrix for'
	print classnames
	print confuse

print_confusion(res,test_labels,classnames)
'''