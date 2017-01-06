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
'''
path = 'test/'
filelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print "can't convert", infile
'''

##Directory Setting##
script_path = os.path.dirname(sys.argv[0]) + '/'
train_path = script_path + 'train_hello/'
test_path = script_path + 'test_hello/'

##Print Directory##
print '\nScript Directory : ' + script_path + '\n'
print 'Train Images Directory : ' + train_path
print 'Test Images Directory : ' + test_path + '\n'


# create a list of images

train_image_list = [os.path.join(train_path,f) for f in os.listdir(train_path) if f.endswith('.jpg')]
print "Train Image : %d" % len(train_image_list)
for filename in train_image_list:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))


test_image_list = [os.path.join(test_path,f) for f in os.listdir(test_path) if f.endswith('.jpg')]
print "Test Image : %d\n" % len(test_image_list)
for filename in test_image_list:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))

def read_features_labels(path):
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
        
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)
    labels = [featfile.split('/')[-1][0] for featfile in featlist]

    return features,array(labels)

def print_confusion(res,labels,classnames):
    n = len(classnames)
        
    class_ind = dict([(classnames[i],i) for i in range(n)])
            
    confuse = zeros((n,n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1
                        
    print 'Confusion matrix for'
    print confuse

##labeling
features,labels = read_features_labels(train_path)
test_features,test_labels = read_features_labels(test_path)
classnames = unique(labels)

V,S,m = pca.pca(features)

# keep most important dimensions
V = V[:50]
features = array([dot(V,f-m) for f in features])
test_features = array([dot(V,f-m) for f in test_features])

bc = bayes.BayesClassifier()
blist = [features[where(labels==c)[0]] for c in classnames]
            
bc.train(blist,classnames)

print len(test_features)
print bc.classify(test_features)
'''
for i in range(0, len(test_features)):
    res = bc.classify(test_features)[0]
    acc = str(res[i]==test_labels[i])
    #acc = sum(1.0*(res==test_labels)) / len(test_labels)
    print 'Accuracy:', acc, res, test_labels
'''
#print_confusion(res,test_labels,classnames)
