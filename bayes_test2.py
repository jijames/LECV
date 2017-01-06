import sys, getopt, os, bayes, pca
from PIL import Image
from numpy import *
from pylab import *
import sift, dsift, imtools
from os.path import basename
import pickle
from scipy.cluster.vq import *
import bayes

""" .png file -> .jpg file """

path = 'train_hello'

filelist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

for infile in filelist:
    outfile = os.path.splitext(infile)[0] + ".jpg"
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print "can't convert", infile



""" test_hello, train_hello folder create dsift file"""

# create a list of images
path = 'test_hello'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))


# create a list of images
path = 'train_hello'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))



""" bayes classifier """

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
    labels = [featfile.split('/')[-1][0] for featfile in featlist]
    return features, labels
    # return features,array(labels)

bc = bayes.BayesClassifier()

#features = read_gesture_features_labels('train_hello/')
features, labels = read_gesture_features_labels('train_hello/')
# test_features = read_gesture_features_labels('test_hello/')
test_features, test_labels = read_gesture_features_labels('test_hello/')

name = unique(features)
# blist = [features[0] for c in name]
blist = [features[where(labels==c)[0]] for c in name]

bc.train(blist,name)
test = bc.classify(test_features)[0]

path = 'test_hello'
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imnbr = len(imlist)

for i in range(len(test)):
    for filename in imlist:
        if math.cell(test[i]):
            im = Image.open(filename).convert('L')
            im.save('pass/' + filename[-3] + '.txt')
            # savetxt('test.txt', c, '%i')
            
print 'End'
