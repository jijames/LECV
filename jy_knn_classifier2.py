from PIL import Image
from pylab import *
from numpy import *
import dsift, sift, knn
import os, pickle, scipy.misc



def read_gesture_features_labels(path):

	# create list of all files ending in .dsift
	featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

	# read the features
	features = []
	for featfile in featlist:
		l,d = sift.read_features_from_file(featfile)
		features.append(d.flatten())

	#set_printoptions(threshold=nan)	
	#print features	
	features = array(features)

	labels = [featfile for featfile in featlist]
	# labels = [featfile.split('/')[-1][0] for featfile in featlist]
	return features,array(labels)

trainfolder = 'train_hellojy/'
testfolder = 'test_hellojy/'
resultfolder1 = '1'
resultfolder2 = '2'

features, labels = read_gesture_features_labels(trainfolder)
test_features, test_labels = read_gesture_features_labels(testfolder)

load_model = ''
save_model = 'output_knn.pkl'

# k is nearest neighbors
k = 3

# load model to file
if load_model:
	with open(load_model, 'rb') as f:
		knn_classifier = pickle.load(f)

	print knn_classifier

# train model		
else:
	knn_classifier = knn.KnnClassifier(labels, features)

	print knn_classifier

	# save model to file
	if save_model:
		with open(save_model, 'wb') as f:
			pickle.dump(knn_classifier, f)

	for i in range(len(test_labels)):
		aa = knn_classifier.classify(features[i], k)
		#set_printoptions(threshold=nan)	 
		#print features[i]

		aaa = aa.split('/')[-1][0]
		print test_labels[i], ' -> ', aaa

		if(aaa == 'c'):
			cmmd = str('move ' + testfolder[:-1] + '\\' + test_labels[i].split('/')[-1][:-6] + '.jpg ' + resultfolder1)
			os.system(cmmd)

			print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder1
		if(aaa =='d'):
			cmmd = str('move ' + testfolder[:-1] + '\\' + test_labels[i].split('/')[-1][:-6] + '.jpg ' + resultfolder2)
			os.system(cmmd)

			print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder2



'''	
res = array([knn_classifier.classify(test_features[i], k) for i in range(len(test_labels))])

for i in range(len(test_labels)):
	print res[i],
	print test_labels[i]
	if(res[i].split('/')[-1][0] == test_labels[i].split('/')[-1][0]):
		""" im = Image.open(test_labels[i][:-6] + '.jpg')
		path = 'C:/Users/user/Desktop/BoB/Project/Source' """

		# im.save('C:/Users/user/Desktop/BoB/Project/Source/pass/')

		# im = Image.open(test_labels[i][:-6] + '.jpg').save(test_labels[i][:-6] + '.png')

		cmmd = str('move ' + testfolder[:-1] + '\\' + test_labels[i].split('/')[-1][:-6] + '.jpg ' + resultfolder)
		os.system(cmmd)

		print 'success', test_labels[i][:-6] + '.jpg', 'move to', resultfolder

		""" filename = test_labels[i].split('/')[-1][:-6] + '.jpg'
		dirname = 'pass'

		os.mkdir(dirname)
		cv2.imwrite(os.path.join(dirname, filename), im)

		os.chdir(dirname)
		cv2.imwrite(filename, im) """
'''

print 'End'
