from sklearn import tree, ensemble
from skimage import feature
from skimage import io, transform
import os
import numpy

#Tags
tags = {1:'sky', 2:	'greenery' , 3:'building'}#, 4:'crowd'}
tc = 3

#LBP Features
radius = 1
n_points = 8
METHOD = 'default'

a = numpy.zeros(shape = (tc*20, 15000))

dt = ensemble.RandomForestClassifier()
#dt = tree.DecisionTreeClassifier()

def trainSys():
	global a
	global tags
	i = 0
	print "Starting training..."

	directory = "/home/leroy/Desktop/ATI - Mini/datasets/"
	for key, ta in tags.iteritems():
		folder = directory + ta + '/'
		for file in os.listdir(folder):
			file2 = folder + file
			img = io.imread(file2, as_grey = True)
			img = transform.resize(img, (100, 150))
			lbp = feature.local_binary_pattern(img, n_points, radius, METHOD)
			lbp = lbp.flatten()
			a[i] = lbp
			i = i + 1

	input_images_tags = numpy.ones(20)
	input_images_tags2 = numpy.empty(20)
	input_images_tags2.fill(2)
	input_images_tags3 = numpy.empty(20)
	input_images_tags3.fill(3)
	input_images_tags4 = numpy.empty(20)
	input_images_tags4.fill(4)
	
	#input_images_tags3 = numpy.concatenate([input_images_tags3, input_images_tags4])
	input_images_tags2 = numpy.concatenate([input_images_tags2, input_images_tags3])
	input_images_tags = numpy.concatenate([input_images_tags, input_images_tags2])


	
	dt.fit(a, input_images_tags)

	print "Training Successfully Completed!"


def predictImg(img):
	img = transform.resize(img, (100, 150))
	lbp = feature.local_binary_pattern(img, n_points, radius, METHOD)
	lbp = lbp.flatten()
	k = dt.predict(lbp)
	return tags[k[0]]
	