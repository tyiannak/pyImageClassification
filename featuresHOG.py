#!/usr/bin/env python

from itertools import product
from math import floor, pi
import numpy as np
import cv2  # opencv 2
from SimpleCV import *
import skimage

# TODO: should we do this using scikit-image (http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=peak_local_max#skimage.feature.local_binary_pattern)

def normalize(v):
	norm=np.linalg.norm(v)
	if norm==0: 
	   return v
	return v/norm

def findHOGFeatures(img, n_divs=5, n_bins=4):
	"""
	**SUMMARY**
	Get HOG(Histogram of Oriented Gradients) features from the image.


	**PARAMETERS**
	* *img*	- SimpleCV.Image instance
	* *n_divs* - the number of divisions(cells).
	* *n_bins* - the number of orientation bins.

	**RETURNS**
	Returns the HOG vector in a numpy array

	"""
	# Size of HOG vector
	n_HOG = n_divs * n_divs * n_bins

	# Initialize output HOG vector
	# HOG = [0.0]*n_HOG
	HOG = np.zeros((n_HOG, 1))
	# Apply sobel on image to find x and y orientations of the image
	Icv = img.getNumpyCv2()
	Ix = cv2.Sobel(Icv, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
	Iy = cv2.Sobel(Icv, ddepth=cv.CV_32F, dx=0, dy=1, ksize=3)

	Ix = Ix.transpose(1, 0, 2)
	Iy = Iy.transpose(1, 0, 2)
	cellx = img.width / n_divs  # width of each cell(division)
	celly = img.height / n_divs  # height of each cell(division)

	#Area of image
	img_area = img.height * img.width

	#Range of each bin
	BIN_RANGE = (2 * pi) / n_bins

	# m = 0
	angles = np.arctan2(Iy, Ix)
	magnit = ((Ix ** 2) + (Iy ** 2)) ** 0.5
	it = product(xrange(n_divs), xrange(n_divs), xrange(cellx), xrange(celly))

	for m, n, i, j in it:
		# grad value
		grad = magnit[m * cellx + i, n * celly + j][0]
		# normalized grad value
		norm_grad = grad / img_area
		# Orientation Angle
		angle = angles[m*cellx + i, n*celly+j][0]
		# (-pi,pi) to (0, 2*pi)
		if angle < 0:
			angle += 2 * pi
		nth_bin = floor(float(angle/BIN_RANGE))
		HOG[((m * n_divs + n) * n_bins + int(nth_bin))] += norm_grad

	return HOG[:,0].tolist()

def getHOG(img):
	image = Image(img, cv2image=True)		# convert it to SimpleCV image
	HOG = findHOGFeatures(image, 2, 4)
	Fnames = ["HOG"+str(i).zfill(2) for i in range(len(HOG))]
	HOG = normalize(HOG)
	if type(HOG) is list:
		return HOG, Fnames
	else:
		return HOG.tolist(), Fnames


