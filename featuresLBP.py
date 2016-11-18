#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def normalize(v):
	norm=np.linalg.norm(v)
	if norm==0: 
	   return v
	return v/norm

def bilinear_interpolation(x, y, img):
	x1, y1 = int(x), int(y)
	x2, y2 = math.ceil(x), math.ceil(y)

	r1 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y1)
	r2 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y2)

	return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2	

def thresholded(center, pixels):
	out = []
	for a in pixels:
		if a >= center:
			out.append(1)
		else:
			out.append(0)
	return out

def get_pixel_else_0(image, idx, idy):
	if idx < int(len(image)) - 1 and idy < len(image[0]):
		return image[idx,idy]
	else:
		return 0

def find_variations(pixel_values):
	prev = pixel_values[-1]
	t = 0
	for p in range(0, len(pixel_values)):
		cur = pixel_values[p]
		if cur != prev:
			t += 1
		prev = cur
	return t


def getLBP(img):
	#img = cv2.imread('../images/mug.jpeg', 0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	transformed_img = img.copy()
	

	P = 8 # number of pixels
	R = 1 # radius 

	for x in range(0, len(img)):
		for y in range(0, len(img[0])):
			center		= img[x,y]
			pixels = []
			for point in range(0, P):
				r = x + R * math.cos(2 * math.pi * point / P)
				c = y - R * math.sin(2 * math.pi * point / P)
				if r < 0 or c < 0:
					pixels.append(0)
					continue
				if int(r) == r:
					if int(c) != c:
						c1 = int(c)
						c2 = math.ceil(c)
						w1 = (c2 - c) / (c2 - c1)
						w2 = (c - c1) / (c2 - c1)
										
						pixels.append(int((w1 * get_pixel_else_0(img, int(r), int(c)) + \
									   w2 * get_pixel_else_0(img, int(r), math.ceil(c))) / (w1 + w2)))
					else:
						pixels.append(get_pixel_else_0(img, int(r), int(c)))
				elif int(c) == c:
					r1 = int(r)
					r2 = math.ceil(r)
					w1 = (r2 - r) / (r2 - r1)
					w2 = (r - r1) / (r2 - r1)				
					pixels.append((w1 * get_pixel_else_0(img, int(r), int(c)) + \
								   w2 * get_pixel_else_0(img, math.ceil(r), int(c))) / (w1 + w2))
				else:
					pixels.append(bilinear_interpolation(r, c, img))				
			values = thresholded(center, pixels)
			res = 0
			for a in range(0, len(values)):
				res += values[a] * (2 ** a)
			transformed_img.itemset((x,y), res)

	hist,bins = np.histogram(img.flatten(),16,[0,256])

	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	print transformed_img
	LBP = plt.hist(transformed_img.flatten(),16,[0,256], color = 'r')

	return normalize(LBP[0])

