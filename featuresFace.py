import cv2, time, sys, glob, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy


# face detection-related paths:
HAAR_CASCADE_PATH_FRONTAL = "haarcascade_frontalface_default.xml"
HAAR_CASCADE_PATH_PROFILE = "haarcascade_frontalface_default.xml"

def intersect_rectangles(r1, r2):
	x11 = r1[0]; y11 = r1[1]; x12 = r1[0]+r1[2]; y12 = r1[1]+r1[3];
	x21 = r2[0]; y21 = r2[1]; x22 = r2[0]+r2[2]; y22 = r2[1]+r2[3];
		
	X1 = max(x11, x21); X2 = min(x12, x22);
	Y1 = max(y11, y21); Y2 = min(y12, y22);

	W = X2 - X1
	H = Y2 - Y1
	if (H>0) and (W>0):
		E = W * H;
	else:
		E = 0.0;
	Eratio = 2.0*E / (r1[2]*r1[3] + r2[2]*r2[3])
	return Eratio


def initialize_face():
	cascadeFrontal = cv2.cv.Load(HAAR_CASCADE_PATH_FRONTAL);
	cascadeProfile = cv2.cv.Load(HAAR_CASCADE_PATH_PROFILE);
	storage = cv2.cv.CreateMemStorage()
	return (cascadeFrontal, cascadeProfile, storage)

def detect_faces(image, cascadeFrontal, cascadeProfile, storage):
	facesFrontal = []; facesProfile = []
	detectedFrontal = cv2.cv.HaarDetectObjects(image, cascadeFrontal, storage, 1.3, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (image.width/20,image.width/20))
	#detectedProfile = cv2.cv.HaarDetectObjects(image, cascadeProfile, storage, 1.3, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (newWidth/10,newWidth/10))
	if detectedFrontal:
		for (x,y,w,h),n in detectedFrontal:
			facesFrontal.append((x,y,w,h))
	#if detectedProfile:
	#	for (x,y,w,h),n in detectedProfile:
	#		facesProfile.append((x,y,w,h))

	# remove overlaps:
	while (1):
		Found = False
		for i in range(len(facesFrontal)):
			for j in range(len(facesFrontal)):
				if i != j:
					interRatio = intersect_rectangles(facesFrontal[i], facesFrontal[j])
					if interRatio>0.3:
						Found = True;
						del facesFrontal[i]
						break;
			if Found:
				break;

		if not Found:	# not a single overlap has been detected -> exit loop
			break;


	#return (facesFrontal, facesProfile)
	return (facesFrontal)


def getFeaturesFace(img):
	RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	(cascadeFrontal, cascadeProfile, storage) = initialize_face()
	facesFrontal = detect_faces(cv2.cv.fromarray(RGB), cascadeFrontal, cascadeProfile, storage)

	tempF = 0.0
	faceSizes = []
	for f in facesFrontal:
		faceSizes.append(f[2] * f[3] / float(img.shape[0] * img.shape[1]))

	F = []
	F.append(len(facesFrontal))
	if len(facesFrontal)>0:
		F.append(min(faceSizes))
		F.append(max(faceSizes))
		F.append(numpy.mean(faceSizes))
	else:
		F.extend([0, 0, 0]);

	Fnames = ["Faces-Total", "Faces-minSizePer", "Faces-maxSizePer", "Faces-meanSizePer"]
	return (F, Fnames)
	#print F
	#print tempF/len(facesFrontal)

