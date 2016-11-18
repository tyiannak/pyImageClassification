import cv2
import math, glob, os, ntpath
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from math import atan2, degrees, pi, sqrt
from mpl_toolkits.mplot3d.axes3d import Axes3D
#from shapely.geometry import LineString
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist

def temp():
	a = []
	for i in range(100):
		x1 = np.random.randn(1)
		x2 = np.random.randn(1)
		x3 = np.random.randn(1)
		x4 = np.random.randn(1)
		(A,_) = (angle(x1,x2,x3,x4))
		a.append(int(A))
	plt.plot(a)
	plt.show()


def perp( a ) :
	b = np.empty_like(a)
	b[0] = -a[1]
	b[1] = a[0]
	return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
	da = a2-a1
	db = b2-b1
	dp = a1-b1
	dap = perp(da)
	denom = np.dot( dap, db)
	num = np.dot( dap, dp )
	if denom==0:
		return np.NaN + b1
	else:
		return (num / denom)*db + b1

def angle(x1,x2,y1,y2):
	dx = x2 - x1
	dy = y2 - y1
	rads = atan2(-dy,dx)
	rads %= 2*pi
	degs = degrees(rads)
	dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
	return degs, dist

def getLineFeatures(img): # bgr image as argument:

	MIN_LENGTH = (int(np.floor(img.shape[0]/30)) + int(np.floor(img.shape[1]/30)))/2

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	lines = []
	for blurRatio in [40, 50, 60]:
		if blurRatio==0:
			blur = gray
		else:
			blurWindow = (int(np.floor(img.shape[0]/blurRatio)) + int(np.floor(img.shape[1]/blurRatio)))/2
			if blurWindow % 2 == 0:
				blurWindow += 1
			blur = cv2.bilateralFilter(gray, blurWindow, 100, 100)

		edges1 = cv2.Canny(blur,100,200)
		edges2 = cv2.Canny(blur,50,100)
		edges3 = cv2.Canny(blur,50,150)
		edges4 = cv2.Canny(blur,100,250)
		edges = (edges1+edges2+edges3+edges4)/4
	
		minLineLength = int( (img.shape[0] / 20 + img.shape[1]/20) / 2 ) 
		maxLineGap = int( (img.shape[0] / 30 + img.shape[1]/30) / 2 ) 

		for param in [50, 100]:			
			l = cv2.HoughLinesP(edges,1,np.pi/180,param, minLineLength, maxLineGap)
			if l==None:
				l = []
			else:
				l = l[0]
				if len(lines)==0:
					lines = l;
				else:
					for nl in l:
						D = dist.cdist(lines, np.matrix(nl))
						if D.min() > int( (img.shape[0] / 50 + img.shape[1]/50) / 2 ) :
							lines = np.append(lines, [nl], axis = 0 )

	Angles = []; AnglesTemp = []
	Distances = []; DistancesTemp = []
	finalLines = []
	#plt.subplot(2,1,1); 
	#plt.imshow(blur,cmap = cm.Greys_r);   plt.axis('off'); plt.autoscale()
	#plt.subplot(2,1,2); plt.imshow(edges,cmap = cm.Greys_r);  plt.axis('off'); plt.autoscale()


	for (x1,y1,x2,y2) in lines:
		a, d = angle(x1,x2,y1,y2)
		AnglesTemp.append(a)
		DistancesTemp.append(d)
	AnglesTemp = np.array(AnglesTemp)
	DistancesTemp = np.array(DistancesTemp)
	
	Angles = AnglesTemp[np.nonzero(DistancesTemp>MIN_LENGTH)]
	Distances = DistancesTemp[np.nonzero(DistancesTemp>MIN_LENGTH)]
	if len(lines)>0:
		finalLines = lines[np.nonzero(DistancesTemp>MIN_LENGTH)]

	if len(finalLines)>100:
		ThresholdDist = (np.sort(Distances)[::-1])[100]
		Angles =Angles[np.nonzero(Distances>ThresholdDist)]		
		finalLines = finalLines[np.nonzero(Distances>ThresholdDist)]
		Distances = Distances[np.nonzero(Distances>ThresholdDist)]
	
	#for (x1,y1,x2,y2) in finalLines:
	#	plt.plot([x1, x2], [y1, y2], color='r')

	# Histogram of orientations: 0 for horizontal, 1 for diagonal and 2 for vertical line orientations					(3 features)
	Angles = np.array(Angles)

	# acute Threshold:
	aThreshold = 40;
	if Angles.shape[0]==0:
		perAcute = -0.01;
	else:
		perAcute = (np.nonzero((Angles>0)*(Angles<aThreshold)+(Angles<360)*(Angles>360-aThreshold))[0]).shape[0]/float(Angles.shape[0])

	AngleBins = np.arange(0,360,45);
	AngleBinsOrientation = np.array([0, 1, 2, 1, 0, 1, 2, 1])
	OrientationHist = np.zeros((3,1))
	for a, d in zip(Angles, Distances):
		OrientationHist[AngleBinsOrientation[np.argmin(np.abs(a-AngleBins))]] += d	
	if OrientationHist.sum()>0:
		OrientationHist = OrientationHist / OrientationHist.sum()
	else:
		OrientationHist = - 0.01*np.ones((3,1))

	# Find intersections:
	Xs = []; Ys = []

	#plt.subplot(2,1,1)
	for i,(xa1,ya1,xa2, ya2) in enumerate(finalLines):
		for j,(xb1,yb1,xb2,yb2) in enumerate(finalLines):			
			if (i > j):	
				[X,Y] =  seg_intersect( np.array([float(xa1),float(ya1)]), np.array([float(xa2),float(ya2)]), np.array([float(xb1),float(yb1)]), np.array([float(xb2),float(yb2)]))
				Xs.append(X);
				Ys.append(Y);

	Colors = ["r","b","g","k","y"]
	countParallel = 0
	Xoriginal = Xs
	Yoriginal = Ys
	Xfinal = []; Yfinal = []
	for i in range(len(Xs)):		
		if (Xs[i] > -0.2*img.shape[1]) and (Xs[i] < 1.2*img.shape[1]) and (Ys[i] > -0.2*img.shape[0]) and (Ys[i] < 1.2*img.shape[0]):
			#plt.plot(Xs[i],Ys[i],'r*')
			Xfinal.append(Xs[i])
			Yfinal.append(Ys[i])
		else:
			countParallel += 1
	
	# percentage of parallel lines														(1 feature)
	if len(Xs):	
		countParallel /= float(len(Xs))
	else:
		countParallel = -0.01

	Xs = Xfinal
	Ys = Yfinal
	countImportantClusters = 0
	if len(Xs)>1:
		intersectPointMatrx = np.matrix([Xs, Ys]).T
		Dists = dist.pdist(intersectPointMatrx) / (np.sqrt(img.shape[0]**2+img.shape[1]**2))
		#cls, means, steps = mlpy.kmeans(intersectPointMatrx, k=3, plus=True)
		#STDs = []
		#Weights = []
		#for c in range(3):
		#	temp = intersectPointMatrx[cls==c, :]
		#	STDs.append(np.std(dist.pdist(temp)/ (np.sqrt(img.shape[0]**2+img.shape[1]**2))))			
		#	Weights.append(np.nonzero(np.array(cls)==c)[0].shape[0])		
		#Weights = np.array(Weights)
		#STDs = np.array(STDs)
		#countImportantClusters = STDs[np.argmax(Weights)]	
		per5  = float(np.nonzero(Dists<0.05)[0].shape[0]) / Dists.shape[0]
		per10 = float(np.nonzero(Dists<0.20)[0].shape[0]) / Dists.shape[0]
		#for i,(X,Y) in enumerate(zip(Xs, Ys)):
			#if (X>-0.2*img.shape[1]) and (X<1.2*img.shape[1]) and (Y<1.2*img.shape[0]) and (Y>-0.2*img.shape[0]):				
		#	plt.plot(X, Y,'*')
		#plt.subplot(3,1,3); plt.hist(Dists)

	else:
		if countParallel == -0.01:
			per5 = -0.01; per10 = -0.01
		else:
			per5 = 0.0; per10 = 0.0

		#print float(np.nonzero(Dists<0.05)[0].shape[0]) / Dists.shape[0]
		#linkageMatrix = hier.linkage(Dists)
		#cls = hier.fcluster(linkageMatrix, 5, 'maxclust')-1

	#plt.show()

	F = [];
	F.extend([countParallel])			# feature 1 	percentage of parallel lines
	F.extend(OrientationHist[:,0].tolist())		# features 2-4	line orientation normalized histogram (horizontal, diagonal and vertical respectivelly)
	F.extend([per5, per10])	# features 5-6	percentage of very close and close intersection points
	F.extend([perAcute])

	Fnames = ["Lines-Parallel", "Lines-Hor", "Lines-Diag", "Lines-Ver", "Lines-InterPointsClose1", "Lines-InterPointsClose2","Lines-PerAcute"]

	return (F, Fnames)

def scriptDemo():
	dirName = "demoData/lineDemo"
	types = ('*.jpg', )
	imageFilesList = []
	for files in types:
		imageFilesList.extend(glob.glob(os.path.join(dirName, files)))
	
	imageFilesList = sorted(imageFilesList)

	Features = np.zeros( (len(imageFilesList), 7) )
	labels = []
	for i, f in enumerate(imageFilesList):		
		print f 
		if ntpath.basename(f)[0:5]=="noPer":
			labels.append(0)
		else:
			labels.append(1)		
		img = cv2.imread(f, cv2.CV_LOAD_IMAGE_COLOR)	# read image
		[F, names] = getLineFeatures(img)
		Features[i,:] = F

	FeaturesPrespective = Features[:,4:7];

	fig = plt.figure()
	color = ["ro","gx"]
	labels = np.array(labels)
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(FeaturesPrespective[np.nonzero(labels==0)[0],0], FeaturesPrespective[np.nonzero(labels==0)[0],1], FeaturesPrespective[np.nonzero(labels==0)[0],2], color[0], label='Non Perspective')
	ax.plot(FeaturesPrespective[np.nonzero(labels==1)[0],0], FeaturesPrespective[np.nonzero(labels==1)[0],1], FeaturesPrespective[np.nonzero(labels==1)[0],2], color[1], label='Perspective')

	ax.set_xlabel('Close Intersections - 5%')
	ax.set_ylabel('Close Intersections - 20%')
	ax.set_zlabel('Acute Angles')

	plt.legend(loc='upper left', numpoints=1)

	plt.show()

	for f in range(Features.shape[0]):
		print "{0:.4f}\t{1:.4f}\t{2:.4f}".format(Features[f, 4], Features[f, 5], Features[f, 6])


# For generating figures for paper
#scriptDemo()

# test intersection:
#A1 = np.array([0.0,0.0])
#A2 = np.array([4.0,2.0])
#B1 = np.array([1.0,1.2])
#B2 = np.array([2.0,1.0])
#plt.plot([A1[0], A2[0]], [A1[1],A2[1]])
#plt.plot([B1[0], B2[0]], [B1[1],B2[1]])
#[X, Y] = seg_intersect(A1, A2, B1, B2)
#print X, Y
#plt.plot(X, Y, '*');
#plt.show()


