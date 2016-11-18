import numpy, cv2, pywt, glob, os
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided as ast

def block_view(A, block= (32, 32)):
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def getDepthOfFieldFeature(img):

	# ARGUMENT:	cv2 bgr image
	
	Grayscale = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)

	# get 2-level wavelets coefs (TODO: 3level???)
	coeffs = pywt.wavedec2(Grayscale, 'db1', level=2)
	cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

	# compute aggregated coefs
	C = numpy.abs(cH2) + numpy.abs(cV2) + numpy.abs(cD2)
	print C.shape, img.shape, cH2.shape, cA2.shape, cH1.shape
	# block - process (5 x 5 area)
	W = block_view(C, ( int(C.shape[0] / 5), int(C.shape[1] / 5)))

	# compute 3 features:
	S = 0
	for i in range(len(W)):
		for j in range(len(W[i])):
			S += W[i,j].sum()

	S1 = W[2,2].sum()
	S2 = W[2,2].sum() + W[1,2].sum() + W[2,1].sum() + W[2,3].sum() + W[3,2].sum()
	S3 = 0
	for i in range(1,4):
		for j in range(1,4):
			S3 += W[i,j].sum()

	S4 = 0
	for i in range(0,3):
		for j in range(0,5):
			S4 += W[i,j].sum()
	S5 = 0
	for i in range(2,5):
		for j in range(0,5):
			S5 += W[i,j].sum()
	S6 = 0
	for i in range(0,5):
		for j in range(0,3):
			S6 += W[i,j].sum()
	S7 = 0
	for i in range(0,5):
		for j in range(2,5):
			S7 += W[i,j].sum()

	F1 = S1 / S
	F2 = S2 / S
	F3 = S3 / S
	F4 = S4 / S
	F5 = S5 / S
	F6 = S6 / S
	F7 = S7 / S

	Fnames = ["DOF-central1", "DOF-central2", "DOF-central3", "DOF-upper", "DOF-lower", "DOF-left", "DOF-right"]
	features = [F1, F2, F3, F4, F5, F6, F7]

#	print features
#	plt.subplot(2,1,1); plt.imshow(Grayscale)
#	plt.subplot(2,1,2); plt.imshow(C)
#	plt.show()

	return features, Fnames

def getDepthOfFieldFeature2(img):
	# ARGUMENT:	cv2 bgr image
	
	Grayscale = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)

	C = cv2.Canny(Grayscale,100,200)
	# block - process (5 x 5 area)
	W = block_view(C, ( int(C.shape[0] / 5), int(C.shape[1] / 5)))

	# compute 3 features:
	S = 0
	for i in range(len(W)):
		for j in range(len(W[i])):
			S += W[i,j].sum()

	S1 = W[2,2].sum()
	S2 = W[2,2].sum() + W[1,2].sum() + W[2,1].sum() + W[2,3].sum() + W[3,2].sum()
	S3 = 0
	for i in range(1,4):
		for j in range(1,4):
			S3 += W[i,j].sum()

	S4 = 0
	for i in range(0,3):
		for j in range(0,5):
			S4 += W[i,j].sum()
	S5 = 0
	for i in range(2,5):
		for j in range(0,5):
			S5 += W[i,j].sum()
	S6 = 0
	for i in range(0,5):
		for j in range(0,3):
			S6 += W[i,j].sum()
	S7 = 0
	for i in range(0,5):
		for j in range(2,5):
			S7 += W[i,j].sum()

	F1 = S1 / (S + 0.00000001)
	F2 = S2 / (S + 0.00000001)
	F3 = S3 / (S + 0.00000001)
	F4 = S4 / (S + 0.00000001)
	F5 = S5 / (S + 0.00000001)
	F6 = S6 / (S + 0.00000001)
	F7 = S7 / (S + 0.00000001)

	Fnames = ["DOF-central1", "DOF-central2", "DOF-central3", "DOF-upper", "DOF-lower", "DOF-left", "DOF-right"]
	features = [F1, F2, F3, F4, F5, F6, F7]

#	print features
#	plt.subplot(2,1,1); plt.imshow(Grayscale)
#	plt.subplot(2,1,2); plt.imshow(C)
#	plt.show()

	return features, Fnames


def scriptDemo():
	dirName = "demoData/dofDemo"
	types = ('*.jpg', )
	imageFilesList = []
	for files in types:
		imageFilesList.extend(glob.glob(os.path.join(dirName, files)))
	
	imageFilesList = sorted(imageFilesList)
	print imageFilesList
	Features = numpy.zeros( (len(imageFilesList), 7) )
	for i, f in enumerate(imageFilesList):
		img = cv2.imread(f, cv2.CV_LOAD_IMAGE_COLOR)	# read image
		[F, names] = getDepthOfFieldFeature2(img)
		Features[i,:] = F
	# Features[i,j] contains the j-th feature of the i-th file
	for i in range(7):
		plt.subplot(7,1,i+1); plt.bar(range(Features[:,i].shape[0]), Features[:,i])
		plt.title(names[i])
	plt.show()
	print Features
#scriptDemo()
