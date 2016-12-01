import numpy as np
import sys, time, glob, os, ntpath, cv2, numpy, cPickle
import featuresLBP
import featuresLBP2
import featuresColor
import featuresHOG
import featuresLines
import featuresCorners
import featuresFace
import featuresDOF
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
import sklearn.svm
import sklearn.decomposition
import sklearn.ensemble


def resizeFrame(frame, targetWidth):
    (Width, Height) = frame.shape[1], frame.shape[0]

    if targetWidth > 0:                             # Use FrameWidth = 0 for NO frame resizing
        ratio = float(Width) / targetWidth        
        newHeight = int(round(float(Height) / ratio))
        frameFinal = cv2.resize(frame, (targetWidth, newHeight))
    else:
        frameFinal = frame;

    return frameFinal

def blockshaped(arr):
    blocks = []    
    blocks.append(arr[0:arr.shape[0],0:arr.shape[1]/2])
    blocks.append(arr[0:arr.shape[0],arr.shape[1]/2:-1])
    #blocks.append(arr[arr.shape[0]/2:-1,0:arr.shape[1]/2])
    #blocks.append(arr[arr.shape[0]/2:-1,arr.shape[1]/2:-1])
    return blocks

def featureExtraction(img, PLOT = False):
    start = time.clock()
    
    
    #[fLines, namesLines] = featuresLines.getLineFeatures(img);     t1 = time.clock();    #print "{0:.2f} seconds used in line detection".format(t1-start)
    #[fFaces, namesFaces] = featuresFace.getFeaturesFace(img);    t2 = time.clock();    #print "{0:.2f} seconds used in Face features".format(t3-t2)
    #[fLBP, namesLBP] = featuresLBP2.getLBP(img);            t3 = time.clock();    #print "{0:.2f} seconds used in LBP".format(t4-t3)
    #[fColor, namesColor] = featuresColor.getRGBS(img, PLOT);    t4 = time.clock();    #print "{0:.2f} seconds used in Color features".format(t5-t4)
    #[fHOG, namesHOG] = featuresHOG.getHOG(img);            t5 = time.clock();    #print "{0:.2f} seconds used in HOG".format(t6-t5)
    #[fDOF, namesDOF] = featuresDOF.getDepthOfFieldFeature2(img);     t6 = time.clock();
    
    blocks = blockshaped(img)    
    [fLines1, namesLines] = featuresLines.getLineFeatures(blocks[0]); 
    [fLines2, namesLines] = featuresLines.getLineFeatures(blocks[1]); 
    #[fLines3, namesLines] = featuresLines.getLineFeatures(blocks[2]); 
    #[fLines4, namesLines] = featuresLines.getLineFeatures(blocks[3]); 
    t1 = time.clock();    #print "{0:.2f} seconds used in line detection".format(t1-start)    
    #[fFaces1, namesFaces] = featuresFace.getFeaturesFace(blocks[0]);    
    #[fFaces2, namesFaces] = featuresFace.getFeaturesFace(blocks[1]);    
    #[fFaces3, namesFaces] = featuresFace.getFeaturesFace(blocks[2]);    
    #[fFaces4, namesFaces] = featuresFace.getFeaturesFace(blocks[3]);        
    [fLBP1, namesLBP] = featuresLBP2.getLBP(blocks[0]);            
    [fLBP2, namesLBP] = featuresLBP2.getLBP(blocks[1]);            
    #[fLBP3, namesLBP] = featuresLBP2.getLBP(blocks[2]);            
    #[fLBP4, namesLBP] = featuresLBP2.getLBP(blocks[3]);                
    [fColor1, namesColor] = featuresColor.getRGBS(blocks[0], PLOT);    
    [fColor2, namesColor] = featuresColor.getRGBS(blocks[1], PLOT);    
    #[fColor3, namesColor] = featuresColor.getRGBS(blocks[2], PLOT);    
    #[fColor4, namesColor] = featuresColor.getRGBS(blocks[3], PLOT);        
    [fHOG1, namesHOG] = featuresHOG.getHOG(blocks[0]);            
    [fHOG2, namesHOG] = featuresHOG.getHOG(blocks[1]);            
    #[fHOG3, namesHOG] = featuresHOG.getHOG(blocks[2]);            
    #[fHOG4, namesHOG] = featuresHOG.getHOG(blocks[3]);      
    #[fDOF1, namesDOF] = featuresDOF.getDepthOfFieldFeature2(blocks[0]);
    #[fDOF2, namesDOF] = featuresDOF.getDepthOfFieldFeature2(blocks[1]);
    #[fDOF3, namesDOF] = featuresDOF.getDepthOfFieldFeature2(blocks[2]);
    #[fDOF4, namesDOF] = featuresDOF.getDepthOfFieldFeature2(blocks[3]);    
    t6 = time.clock();    

    #print "Total time: {0:.2f}".format(t6-start)

    # WITH BLOCKING
    #fv = fLines1 + fLines2 +fLines3 +fLines4 + fFaces1 + fFaces2 + fFaces3 +fFaces4 + fLBP1 + fLBP2 + fLBP3 + fLBP4 + fColor1 + fColor2 + fColor3 + fColor4 + fHOG1 + fHOG2 + fHOG3 + fHOG4 + fDOF1 + fDOF2 + fDOF3 + fDOF4;             
    #fNames = namesLines + namesLines + namesLines + namesLines +  namesFaces +  namesFaces +  namesFaces +  namesFaces + namesLBP + namesLBP + namesLBP + namesLBP + namesColor + namesColor + namesColor + namesColor + namesHOG + namesHOG + namesHOG + namesHOG + namesDOF+ namesDOF+ namesDOF+ namesDOF    
    #fv = fLines1 + fLines2 +fLines3 +fLines4 + fLBP1 + fLBP2 + fLBP3 + fLBP4 + fColor1 + fColor2 + fColor3 + fColor4 + fHOG1 + fHOG2 + fHOG3 + fHOG4
    #fNames = namesLines + namesLines + namesLines + namesLines +  namesLBP + namesLBP + namesLBP + namesLBP + namesColor + namesColor + namesColor + namesColor + namesHOG + namesHOG + namesHOG + namesHOG
    fv = fLines1 + fLines2 + fLBP1 + fLBP2 + fColor1 + fColor2  + fHOG1 + fHOG2
    fNames = namesLines + namesLines  +  namesLBP + namesLBP + namesColor + namesColor + namesHOG + namesHOG 

    # WITHOUT BLOCKING
    #fv     = fLines + fFaces + fLBP + fColor + fHOG + fDOF;             
    #fNames = namesLines + namesFaces + namesLBP + namesColor + namesHOG + namesDOF

    #fv     = fLines + fLBP + fColor + fHOG + fDOF;             
    #fNames = namesLines + namesLBP + namesColor + namesHOG + namesDOF        

    return fv, fNames

def getFeaturesFromFile(fileName, PLOT = False):
    img = cv2.imread(fileName, cv2.CV_LOAD_IMAGE_COLOR)    # read image
    #img2 = resizeFrame(img, 128)# resize

    #img2[:,:,0] = img2[:,:,0] + 3.5 * img2.std() * np.random.random([img2.shape[0], img2.shape[1]])
    #img2[:,:,1] = img2[:,:,1] + 3.5 * img2.std() * np.random.random([img2.shape[0], img2.shape[1]])
    #img2[:,:,2] = img2[:,:,2] + 3.5 * img2.std() * np.random.random([img2.shape[0], img2.shape[1]])    
    #plt.imshow(img2)
    #plt.show()    
    #[F, N] = featureExtraction(img2, PLOT)            # feature extraction
    [F, N] = featureExtraction(img, PLOT)            # feature extraction
    return F, N

def getFeaturesFromDir(dirName):
    types = ('*.jpg', '*.JPG', '*.png')    
    imageFilesList = []
    for files in types:
        imageFilesList.extend(glob.glob(os.path.join(dirName, files)))
    
    imageFilesList = sorted(imageFilesList)
    
    Features = []; 
    for i, imFile in enumerate(imageFilesList):    
        print "{0:.1f}".format(100.0 * float(i) / len(imageFilesList))
        [F, Names] = getFeaturesFromFile(imFile)
        Features.append(F)

    Features = np.matrix(Features)

    return (Features, imageFilesList, Names)

def getFeaturesFromDirs(dirNames):
    features = [];
    classNames = []
    fileNames = []
    for i,d in enumerate(dirNames):
        print " * * * * * * * * *"
        print d
        print " * * * * * * * * *"
        [f, fn, featureNames] = getFeaturesFromDir(d)
        if f.shape[0] > 0: # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])

    return features, classNames, fileNames, featureNames
    #return (Features, imageFilesList, Names)

def pcaDimRed(features, nDims):        
    pca = sklearn.decomposition.PCA(n_components = nDims)    
    pca.fit(features)
    coeff = pca.components_          
    featuresNew = []
    for f in features:
        ft = f.copy()                        
        ft = numpy.squeeze(numpy.asarray(numpy.dot(f, coeff.T)))        
        featuresNew.append(ft)
    print numpy.array(featuresNew).shape
    return (featuresNew, coeff)

def visualizeFeatures(Features, Files, Names):    
    y_eig, coeff = pcaDimRed(Features, 2)    
    plt.close("all")
    print y_eig
    plt.subplot(2,1,1);
    ax = plt.gca()
    for i in range(len(Files)):
        im = cv2.imread(Files[i], cv2.CV_LOAD_IMAGE_COLOR)    
        Width = 0.2;  Height = 0.2; startX = y_eig[i][0]; startY = y_eig[i][1];
        print startX, startY
        myaximage = ax.imshow(cv2.cvtColor(im, cv2.cv.CV_RGB2BGR), extent=(startX-Width/2.0, startX+Width/2.0, startY-Height/2.0, startY+Height/2.0), alpha=1.0, zorder=-1)
        plt.axis((-3,3,-3,3))
    # Plot feaures
    plt.subplot(2,1,2)    
    ax = plt.gca()
    for i in range(len(Files)):            
        plt.plot(numpy.array(Features[i,:].T));
    plt.xticks(range(len(Names)))
    plt.legend(Files)
    ax.set_xticklabels(Names)
    plt.setp(plt.xticks()[1], rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    
    plt.show()

def main(argv):
    if argv[1] == "-featuresFile":
        if len(argv)==3:
            [F, Names] = getFeaturesFromFile(argv[2], True)
            plt.plot(F)    
            ax = plt.gca()
            plt.xticks(range(len(Names)))
            ax.set_xticklabels(Names)
            plt.setp(plt.xticks()[1], rotation=90)
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.show()

    if argv[1] == "-featuresFilesCompare":
        if len(argv)==4:
            F1,_      = getFeaturesFromFile(argv[2], False)
            F2,Names = getFeaturesFromFile(argv[3], False)
            plt.clf()
            plt.plot(F1,'g');
            plt.plot(F2,'r');
            plt.legend([ntpath.basename(argv[2]), ntpath.basename(argv[3])])
            ax = plt.gca()
            plt.xticks(range(len(Names)))        
            ax.set_xticklabels(Names)
            plt.setp(plt.xticks()[1], rotation=90)
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=8)
            plt.show()

    
    elif argv[1] == "-featuresDir":
        if len(argv)==3:
            (FM, Files, FeatureNames) = getFeaturesFromDir(argv[2])
            visualizeFeatures(FM, Files, FeatureNames)

    elif argv[1] == "-featuresDirs":
        if len(argv)>3:
            outputfileName = argv[2]
            dirNames = argv[3:len(argv)]
            (features, classNames, fileNames, featureNames) = getFeaturesFromDirs(dirNames)
            fo = open(outputfileName + "_features", "wb")
            cPickle.dump(features, fo, protocol = cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(fileNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(featureNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
            fo.close()
            
            

if __name__ == '__main__':
    main(sys.argv)
