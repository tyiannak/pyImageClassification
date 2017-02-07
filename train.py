import numpy as np
import sys, time, glob, os, ntpath, cv2,  numpy, cPickle
import featureExtraction
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn import mixture
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
#from nolearn.dbn import DBN
from scipy import linalg as la
#from sklearn.lda import LDA
import random
RBM_NCOMPONENTS = 120;
import sklearn.svm


class kNN:
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = int(k)
    def classify(self, testSample):
        nClasses = numpy.unique(self.Y).shape[0]
        if testSample.shape[0] != 1:
            YDist =  (distance.cdist(self.X, testSample.reshape(1, testSample.shape[0]), 'euclidean')).T
        else:
            YDist =  (distance.cdist(self.X, testSample, 'euclidean')).T
        iSort = numpy.argsort(YDist)
        P = numpy.zeros((nClasses,))
        for i in range(nClasses):
            P[i] = numpy.nonzero(self.Y[iSort[0][0:self.k]]==i)[0].shape[0] / float(self.k)
        return (numpy.argmax(P), P)



def classifierWrapper(classifier, classifierType, testSample):
    '''
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - classifier:        a classifier object of type mlpy.LibSvm or kNN (defined in this library)
        - classifierType:    "svm" or "knn"
        - testSample:        a feature vector (numpy array)
    RETURNS:
        - R:            class ID
        - P:            probability estimate
    '''
    R = -1; P = -1
    if classifierType == "knn":
        [R, P] = classifier.classify(testSample)
    elif classifierType == "svm":        
        R = classifier.predict(testSample.reshape(1,-1))[0]
        P = classifier.predict_proba(testSample.reshape(1,-1))[0]
    elif classifierType == "gmm":
        P = []
        array = numpy.matrix(testSample)
        for i, gmm in enumerate(classifier):        
            P.append(gmm.score(array))
        R = numpy.argmax(numpy.array(P))
        P = numpy.array(P)
        P[P<0] = 0.0;
        if P.sum()>0:        
            P = P / P.sum()
        P = P[:, 0]
    elif classifierType == "gmmNoNormalize":
        P = []
        array = numpy.matrix(testSample)
        for i, gmm in enumerate(classifier):        
            P.append(gmm.score(array))
        R = numpy.argmax(numpy.array(P))
        P = numpy.array(P)
        P[P<0] = 0.0;
        P = P[:, 0]
    elif classifierType == "rbm":
        R = classifier.predict(testSample)
        P = 1
    elif classifierType == "rbm+svm":
        testSample2 = classifier["rbm"].transform(testSample)
        R = classifier["svm"].predict(testSample2.reshape(1,-1))[0]
        P = classifier["svm"].predict_proba(testSample2.reshape(1,-1))[0]
    #elif classifierType == "dbn":
    #    A =numpy.matrix(testSample)
    #    R = classifier.predict(A)        
    return [R, P]

def randSplitFeatures(features, textSents, ClassNames, partTrain):
    """
    def randSplitFeatures(features):
    
    This function splits a feature set for training and testing.
    
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                    each matrix features[i] of class i is [numOfSamples x numOfDimensions] 
        - partTrain:        percentage 
    RETURNS:
        - featuresTrains:    a list of training data for each class
        - featuresTest:        a list of testing data for each class
    """

    featuresTrain = []
    featuresTest = []
    if ClassNames[0] == "negative":
        features = [features[1], features[0]]

    sentimentPTrain = []; sentimentPTest = []
    for i,f in enumerate(features):
        [numOfSamples, numOfDims] = f.shape
        randperm = numpy.random.permutation(range(numOfSamples))
        nTrainSamples = int(round(partTrain * numOfSamples));
        featuresTrain.append(f[randperm[1:nTrainSamples]])
        featuresTest.append(f[randperm[nTrainSamples+1::]])
        if len(textSents)>i:
            sentimentPTrain.append(textSents[i][randperm[1:nTrainSamples]])
            sentimentPTest.append(textSents[i][randperm[nTrainSamples+1::]])
        
    return (featuresTrain, featuresTest, sentimentPTrain, sentimentPTest)

def trainGMM(features, nComp, covType = 'diag'):
    Gs = []
    for f in features:
        gmm = mixture.GMM(n_components=int(nComp), covariance_type = covType)
        Gs.append(gmm.fit(f))

    return Gs

def trainKNN(features, K):
    '''
    Train a kNN  classifier.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                    each matrix features[i] of class i is [numOfSamples x numOfDimensions] 
        - K:            parameter K 
    RETURNS:
        - kNN:            the trained kNN variable

    '''
    [Xt, Yt] = listOfFeatures2Matrix(features)
    knn = kNN(Xt, Yt, K)
    return knn

def trainSVM(features, Cparam):
    '''
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the mlpy-LibSVM functionality for SVM training
        See function trainSVM_feature() to use a wrapper on both the feature extraction and the SVM training (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                    each matrix features[i] of class i is [numOfSamples x numOfDimensions] 
        - Cparam:        SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:            the trained SVM variable

    NOTE:    
        This function trains a linear-kernel SVM for a given C value. For a different kernel, other types of parameters should be provided.
        For example, gamma for a polynomial, rbf or sigmoid kernel. Furthermore, Nu should be provided for a nu_SVM classifier.
        See MLPY documentation for more details (http://mlpy.sourceforge.net/docs/3.4/svm.html)
    '''
    [X, Y] = listOfFeatures2Matrix(features)    
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)        
    svm.fit(X,Y)

    return svm

'''
def trainDBN(features, nComponents):

    [X, Y] = listOfFeatures2Matrix(features)
    # TODO !!!!!!!!!!!!!!!!!!!!!!!
    X = numpy.concatenate([X, X])
    Y = numpy.concatenate([Y,Y])
    dbn = DBN(
        [X.shape[1], int(nComponents), int(nComponents), len(features)],
        learn_rates = 0.2,
#        learn_rates_pretrain=0.005,
        learn_rate_decays = 0.9,
        epochs = 200,
        verbose = 0)

    dbn.fit(X, Y)
    return dbn
'''
def trainRBM_LR(features, paramC, nComponents):
    [X, Y] = listOfFeatures2Matrix(features)
    logistic = LogisticRegression(C = paramC)
    rbm = BernoulliRBM(n_components = nComponents, n_iter = 100, learning_rate = 0.01,  verbose = False)    
    # NORMALIZATION???
    classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
    classifier.fit(X, Y)
    return classifier

def trainRBM_SVM(features, Cparam, nComponents):
    [X, Y] = listOfFeatures2Matrix(features)
    rbm = BernoulliRBM(n_components = nComponents, n_iter = 30, learning_rate = 0.2,  verbose = True)
    rbm.fit(X,Y)
    newX = rbm.transform(X)
#    colors = ["r","g","b"]
#    for i in range(1,Y.shape[0],5):
#        plt.plot(newX[i,:], colors[int(Y[i])])
#    plt.show()

    classifier = {}
    classifier["rbm"] = rbm    
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)        
    svm.fit(newX,Y)

    classifier["svm"] = svm

    return classifier    


def featureAndTrain(listOfDirs, classifierType, modelName):
    '''
    This function is used as a wrapper to segment-based audio feature extraction and classifier training.
    ARGUMENTS:
        listOfDirs:        list of paths of directories. Each directory contains a signle audio class whose samples are stored in seperate WAV files.
        modelName:        name of the model to be saved
    RETURNS: 
        None. Resulting classifier along with the respective model parameters are saved on files.
    '''
    # STEP A: Feature Extraction:
    [features, classNames, _, featureNames] = featureExtraction.getFeaturesFromDirs(listOfDirs)

    # write to arff:
    writeTrainDataToARFF(modelName, features, classNames, featureNames)

    if len(features)==0:
        print "trainSVM_feature ERROR: No data found in any input folder!"
        return
    for i,f in enumerate(features):
        if len(f)==0:
            print "trainSVM_feature ERROR: " + listOfDirs[i] + " folder is empty or non-existing!"
            return

    # STEP B: Classifier Evaluation and Parameter Selection:
    if classifierType == "svm":
        classifierParams = numpy.array([0.001, 0.01,  0.25, 0.5, 1.0, 2.0, 5.0])
    elif classifierType == "knn":
        classifierParams = numpy.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]); 
    elif classifierType == "gmm":
        classifierParams = numpy.array([2, 3, 4, 5]); 
    elif classifierType == "rbm":
        classifierParams = numpy.array([1500, 2000, 2500, 3000, 3500, 4000, 10000]); 
    elif classifierType == "rbm+svm":
        classifierParams = numpy.array([0.001, 0.01,  0.25, 0.5, 1.0, 2.0, 5.0]); 
    elif classifierType == "dbn":
        classifierParams = numpy.array([10, 50, 100, 150, 160, 170, 180, 200, 500, 1000]); 

    # get optimal classifeir parameter:
    bestParam = evaluateClassifier(features, [],         classNames, 100, classifierType, classifierParams, 0, 0, "")
    print "Selected params: {0:.5f}".format(bestParam)

    C = len(classNames)
    [featuresNorm, MAX, MIN] = normalizeFeatures(features)        # normalize features
    MAX = MAX.tolist(); MIN = MIN.tolist()
    featuresNew = featuresNorm 
    

    # STEP C: Save the classifier to file
    if classifierType == "svm":
        Classifier = trainSVM(featuresNew, bestParam)
        #Classifier.save_model(modelName)
        with open(modelName, 'wb') as fid:
            cPickle.dump(Classifier, fid)            

        fo = open(modelName + "NORMALIZATION", "wb")
        cPickle.dump(MAX, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MIN,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "knn":
        [X, Y] = listOfFeatures2Matrix(featuresNew)
        X = X.tolist(); Y = Y.tolist()
        fo = open(modelName, "wb")
        cPickle.dump(X, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Y,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MAX, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MIN,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "gmm":
        Classifier = trainGMM(features, bestParam, covType = 'diag')        
        fo = open(modelName,  "wb")
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Classifier, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "rbm":
        Classifier = trainRBM_LR(features, bestParam, RBM_NCOMPONENTS)
        fo = open(modelName, "wb")
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Classifier, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MAX, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MIN,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "rbm+svm":
        Classifier = trainRBM_SVM(features, bestParam, RBM_NCOMPONENTS)
        fo = open(modelName, "wb")
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Classifier, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MAX, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MIN,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    '''    
    elif classifierType == "dbn":
        Classifier = trainDBN(features, bestParam)
        fo = open(modelName, "wb")
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Classifier, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MAX, fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MIN,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
        fo.close()
    '''

def loadGMMModel(gmmModel):
    try:
        fo = open(gmmModel, "rb")
    except IOError:
            print "didn't find file"
            return
    try:
        classNames = cPickle.load(fo)
        Classifier = cPickle.load(fo)
    except:
        fo.close()
    fo.close()    

    return(Classifier, classNames);


def loadKNNModel(kNNModelName):
    try:
        fo = open(kNNModelName, "rb")
    except IOError:
            print "didn't find file"
            return
    try:
        X     = cPickle.load(fo)
        Y     = cPickle.load(fo)
        MAX  = cPickle.load(fo)
        MIN   = cPickle.load(fo)
        classNames =  cPickle.load(fo)
        K     = cPickle.load(fo)
    except:
        fo.close()
    fo.close()    

    X = numpy.array(X)
    Y = numpy.array(Y)
    MAX = numpy.array(MAX)
    MIN = numpy.array(MIN)

    Classifier = kNN(X, Y, K) # Note: a direct call to the kNN constructor is used here, since the 

    return(Classifier, MAX, MIN, classNames);

def loadSVModel(SVMmodelName):
    try:
        fo = open(SVMmodelName+"NORMALIZATION", "rb")
    except IOError:
            print "Load SVM Model: Didn't find file"
            return
    try:
        MAX     = cPickle.load(fo)
        MIN      = cPickle.load(fo)
        classNames =  cPickle.load(fo)
    except:
        fo.close()
    fo.close()    

    MAX = numpy.array(MAX)
    MIN  = numpy.array(MIN)

    COEFF = []    

    with open(SVMmodelName, 'rb') as fid:
        SVM = cPickle.load(fid)    


    return(SVM, MAX, MIN, classNames);                

def loadDBNModel(dbnModelName):
    try:
        fo = open(dbnModelName, "rb")
    except IOError:
            print "didn't find file"
            return
    try:
        classNames     = cPickle.load(fo)
        Classifier     = cPickle.load(fo)
        MAX  = cPickle.load(fo)
        MIN   = cPickle.load(fo)
        classNames =  cPickle.load(fo)
        bestParam     = cPickle.load(fo)
    except:
        fo.close()
    fo.close()    

    return(Classifier, classNames, MAX, MIN);

def evaluateClassifier(features, fileNames, ClassNames, nExp, ClassifierName, Params, parameterMode, method, dimReduction = ""):
    '''
    ARGUMENTS:
        features:     a list ([numOfClasses x 1]) whose elements containt numpy matrices of features.
                each matrix features[i] of class i is [numOfSamples x numOfDimensions] 
        fileNames:
        ClassNames:    
        nExp:        number of experiments
        ClassifierName:    "svm" or "knn"
        Params:        list of classifier params value. For the case of svm+knn combination two values are needed per element
    RETURNS:
         bestParam:    the value of the input parameter that optimizes the selected performance measure        
    '''
    
    textSents = []
    for c in fileNames:
        t = []
        for f in c:
            t.append(text.classifyFile(1, "NBmodelSmall", f))
        textSents.append(numpy.array(t))

    # feature normalization:
    (featuresNorm, MAX, MIN) = normalizeFeatures(features)
    nClasses = len(features)
    CAll = []; acAll = []; F1All = []    
    PrecisionClassesAll = []; RecallClassesAll = []; ClassesAll = []; F1ClassesAll = []
    CMsAll = []

    for Ci, C in enumerate(Params):                # for each Nu value        

                CM = numpy.zeros((nClasses, nClasses))
                for e in range(nExp):        # for each cross-validation iteration:
                    # split features:
                    if (ClassifierName=="svm") or (ClassifierName=="knn") or (ClassifierName=="rbm") or (ClassifierName=="rbm+svm") or (ClassifierName=="dbn") or (ClassifierName=="svm+knn"):
                        featuresTrain, featuresTest, sentPTrain, sentPTest = randSplitFeatures(featuresNorm, textSents, ClassNames, 0.80)
                    elif ClassifierName=="gmm":
                        featuresTrain, featuresTest, sentPTrain, sentPTest = randSplitFeatures(features, textSents, ClassNames, 0.80)
                    # train multi-class svms:
                    if ClassifierName=="svm":
                        Classifier = trainSVM(featuresTrain, C)
                    elif ClassifierName=="knn":
                        Classifier = trainKNN(featuresTrain, C)
                    elif ClassifierName=="gmm":
                        Classifier = trainGMM(featuresTrain, C)
                    elif ClassifierName=="rbm":
                        Classifier = trainRBM_LR(featuresTrain, C, RBM_NCOMPONENTS)
                    elif ClassifierName=="rbm+svm":
                        Classifier = trainRBM_SVM(featuresTrain, C, RBM_NCOMPONENTS)
                    elif ClassifierName=="dbn":
                        Classifier = trainDBN(featuresTrain, C)
                    elif ClassifierName=="svm+knn":
                        Classifier1 = trainSVM(featuresTrain, C[0])
                        Classifier2 = trainKNN(featuresTrain, C[1])

                    CMt = numpy.zeros((nClasses, nClasses))
                    for c1 in range(nClasses):
                        #Results = Classifier.pred(featuresTest[c1])
                        nTestSamples = len(featuresTest[c1])
                        Results = numpy.zeros((nTestSamples,1))
                        for ss in range(nTestSamples):
                            if ClassifierName == "svm+knn":
                                [Rtemp, P1] = classifierWrapper(Classifier1, "svm", featuresTest[c1][ss])
                                [Rtemp, P2] = classifierWrapper(Classifier2, "knn", featuresTest[c1][ss])
                                P = [(P1[pp]+P2[pp])/2.0 for pp in range(len(P1))]
                                Results[ss] = numpy.argmax(numpy.array(P))
                            else:
                                [Results[ss], P] = classifierWrapper(Classifier, ClassifierName, featuresTest[c1][ss])
                        for c2 in range(nClasses):
                            CMt[c1][c2] = float(len(numpy.nonzero(Results==c2)[0]))
                    CM = CM + CMt
                CM = CM + 0.0000000010
                Rec = numpy.zeros((CM.shape[0],))
                Pre = numpy.zeros((CM.shape[0],))

                for ci in range(CM.shape[0]):
                    Rec[ci] = CM[ci,ci] / numpy.sum(CM[ci,:]);
                    Pre[ci] = CM[ci,ci] / numpy.sum(CM[:,ci]);
                PrecisionClassesAll.append(Pre)
                RecallClassesAll.append(Rec)
                F1 = 2 * Rec * Pre / (Rec + Pre)
                F1ClassesAll.append(F1)
                acAll.append(numpy.sum(numpy.diagonal(CM)) / numpy.sum(CM))

                CMsAll.append(CM)
                F1All.append(numpy.mean(F1))                

    print ("\t\t"),
    for i,c in enumerate(ClassNames): 
        if i==len(ClassNames)-1: print "{0:s}\t\t".format(c),  
        else: print "{0:s}\t\t\t".format(c), 
    print ("OVERALL")
    print ("\tC"),
    if ClassifierName == "svm+knn":
        print "\t",
    for c in ClassNames: print "\tPRE\tREC\tF1", 
    print "\t{0:s}\t{1:s}".format("ACC","F1")
    bestAcInd = numpy.argmax(acAll)
    bestF1Ind = numpy.argmax(F1All)
    for i in range(len(PrecisionClassesAll)):
        if ClassifierName == "svm+knn":
            print "\t{0:.3f} + {1:.3f}".format(Params[i][0], Params[i][1]),
        else:
            print "\t{0:.3f}".format(Params[i]),
        for c in range(len(PrecisionClassesAll[i])):
            print "\t{0:.1f}\t{1:.1f}\t{2:.1f}".format(100.0*PrecisionClassesAll[i][c], 100.0*RecallClassesAll[i][c], 100.0*F1ClassesAll[i][c]), 
        print "\t{0:.1f}\t{1:.1f}".format(100.0*acAll[i], 100.0*F1All[i]),
        if i == bestF1Ind: print "\t best F1", 
        if i == bestAcInd: print "\t best Acc",
        print

    if parameterMode==0:    # keep parameters that maximize overall classification accuracy:
        print "Confusion Matrix:"
        printConfusionMatrix(CMsAll[bestAcInd], ClassNames)
        return Params[bestAcInd]
    elif parameterMode==1:  # keep parameters that maximize overall F1 measure:
        print "Confusion Matrix:"
        printConfusionMatrix(CMsAll[bestF1Ind], ClassNames)
        return Params[bestF1Ind]

def printConfusionMatrix(CM, ClassNames):
    '''
    This function prints a confusion matrix for a particular classification task.
    ARGUMENTS:
        CM:        a 2-D numpy array of the confusion matrix 
                (CM[i,j] is the number of times a sample from class i was classified in class j)
        ClassNames:    a list that contains the names of the classes
    '''

    if CM.shape[0] != len(ClassNames):
        print "printConfusionMatrix: Wrong argument sizes\n"
        return

    for c in ClassNames: 
        if len(c)>4: 
            c = c[0:3]
        print "\t{0:s}".format(c),
    print

    for i, c in enumerate(ClassNames):
        if len(c)>4: c = c[0:3]
        print "{0:s}".format(c),
        for j in range(len(ClassNames)):
            print "\t{0:.1f}".format(100.0*CM[i][j] / numpy.sum(CM)),
        print

def normalizeFeatures(features):
    '''
    This function nromalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a numpy matrix)
    RETURNS:
        - featuresNorm:    list of NORMALIZED feature matrices
        - MAX:        mean vector
        - MIN:        std vector
    '''
    X = numpy.array([])
    
    for count,f in enumerate(features):
        if f.shape[0]>0:
            if count==0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1
    
#    MAX = numpy.mean(X, axis = 0)
#    MIN  = numpy.std( X, axis = 0)

#    featuresNorm = []
#    for f in features:
#        ft = f.copy()
#        for nSamples in range(f.shape[0]):
#            ft[nSamples,:] = (ft[nSamples,:] - MAX) / MIN    
#        featuresNorm.append(ft)

    MIN  = numpy.min(X, axis = 0)
    MAX  = numpy.max(X, axis = 0)

    featuresNorm = []
    for f in features:
        ft = f.copy()
        for nSamples in range(f.shape[0]):
            ft[nSamples,:] = (ft[nSamples,:] - MIN) / (MAX    - MIN + 0.000001)
        featuresNorm.append(ft)


    return (featuresNorm, MAX, MIN)

def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)
    
    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces    
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i,f in enumerate(features):
        if i==0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)

def listOfFeatures2MatrixRegression(features, Ys):
    '''
    listOfFeatures2MatrixRegression(features)
    
    Same as MatrixRegression but used for regression (also takes real values as arguments)

    ARGUMENTS:
        - features:        a list of feature matrices
        - Ys:            a list of respective real values

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces    
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i,f in enumerate(features):
        if i==0:
            X = f
            Y = Ys[i] * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, Ys[i] * numpy.ones((len(f), 1)))
    return (X, Y)
    

def fileClassification(inputFile, modelName, modelType):
    # Load classifier:
    if modelType!="svm+knn":
        if not os.path.isfile(modelName):
            print "fileClassification: input modelName not found!"
            return (-1,-1, -1)

    if not os.path.isfile(inputFile):
        print "fileClassification: input modelType not found!"
        return (-1,-1, -1)


    if modelType=='svm':
        [Classifier, MAX, MIN, classNames] = loadSVModel(modelName)
    elif modelType=='knn':
        [Classifier, MAX, MIN, classNames] = loadKNNModel(modelName)
    elif modelType=="gmm" or modelType=="gmmNoNormalize":
        [Classifier, classNames] = loadGMMModel(modelName)
    elif modelType=='dbn':
        [Classifier, classNames, MAX, MIN] = loadDBNModel(modelName)
    elif modelType=='svm+knn':
        [Classifier1, MAX, MIN, classNames] = loadSVModel(modelName[0])
        [Classifier2, MAX, MIN, classNames] = loadKNNModel(modelName[1])

    [curFV, _] = featureExtraction.getFeaturesFromFile(inputFile, False)    

    if modelType=='svm' or modelType=='knn' or modelType=="rbm" or modelType=="svm+knn":
        curFV = (curFV - MIN) / (MAX-MIN+0.00000001);            # normalization    
    if modelType!='svm+knn':
        [Result, P] = classifierWrapper(Classifier, modelType, curFV)    # classification
    else:
        [R1, P1] = classifierWrapper(Classifier1, "svm", curFV)
        [Re, P2] = classifierWrapper(Classifier2, "knn", curFV)
        P = (P1 + P2) / 2.0
        print P1
        print P2
        print P
        Result = numpy.argmax(numpy.array(P))

    # check if text file exists:
    inputFileText = inputFile[0:-4] + ".txt"
    print inputFileText
    if os.path.isfile("text" + os.sep + inputFileText):
        Ptext = text.classifyFile(1, "NBmodelSmall", inputFileText)
        print Ptext
        P = (1.3 * P + 0.7 * numpy.array([Ptext, 1-Ptext])) / 2.0        
    return P, classNames

def dirClassificationSentiment(dirName, modelName, modelType):
    types = ('*.jpg', '*.png',)
    filesList = []
    for files in types:
        filesList.extend(glob.glob(os.path.join(dirName, files)))

    filesList = sorted(filesList)
    print filesList
    Features = []
    plt.close('all');    
    ax = plt.gca()
    plt.hold(True)
    for fi in filesList:
        P, classNames = fileClassification(fi, modelName, modelType)
        im = cv2.imread(fi, cv2.CV_LOAD_IMAGE_COLOR)    
        Width = 0.1;  Height = 0.1; startX = P[classNames.index("positive")]; startY = 0;
        myaximage = ax.imshow(cv2.cvtColor(im, cv2.cv.CV_RGB2BGR), extent=(startX-Width/2.0, startX+Width/2.0, startY-Height/2.0, startY+Height/2.0), alpha=1.0, zorder=-1)
        plt.axis((0,1,-0.1,0.1))
        plt.show(block = False);
        plt.draw()
    plt.show(block = True);

def writeTrainDataToARFF(modelName, features, classNames, featureNames):
    f = open(modelName + ".arff", 'w')
    f.write('@RELATION ' + modelName + '\n');
    for fn in featureNames:
        f.write('@ATTRIBUTE ' + fn + ' NUMERIC\n')
    f.write('@ATTRIBUTE class {')
    for c in range(len(classNames)-1):
        f.write(classNames[c] + ',')
    f.write(classNames[-1] + '}\n\n')
    f.write('@DATA\n')
    for c, fe in enumerate(features):
        for i in range(fe.shape[0]):
            for j in range(fe.shape[1]):
                f.write("{0:f},".format(fe[i,j]))
            f.write(classNames[c]+"\n")
    f.close()

def temp(features):
    [featuresNorm, MAX, MIN] = normalizeFeatures(features)
    [X, Y] = listOfFeatures2Matrix(featuresNorm)
    rbm = BernoulliRBM(n_components = 10, n_iter = 1000, learning_rate = 0.01,  verbose = False)
    X1 = X[0::2]
    X2 = X[1::2]
    Y1 = Y[0::2]
    Y2 = Y[1::2]    
    rbm.fit(X1,Y1)
    YY = rbm.transform(X1)

    for i in range(10):plt.plot(YY[i,:],'r')
    for i in range(10):plt.plot(YY[i+10,:],'g')
    for i in range(10):plt.plot(YY[i+20,:],'b')
    plt.show()

def main(argv):
    if argv[1] == "-train":
        if len(argv)>=5:
            modelType = argv[2]
            modelName = argv[3]
            dirNames = argv[4:len(argv)]
            #(features, classNames, fileNames) = featureExtraction.getFeaturesFromDirs(dirNames)
            featureAndTrain(dirNames, modelType, modelName)


    if argv[1] == "-evaluateClassifierFromFile":         # evaluate a classifier for features stored in file
        # to be used after featureExtraction's getFeaturesFromDirs functionality. Eg:
        # python featureExtraction.py -featuresDirs 4classes  data/beach/ data/building/ data/faces/ daa/groups/
        # python train.py -evaluateClassifierFromFile 4classes_features svm 10 0.1 1 2
        if len(argv)>=8:
            featureFile = argv[2]
            classifierType = argv[3]
            nExp = int(argv[4])
            dimRedFile = argv[5]
            method = int(argv[6])
            if classifierType=="svm+knn":
                cParams = [float(c) for c in argv[7::]]
                if len(cParams) % 2 != 0:
                    print "Number of classifier Params must be event for the svm+knn combination!"
                    return
                else:
                    cParams1 = cParams[0:len(cParams)/2]
                    cParams2 = cParams[len(cParams)/2::]
                    cParams = []
                    for p1 in cParams1:
                        for p2 in cParams2:
                            cParams.append([p1, p2])                
            else:
                cParams = [float(c) for c in argv[7::]]


            fo = open(featureFile, "rb")
            features     = cPickle.load(fo)
            classNames     = cPickle.load(fo)
            fileNames  = cPickle.load(fo)
            featureNames   = cPickle.load(fo)
            fo.close()

            #features = features[0:2]
            #classNames = classNames[0:2]
            #print classNames
            evaluateClassifier(features, fileNames, classNames, nExp, classifierType, cParams, 0, method, dimRedFile)

    if argv[1] == "-classifyFile":
        if len(argv)>=5:
            modelType = argv[2]
            if modelType=="svm+knn":
                modelName = [argv[3], argv[4]]
                fileToClassify = argv[5]
            else:
                modelName = argv[3]
                fileToClassify = argv[4]        
            [P, C] = fileClassification(fileToClassify, modelName, modelType)            
            for c, p in zip(C,P):                
                print "{0:20s} \t {1:15.1f}%".format(c,100*p);

    if argv[1] == "-classifyFileMultiLabel":
        if len(argv)==4:
            Threshold = 0.050
            modelName = argv[2]
#            modelName = argv[3]
            fileToClassify = argv[3]        

            [P, C] = fileClassification(fileToClassify, modelName, "gmmNoNormalize")
            
            for c, p in zip(C,P):
                if p > Threshold:
                    print "{0:20s} \t {1:15.1f}%".format(c,p);


    if argv[1] == "-classifyFolderSentiment":
        if len(argv)==5:
            modelType = argv[2]
            modelName = argv[3]
            folder = argv[4]        
            #print modelType, modelName, fileToClassify
            dirClassificationSentiment(folder, modelName, modelType)
            #[P, C] = fileClassification(fileToClassify, modelName, modelType)
            
            #P = P[0]
            #for c, p in zip(C,P):                
            #    print "{0:20s} \t {1:15.1f}%".format(c,100*p);


if __name__ == '__main__':
    main(sys.argv)
