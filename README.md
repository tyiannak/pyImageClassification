# Sentimagi Python Image Analysis Library

## Requirements
sudo apt-get install python-skimage
sudo pip install svgwrite
sudo apt-get install python-pywt

## General
Bla Bla

## Annotation

### Sentiment annotation
In order to annotate a directory of images:
```
python annotateSentimentDirectory.py -annotate data/temp/ user1
python annotateSentimentDirectory.py -annotate data/temp/ user2
```

Each of this process is saved to a seperate file named sentimentAnnotation_username, e.g. sentimentAnnotation_user1. To load the annotations stored in a folder use the following:
```
python annotateSentimentDirectory.py -load . data/temp
```


## Feature extraction:

### Extract and plot features from a single file
```
python featureExtraction.py -featuresFile data/building/building17.jpg
```

### Extract features from two files and compare
```
python featureExtraction.py -featuresFilesCompare data/building/building17.jpg data/building/building18.jpg
```

### Extract features from a set of images stored in a folder
```
python featureExtraction.py -featuresDir data/positive
```

### Extract features from a set of directories, each one defining an image class
```
python featureExtraction.py -featuresDirs bp ADS/brands/BP/positive/ ADS/brands/BP/negative/
```

## Training and testing classification - regression models:

### Train an image classification model 

Models are trained from samples stored in folders (one folder per class). 

Examples:

* kNN model training (3 classes)
```
python train.py -train knn knn3Classes  data/building/ data/faces/ data/beach/
```
The above example trains a kNN classification model, does cross validation to estimate the best parameter (k value) and stores the model in a file (named knn3Classes).

* kNN model training (Sentiment Analysis)
```
python train.py -train knn knnSentimentAads ADS/positive/ ADS/negative/
```
The above example trains a kNN classification model, does cross validation to estimate the best parameter (k value) and stores the model in a file (named knnSentimentAads).


* SVM model training (Sentiment Analysis)
```
python train.py -train svm svmSentimentAds ADS/positive/ ADS/negative/
```
The above example trains an SVM classification model, does cross validation to estimate the best parameter (C value) and stores the model in a file (named svmSentimentAds).

* GMM model training
```
 python train.py -train gmm gmm3Classes data/building/ data/faces/ data/beach/
```

* TODO RBM 

Also, in all cases, an ARFF file is extracted, which can be imported for evaluating classifiers in WEKA.



### Classify an unknown image
```
python train.py -classifyFile svm svm3Classes data/positive/pos3.jpg
```

### Classify an unknown image multi-label (only gmm)

## Sentiment-specific scripts:
### Extract all features for general dataset and brand-specific datasets:
```
python featureExtraction.py -featuresDirs ads ADS/positive/ ADS/negative/
python featureExtraction.py -featuresDirs apple ADS/brands/Apple/positive/ ADS/brands/Apple/negative/
python featureExtraction.py -featuresDirs bp ADS/brands/BP/positive/ ADS/brands/BP/negative/
python featureExtraction.py -featuresDirs cocacola ADS/brands/coca\ cola//positive/ ADS/brands/coca\ cola/negative/
python featureExtraction.py -featuresDirs nike ADS/brands/Nike/positive/ ADS/brands/Nike/negative/
python featureExtraction.py -featuresDirs mac ADS/brands/Mac\ Donalds/positive/ ADS/brands/Mac\ Donalds/negative/
```
### Evaluate a classifier based on extracted features (see previous step):
Image classifiers:
```
python train.py -evaluateClassifierFromFile ads_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile ads_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile ads_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 11 13 15 17 19
```

Text classifiers (requires that images have also an accompaning text folder):
```
python train.py -evaluateClassifierFromFile ads_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile ads_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile ads_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 11 13 15 17 19
```

Fusion
```
python train.py -evaluateClassifierFromFile ads_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile ads_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile ads_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 11 13 15 17 19
```

Test all brand-related models:
```
python train.py -evaluateClassifierFromFile apple_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile apple_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile apple_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile apple_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile apple_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile apple_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile apple_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile apple_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile apple_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9

python train.py -evaluateClassifierFromFile bp_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile bp_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile bp_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile bp_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile bp_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile bp_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile bp_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile bp_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile bp_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9

python train.py -evaluateClassifierFromFile cocacola_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile cocacola_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile cocacola_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile cocacola_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile cocacola_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile cocacola_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile cocacola_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile cocacola_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile cocacola_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9


python train.py -evaluateClassifierFromFile mac_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile mac_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile mac_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile mac_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile mac_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile mac_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile mac_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile mac_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile mac_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9

python train.py -evaluateClassifierFromFile nike_features svm 2000 "" 0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile nike_features svm 2000 "" 1 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile nike_features svm 2000 "" 2 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7
python train.py -evaluateClassifierFromFile nike_features knn 2000 "" 0 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile nike_features knn 2000 "" 1 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile nike_features knn 2000 "" 2 1 3 5 7 9 11 13 15 17 19 21
python train.py -evaluateClassifierFromFile nike_features svm+knn 2000 "" 0 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile nike_features svm+knn 2000 "" 1 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9
python train.py -evaluateClassifierFromFile nike_features svm+knn 2000 "" 2 0.01 0.05 0.1 0.2 0.3 1 3 5 7 9


```

### Train the SVM Image Sentiment Classifier (with feature extraction):
```
python train.py -train svm svmSentimentAds ADS/positive/ ADS/negative/
```
