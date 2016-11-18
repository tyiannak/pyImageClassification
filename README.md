# Sentimagi Python Image Analysis Library

## Requirements
sudo apt-get install python-skimage
sudo pip install svgwrite
sudo apt-get install python-pywt

## General
This library can be used for general image classification and feature extraction.

## Feature extraction:

### Extract and plot features from a single file
```
python featureExtraction.py -featuresFile sampledata/spectrograms/music/m_5_r_139.png
```

### Extract features from two files and compare
```
python featureExtraction.py -featuresFilesCompare sampledata/spectrograms/music/m_5_r_139.png sampledata/spectrograms/speech/kill_bill_2_speech_17.png
```

### Extract features from a set of images stored in a folder
```
python featureExtraction.py -featuresDir sampledata/spectrograms2/music/
```

### Extract features from a set of directories, each one defining an image class
```
python featureExtraction.py -featuresDirs spectrograms sampledata/spectrograms/music sampledata/spectrograms/speech
```
(Features are stored in file "sectrograms_features")

## Training and testing classification - regression models:

### Train an image classification model 

Models are trained from samples stored in folders (one folder per class). 

Examples:

* kNN model training 
```
python train.py -train knn knnSpeechMusicSpecs  sampledata/spectrograms/music sampledata/spectrograms/speech
```
The above example trains a kNN classification model, does cross validation to estimate the best parameter (k value) and stores the model in a file (named knn3Classes).


* SVM model training 
```
python train.py -train svm svmSpeechMusicSpecs  sampledata/spectrograms/music sampledata/spectrograms/speech
```
The above example trains an SVM classification model, does cross validation to estimate the best parameter (C value) and stores the model in a file (named svmSentimentAds).


### Classify an unknown image examples
```
python train.py -classifyFile knn knnSpeechMusicSpecs sampledata/music.melodies_snatch_0081.png
python train.py -classifyFile knn knnSpeechMusicSpecs sampledata/s_30_r_335.png

```



