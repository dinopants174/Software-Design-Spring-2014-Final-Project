"""
SmarterBoard Component Recognition 
author: rlouie
"""
# --- Imports

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Image
from scipy import misc
from skimage import io, transform, exposure
from skimage.filter import threshold_otsu, gabor_filter

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# --- Linux vs Windows Data Directory Compatibility

OS = os.name # 'nt' if in Windows 7

if OS == 'nt':
    TRAIN_DATA_DIR = os.path.join(os.path.abspath('.'),'data\\')
else:
    TRAIN_DATA_DIR = os.path.join(os.path.abspath('.'),'data/')
    
NUM_TRAIN = len(os.listdir(TRAIN_DATA_DIR))

class Preprocessing:
    
    @staticmethod
    def binary_from_thresh(img):
        """ Base function for converting to binary using a threshold """
        thresh = threshold_otsu(img)
        binary = img < thresh
        return binary

    @staticmethod
    def binary_from_laplacian(img):
        """ Function that converts an image into binary using
        a laplacian transform and feeding it into binary_from_thresh """
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        return Preprocessing.binary_from_thresh(laplacian)

    @staticmethod
    def scale_image(img,scaler,org_dim):
        """ resizes an image based on a certain scaler 
        args: 
            scaler: int or float. A value of 1.0 would output a image.shape = org_dim
            org_dim: tuple. Denoting (width, height)
        returns: ndarray. Scaled image """
        width, height = org_dim
        output_size = (int(scaler*width), int(scaler*height))
        return cv2.resize(img, output_size)
    
    @staticmethod
    def standardize_shape(img):
        """ standardizes the shape of the images to a tested shape for gabor filters """
        return Preprocessing.scale_image(img, scaler=.25, org_dim=(256,153))
    
    @staticmethod
    def angle_pass_filter(img, frequency, theta):
        """ returns the magnitude of a gabor filter response for a certain angle """
        real, imag = gabor_filter(img, frequency, theta)
        mag = np.sqrt(np.square(real) + np.square(imag))
        return mag
    
    @staticmethod
    def frontslash_filter(img,denom=5.0,freq=0.4):
        """ intensifies edges that look like a frontslash '/' """
        theta = np.pi*(1.0/denom)
        return Preprocessing.angle_pass_filter(img,freq,theta)
    
    @staticmethod
    def backslash_filter(img,denom=5.0,freq=0.4):
        """ intensifies edges that look like a backslash '\' """
        theta = np.pi*(1.0 - 1.0/denom)
        return Preprocessing.angle_pass_filter(img,freq,theta)


class FeatureExtraction:
    
    @staticmethod
    def mean_exposure_hist(nbins, *images):
        """ calculates mean histogram of many exposure histograms
        args:
            nbins: number of bins
            *args: must be images (ndarrays) i.e r1, r2
        returns:
            histogram capturing pixel intensities """
        hists = []
        for img in images:
            hist, _ = exposure.histogram(img,nbins)
            hists.append(hist)
        return np.sum(hists, axis=0) / len(images)
    
    @staticmethod
    def mean_exposure_hist_from_gabor(img,nbins):
        frontslash = Preprocessing.frontslash_filter(img)
        backslash = Preprocessing.backslash_filter(img)
        return np.array(FeatureExtraction.mean_exposure_hist(nbins,frontslash,backslash))

def getTrainFilenames(n):
    filenames = os.listdir(TRAIN_DATA_DIR)
    np.random.shuffle(filenames)
    filenames = filenames[:n]
    return filenames

def isResistorFromFilename(filenames):
    is_resistor = [fn[0]=="r" for fn in filenames]
    return is_resistor

def loadImage(filename, nbins):
    image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE) 
    image = Preprocessing.standardize_shape(image)
    h = FeatureExtraction.mean_exposure_hist_from_gabor(image,nbins)
    image = image.flatten()
    return np.hstack((image, h)) # h takes up the last nbins rows of the feature vector

def loadTrain(n, nbins, verbose=False):
    filenames = getTrainFilenames(n)
    is_resistor = isResistorFromFilename(filenames)
    I = []
    if verbose:
        for i in range(n):
            fn = filenames[i]
            print os.sys.stdout.write('.')
            I.append(loadImage(TRAIN_DATA_DIR + fn, nbins))
    else:
        for i in range(n):
            fn = filenames[i]
            I.append(loadImage(TRAIN_DATA_DIR + fn, nbins))      
    return I, is_resistor

# Component Classification System

def component_clf_sys(nbins, clf):
    """ returns an accuracy score for X (pixels + hist)
    
    nbins: number of bins for exposure histogram of the gabor filtered images
    clf: an instantiated classifier object
    """
    X, y = loadTrain(NUM_TRAIN, nbins)
    X, y = (np.array(X), np.array(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    return accuracy

def avg_perf_component_clf_sys(nruns, nbins, clf):
    """ Caculates average accuracy performance from nruns of the classifier system
    
    nruns: number of runs
    nbins: number of bins for the exposure histogram
    clf: instantiated classifier object
    """
    perf = []
    for i in xrange(nruns):
        accuracy = component_clf_sys(nbins, clf) 
        perf.append(accuracy)
    return np.mean(perf)

def compareClassifiers(clfs):
    """ Caculates average performance of the component classification system for given ML Classifiers """
    nruns = 3
    nbins = 3
    for clf in clfs:
        avg_perf = avg_perf_component_clf_sys(3, 3, clf)
        print "\nAverage performance of {} when trained on pixels + histogram (nbins = {}): ".format(clf.__str__(), nbins) + str(avg_perf)

def main():
    classifiers = [
        LogisticRegression(C=0.0001),
        SVC(kernel='linear',C=0.0001)]

    compareClassifiers(classifiers)


if __name__ == '__main__':
    main()