# SmarterBoards Software Design Spring 2014 Final Project
# An affiliate of Lazy Man Notes - Olin Project, Inc.
# Doyung Lee and Ryan Louie 

# -*- coding: utf-8 -*-
import os
import sys

from scipy import misc
from skimage import io, transform, exposure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

TRAIN_DATA_DIR = "C:\\Users\\rlouie\\Documents\\GitHub\\SmarterBoard\\circuits\\data\\"

# --- Functions

def getTrainFilenames(n=75):
    filenames = os.listdir(TRAIN_DATA_DIR)
    np.random.shuffle(filenames)
    filenames = filenames[:n]
    return filenames

def isResistorFromFilename(filenames):
    is_resistor = [fn[:3]=="res" for fn in filenames]
    return is_resistor

def loadImage(filename):
    scaler = 15
    image = misc.imread(filename)
    h = []
    for channel in range(3):
        tmp = image.astype(np.float64)
        h.append(exposure.histogram(tmp[:,:,channel], nbins=10)[0])
    image = transform.resize(image, (int(scaler*2.56), int(scaler*1.53))) # small image 256px x 153px, largeimage 2592px 1552px
    image = image.flatten()
    h = np.array(h)
    h = h.flatten()
    return np.hstack((image, h))

def loadTrain(n=75, verbose=False):
    filenames = getTrainFilenames(n)
    is_resistor = isResistorFromFilename(filenames)
    I = []
    if verbose:
        for i in range(n):
            fn = filenames[i]
            print "loading image " + fn
            I.append(loadImage(TRAIN_DATA_DIR + fn))
    else:
        for i in range(n):
            fn = filenames[i]
            sys.stdout.write(".")
            I.append(loadImage(TRAIN_DATA_DIR + fn))      
    return I, is_resistor

def showcomponent(fn="resistor1.jpg"):
    """ displays resistor1.jpg """
    filename = (TRAIN_DATA_DIR + fn)
    image = misc.imread(filename)
    plt.imshow(image)
    plt.show()

# --- Do Machine Learning!

# Load Data
I, is_resistor = loadTrain(84)
I, is_resistor = (np.array(I), np.array(is_resistor))

# Split into Train/Test sets
I_train, I_test, is_resistor_train, is_resistor_test = train_test_split(I, is_resistor, train_size=.6)

# Train Model
clf = RandomForestClassifier()
clf.fit(I_train, is_resistor_train)

# Evaluate Predictions
is_resistor_predict = clf.predict(I_test)
print "accuracy: " + str(accuracy_score(is_resistor_test,is_resistor_predict))

# Confusion Matrix
cm = confusion_matrix(is_resistor_test, is_resistor_predict)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.clim(vmin=0)
plt.show()