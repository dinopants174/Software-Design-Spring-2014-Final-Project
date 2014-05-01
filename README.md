Software-Design-Spring-2014-Final-Project
=========================================

This is the base repository for the Software Design Spring 2014 Final Project, SmarterBoard. We have currently split up into two groups. One group is working on rendering a clean circuit diagram from a drawing and the other group is working on compiling programs from hand-written code.

The members working on this project are Ryan Louie, Sarah Walters, Doyung Lee, and Zoher Ghadyali

UPDATE: 4/30/2014

SmarterBoard is fleshing out the idea of converting hand-drawn circuit diagrams into digitally understandable and beautiful formats.  

Respository Organization
========================
/Code_Recognition - Depreciated.  Was the respository for exploring compiling programs from hand-written computer code
/Meeting Notes - Self Explanatory.  No code resides here!

Okay on to the fun stuff:
/circuits - This repository was created the first week when we split up into pairs to explore the circuit diagram recognition problem. Import files and directories are described below:
    /data - Raw colored pictures of all training images (including contents of '/resistors and '/capacitors')
    /ressitors - Raw colored pictures of resistors
    /capacitors - Raw colored pictures of capacitors
    /tests - Black and White photos of data directory, all which have been customly thresholded. Training directory for               bw_componentrecognition.py
    bw_componentrecognition.py - Modules for component classifier.  Most importantly contains class 'ComponentClassifier'                                  which has method 'predict' which is the part of the pipeline after component                                             segmentation
    utils.py - Utils function that bw_componentrecognition.py depends on.
    

