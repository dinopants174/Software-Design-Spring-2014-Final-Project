# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:56:27 2014

@author: swalters
"""

import cv2
import numpy as np

img = cv2.imread('lines.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 1000) #apertureSize = 3)
print edges
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)