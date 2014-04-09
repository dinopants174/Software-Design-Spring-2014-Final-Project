"""
Webcam Video Capture using OpenCV
Written by Chris Lee through the massive help of OpenCV tutorials
Now belonging to Lazy Man's Notes - Olin Project, Inc.
"""

import numpy as np
import cv2
import straighten

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # automatic thresholding
    #(thresh, im_bw) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #print thresh
    
    # fixed threshold
    im_bw = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]

    # Display the resulting frame
    cv2.imshow('frame',im_bw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()