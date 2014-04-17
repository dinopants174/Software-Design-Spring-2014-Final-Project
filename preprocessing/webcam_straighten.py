import cv2
import numpy as np
import straighten as s

c = cv2.VideoCapture(0)

while(1):
    _,f = c.read()
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #im_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    lines = s.draw_horizontals_a(im_bw, gray)
    cv2.imshow('Frame',gray)
    if cv2.waitKey(5)==ord('q'):
        break

cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

print 'done'