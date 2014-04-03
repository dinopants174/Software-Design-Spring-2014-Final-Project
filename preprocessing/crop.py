# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:28:20 2014

@author: swalters
"""


import Image
import numpy

def all_crops(filename):
    # open image and convert to black/white
    im = Image.open(filename)
    gray = im.convert('L')
    bw = gray.point(lambda x: 0 if x<200 else 255, '1')
    bw_pix = bw.load()
    
    # get image size and convert to rows x cols array
    (cols, rows) = im.size
    a = pix_to_array(bw_pix, rows, cols)
    
    # get bounding dimensions (for removing whitespace)
    [l, r, t, b] = strip_array(a)

    # save to new image, cropped
    res = Image.new('L',(r-l, b-t))  
    res_pix = res.load()
    for i in range(l, r):
        for j in range(t, b):
            res_pix[i-l, j-t] = bw_pix[i, j]
    croppedname = 'cp_' + filename
    res.save(croppedname)

def pix_to_array(pix, rows, cols):
    array = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(pix[c,r])
        array.append(row)
    return numpy.matrix(array) # dimensions R x C (matches image layout)
    
def strip_array(a):
    (rows, cols) = a.shape
    
    left = -1
    # left to right
    for c in range(cols):
        if not all_white(a[:,c]):
            left = c
            break
        
    right = -1
    # right to left
    for c in range(cols-1, 0, -1):
        if not all_white(a[:,c]):
            right = c+1
            break
        
    top = -1
    # top to bottom
    for r in range(rows):
        if not all_white(a[r,:]):
            top = r
            break
        
    bottom = -1
    # bottom to top
    for r in range(rows-1, 0, -1):
        if not all_white(a[r,:]):
            bottom = r+1
            break
        
    return [left, right, top, bottom]
        
def all_white(a):
    (rows, cols) = a.shape
    for c in range(cols):
        for r in range(rows):
            if a[r,c] != 255:
                return False
    return True
                
if __name__ == '__main__':
    all_crops('fnord.tif')