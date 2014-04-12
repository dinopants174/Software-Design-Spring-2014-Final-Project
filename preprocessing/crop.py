# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:28:20 2014

@author: swalters
"""

from PIL import Image
import numpy

# file reading and writing methods
def open_file(filename):
    return Image.open(filename)
    
def save_image(im, filename):
    im.save(filename)


# actual cropping method
def all_crops(im):
    [crop_bw_pix, rows, cols] = im_to_size_px(im, 50)
    a = pix_to_array(crop_bw_pix, rows, cols)
    
    # get bounding dimensions (for removing whitespace)
    [l, r, t, b] = strip_array(a)
    print [l, r, t, b]
    
    [save_bw_pix, rows, cols] = im_to_size_px(im, 100)

    # make new image, cropped
    res = Image.new('L',(r-l, b-t))
    res_pix = res.load()
    for i in range(l, r):
        for j in range(t, b):
            res_pix[i-l, j-t] = save_bw_pix[i, j]

    return res

# convert image to thresholded set of pixels
def im_to_size_px(im, threshold):
    gray = im.convert('L')
    crop_bw = gray.point(lambda x: 0 if x<threshold else 255, '1')
    crop_bw_pix = crop_bw.load()
    (cols, rows) = im.size
    
    radius = 5
    for r in range(rows):
        for c in range(cols):
            if crop_bw_pix[c,r] == 0:
                dark = 0
                for mr in range(max(0,r-radius), min(rows,r+radius)):
                    for mc in range(max(0,c-radius), min(cols,c+radius)):
                        if crop_bw_pix[mc, mr] == 0:
                            dark += 1
                if dark < 1000:
                    crop_bw_pix[c,r] == 255
                        
    return [crop_bw_pix, rows, cols]

# convert set of pixels to array
def pix_to_array(pix, rows, cols):
    array = []
    for r in range(rows):
        row = []
        for c in range(cols):
            val = pix[c,r]
            if val == 255:
                row.append(1)
            else:
                row.append(0)
        array.append(row)
    return numpy.matrix(array) # dimensions R x C (matches image layout)
  
# remove bounding whitespace
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

# check whether an array is all white
def all_white(a):
    (rows, cols) = a.shape
    for c in range(cols):
        for r in range(rows):
            if a[r,c] != 1:
                return False
    return True
                
if __name__ == '__main__':
    filename = 'Doyung_Zoher_Test.jpg'
    im = open_file(filename)
    cropped = all_crops(im)
    save_image(cropped, 'cp_'+filename)
