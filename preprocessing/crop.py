# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:28:20 2014

@author: swalters
"""

from PIL import Image
import numpy

'''************
file reading and writing methods
************'''
def open_file(filename):
    ''' creates image object from image at filename
        input: filename (.jpg, .png, .tif, etc)
        output: image object
    '''
    return Image.open(filename)
    
def save_image(im, filename):
    ''' saves image object to file
        input: image object im, string filename
        output: file is saved; no return value
    '''
    im.save(filename)


'''************
use cases
************'''
def process(filename): 
    ''' black/white converts and crops image
        input: filename (string)
        output: cropped image is saved to 'cp_'+filename
    '''
    im = open_file(filename)
    cropped = all_crops(im) # remove bordering whitespace
    save_image(cropped, 'cp_'+filename) # add prefix so as not to overwrite original image file


'''************
actual cropping method
************'''
def all_crops(im):
    ''' removes bounding whitespace from image
        input: uncropped image object
        output: cropped image object
    '''
    # crop image
    [crop_bw_pix, rows, cols] = im_to_size_px(im, 50) # threshold low intentionally - looks wrong, but cuts down on insignificant dark spots for cropping
    a = pix_to_array(crop_bw_pix, rows, cols)
    [l, r, t, b] = strip_array(a) # bounding dimensions
    
    # make black/white image to use as pixel source for cropped image
    [save_bw_pix, rows, cols] = im_to_size_px(im, 100) # more visually appropriate threshold than for crop_bw_pix
    
    # make new image, cropped size
    res = Image.new('L',(r-l, b-t)) # r-l = new width, b-t = new height
    res_pix = res.load()
    
    # fill new image
    for i in range(l, r):
        for j in range(t, b):
            res_pix[i-l, j-t] = save_bw_pix[i, j] # res_pix[0,0] is save_bw_pix[l,t]

    return res # image object


'''************
helper methods
************'''
def im_to_size_px(im, threshold):
    ''' converts image to thresholded set of pixel values (all 0 or 255)
        input: image, black vs white threshold (close to 0 = lighter result, close to 255 = darker result)
        output: [cropped pixel object, number of rows, number of columns]
    '''
    gray = im.convert('L')
    crop_bw = gray.point(lambda x: 0 if x<threshold else 255, '1')
    crop_bw_pix = crop_bw.load()
    (cols, rows) = im.size
    
    ### unnecessary? attempt to point lone black pixels to white so cropping works
    #radius = 5
    #for r in range(rows):
    #    for c in range(cols):
    #        if crop_bw_pix[c,r] == 0:
    #            dark = 0
    #            for mr in range(max(0,r-radius), min(rows,r+radius)):
    #                for mc in range(max(0,c-radius), min(cols,c+radius)):
    #                    if crop_bw_pix[mc, mr] == 0:
    #                        dark += 1
    #            if dark < 1000:
    #                crop_bw_pix[c,r] == 255                 
    return [crop_bw_pix, rows, cols]

def pix_to_array(pix, rows, cols):
    ''' converts set of pixels to array
        input: pixel object, number of rows, number of columns (from im_to_size_px)
        output: numpy matrix representing pixel set; all 1's (white) and 0's (black)
    '''
    array = []
    for r in range(rows):
        row = [] # work through pixel object row by row
        for c in range(cols):
            val = pix[c,r] # for some reason, pixel objects are indexed backwards
            if val == 255:
                row.append(1)
            else:
                row.append(0)
        array.append(row)
    return numpy.matrix(array) # dimensions R x C (matches image layout)
  
def strip_array(a):
    ''' removes bounding whitespace (full-height or full-width blocks of 1's adjacent to edges) from pixel value array
        input: numpy matrix representing pixel object (from pix_to_array)
        output: [left, right, top, bottom] bounding indices
    '''
        
    (rows, cols) = a.shape
    
    left = -1
    # left to right
    for c in range(cols):
        if not all_white(a[:,c]):
            left = c # leftmost column with a dark pixel
            break
        
    right = -1
    # right to left
    for c in range(cols-1, 0, -1):
        if not all_white(a[:,c]):
            right = c+1 # rightmost column with a dark pixel, plus one because Python ranges are inclusive of start but not end
            break
        
    top = -1
    # top to bottom
    for r in range(rows):
        if not all_white(a[r,:]):
            top = r # uppermost column with a dark pixel
            break
        
    bottom = -1
    # bottom to top
    for r in range(rows-1, 0, -1):
        if not all_white(a[r,:]):
            bottom = r+1 # lowest row with a dark pixel, again plus one
            break
    
    return [left, right, top, bottom]

def all_white(a):
    ''' check whether an array is all white
        input: numpy matrix
        output: boolean (true if matrix is all 1's false otherwise)
    '''
    (rows, cols) = a.shape
    for c in range(cols):
        for r in range(rows):
            if a[r,c] != 1:
                return False
    return True


'''************
for testing
************'''
if __name__ == '__main__':
    filename = 'Doyung_Zoher_Test.jpg'
    process(filename)