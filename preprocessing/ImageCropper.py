# -*- coding: utf-8 -*-
"""
Created on Sat May  3 18:28:21 2014

@author: swalters
"""

from PIL import Image
import numpy

################################################################################################

'''METHODS: (** denotes main functionality)
    :: take filename
        -> open_file (creates image object from image at filename)
    
    :: take image
        -> threshold (creates black/white image object from rgb image object)
        -> **smart_crop (creates cropped image object from non-cropped image object)
    
    :: take pixel object
        -> pix_to_array (creates numpy array of 1's (w) and 0's (b) from pixel object)
    
    :: take numpy array
        -> array_bounds (finds bounding box which removes whitespace from edges of array object)
        -> is_white (checks whether array contains only 1's)
'''

###############################################################################################

''' ------ TAKE FILENAME------ '''
def open_file(filename):
    ''' creates image object from image at filename
        input: filename (.jpg, .png, .tif, etc)
        output: image object
    '''
    return Image.open(filename)


''' ------ TAKE IMAGE OBJECT ------ '''
def threshold(im, t):
    ''' converts image object to thresholded black/white image object
        input: image object, black vs white threshold (close to 0 = lighter result, close to 255 = darker result)
        output: black/white image object
    '''
    gray = im.convert('L') # switch modes - to grayscale from rgb
    bw = gray.point(lambda x: 0 if x<t else 255, '1') # 1 is white, 0 is black          
    return bw

def smart_crop(im):
    ''' crops image to remove bounding whitespace
        input: image object
        output: cropped image object
    '''
    # get crop bounds
    toCrop = threshold(im, 50) # overthreshold - recognize only actual dark part, not extra dots
    toCrop_pix = toCrop.load()
    (c, r) = toCrop.size # counts x width (number of columns) first
    a = pix_to_array(toCrop_pix, r, c)
    box = array_bounds(a) # (left, top, right, bottom) tuple
   
    # make cropped image
    toSave = threshold(im, 50) # more normal-looking threshold
    return toSave.crop(box) # built-in image method
    

''' ------ TAKE PIXEL OBJECT ------ '''
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


''' ------ TAKE NUMPY ARRAY ------ '''
def array_bounds(a):
    ''' removes bounding whitespace (full-height or full-width blocks of 1's adjacent to edges) from pixel value array
        input: numpy matrix representing pixel object (from pix_to_array)
        output: (left, top, right, bottom) bounding box
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
    
    return (left, top, right, bottom)
    
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
    
###############################################################################################

''' ------ EXAMPLE USE CASE ------ '''
if __name__ == '__main__':
    directory = 'TestImages/'
    filename = 'Doyung_Zoher_Test.jpg'
    im = open_file(directory+filename)
    cropped = smart_crop(im)
    cropped.save(directory+'cp_'+filename)