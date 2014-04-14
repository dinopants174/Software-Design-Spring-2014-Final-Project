# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop
import numpy
import Image

'''***********
file reading methods
***********'''
def read_image(filename):
    ''' crops image, converts to black/white, and produces a numpy array representation
        input: image filename
        output: pixel array representing cropped, b/w conversion
    '''
    crop.process(filename) # bw's, crops, saves image as cp_filename
    im = crop.open_file('cp_'+filename)
    [bw_pix, rows, cols] = crop.im_to_size_px(im, 100)
    return crop.pix_to_array(bw_pix, rows, cols)
   
   
'''***********
line drawing methods
***********'''
def draw_horizontals(filename):
    ''' draws straight lines over the strongest horizontals in an image
        input: image filename
        output: image with superimposed lines saved to file
    '''
    # get image array and pixel source
    a = read_image(filename)
    (rows, cols) = a.shape
    im = Image.open('cp_'+filename)
    bw_pix = im.load()
    
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < 0.8*cols: # hard-coded, as of now
            line_rows.append(r)
            
    # draw dark rows
    for r in line_rows:
        for c in range(cols-1):
            bw_pix[c,r] = 175
            
    # save
    hzname = 'hz_' + filename
    im.save(hzname)
    
def draw_verticals(filename):
    ''' draws straight lines over the strongest verticals in an image
        input: image filename
        output: image with superimposed lines saved to file
    '''
    # get image array and pixel source
    a = read_image(filename)
    (rows, cols) = a.shape
    im = Image.open('cp_'+filename)
    bw_pix = im.load()
    
    # loop through columns checking for darkness
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < 0.8*rows: # hard-coded, as of now
            line_cols.append(c)
            
    # draw dark columns
    for c in line_cols:
        for r in range(rows-1):
            bw_pix[c,r] = 175
            
    # save
    vtname = 'vt_' + filename
    im.save(vtname)

def darkness(line):
    ''' computes average darkness of a numpy array of 1's and 0's
        input: numpy array line (represents row or column, probably)
        output: average darkness (between 0 and 1)
    '''
    (rows, cols) = line.shape
    dkSum = 0
    for i in range(rows):
        for j in range(cols):
            dkSum += line[i,j]
    return dkSum/float(line.size)
    
    
'''***********
standard deviation analysis
***********'''
def hz_or_vert(a):
    ''' makes a guess at whether an image contains horizontal or vertical lines using standard deviation
        input: numpy matrix
        output: [row darkness standard deviation, column darkness standard deviation]
            large r_std and small c_std indicates horizontal lines; opposite indicates vertical lines
    '''
    dk_rows = []
    dk_cols = []
    (rows, cols) = a.shape
    
    # collect average darknesses for rows and columns
    for r in range(rows):
        dk_rows.append(255-darkness(a[r,:]))
    for c in range(cols):
        dk_cols.append(255-darkness(a[:,c]))
        
    # compute standard deviations
    # large r_std and small c_std indicates horizontal lines; opposite indicates vertical lines
    r_std = numpy.std(dk_rows)
    c_std = numpy.std(dk_cols)
    
    print 'Average row darkness stdev: ' + str(r_std)
    print 'Average column darkness stdev: ' + str(c_std)
    
    return [r_std, c_std]
    

'''***********
main method
***********'''
if __name__ == '__main__':
    print 'hello'