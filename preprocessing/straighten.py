# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop
import numpy
from PIL import Image, ImageDraw

def read_image(filename):
    ''' crops image, converts to black/white, and produces a numpy array representation
        input: image filename
        output: pixel array representing cropped, b/w conversion
    '''
    crop.process(filename) # bw's, crops, saves image as cp_filename
    im = crop.open_file('cp_'+filename)
    [bw_pix, rows, cols] = crop.im_to_size_px(im, 100)
    return crop.pix_to_array(bw_pix, rows, cols)
    
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
    
def draw_horizontals(filename):
    a = read_image(filename)
    (rows, cols) = a.shape
    line_rows = []
    im = Image.open('cp_'+filename)
    bw_pix = im.load()
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < 0.8*cols:
            line_rows.append(r)
    for r in line_rows:
        for c in range(cols-1):
            bw_pix[c,r] = 175
    hzname = 'hz_' + filename
    im.save(hzname)
    return line_rows
    
def draw_verticals(filename):
    a = read_image(filename)
    (rows, cols) = a.shape
    line_cols = []
    im = Image.open('cp_'+filename)
    bw_pix = im.load()
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < 0.8*rows:
            line_cols.append(c)
    for c in line_cols:
        for r in range(rows-1):
            bw_pix[c,r] = 175
    vtname = 'vt_' + filename
    im.save(vtname)
    return line_cols

def darkness(line):
    (rows, cols) = line.shape
    dkSum = 0
    for i in range(rows):
        for j in range(cols):
            dkSum += line[i,j]
    return dkSum/float(line.size)

def resize(filename):
    """Used to resize the image because the giant whiteboard picture Doyung and I were working with had a really long run-time"""
    im = Image.open(filename)
    half = 0.25
    out = im.resize( [int(half * s) for s in im.size] )
    small_name = 'small_' + filename
    out.save(small_name)


def component_finder(line_rows, filename):
    """Using the resized image and the horizontal lines the draw_horizontals function gives us, we find the average of those lines.
    We then offset those lines by 20px and we look for instances of non-white pixels, indicating an irregularity in the line. For 
    visualization purposes, we draw circles at each instance of non-white pixels but we will eventually aim to crop around each
    component and use the classifier to identify it"""
    
    im = Image.open(filename)
    width, height = im.size
    bw_pix = im.load()
    avg_line = 0
    for line in line_rows:  #line_rows comes from Sarah's draw_horizontals function
        avg_line += line
    offset = 0.045*height   #
    line_final = int(float(avg_line)/len(line_rows))-offset
    line_final2 = line_final + 2*offset
    
    a = read_image(filename)
    (rows, cols) = a.shape

    r = []
    L = []
    draw = ImageDraw.Draw(im)
    whitespace = 0
    for i in range(cols):
        if bw_pix[i,line_final] < 25 or bw_pix[i,line_final2] < 25:
            L.append(i)
        if bw_pix[i, line_final] < 25 and bw_pix[i+1, line_final] == 255 or bw_pix[i, line_final2] < 25 and bw_pix[i+1, line_final2] == 255:
            r.append(L)
    for i in r:
        draw.line((i[0], line_final, i[len(i)-1], line_final), width = 10)
    im.show()
    #dotname = 'dot_' + filename
    #im.save(dotname)

if __name__ == '__main__':
    line_rows = draw_horizontals('cp2_Doyung_Zoher_Test.jpg')
    component_finder(line_rows,'hz_cp2_Doyung_Zoher_Test.jpg')