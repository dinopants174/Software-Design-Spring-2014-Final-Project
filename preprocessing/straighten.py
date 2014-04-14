# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop
import numpy
from PIL import Image, ImageDraw

def read_image(filename):
    [bw_pix, rows, cols] = crop.im_to_size_px(filename)
    return [crop.pix_to_array(bw_pix, rows, cols), rows, cols]
    
def find_darkest(a):
    dk_rows = []
    dk_cols = []
    (rows, cols) = a.shape
    for r in range(rows):
        dk_rows.append(255-darkness(a[r,:]))
    for c in range(cols):
        dk_cols.append(255-darkness(a[:,c]))
    r_std = numpy.std(dk_rows)
    c_std = numpy.std(dk_cols)
    print 'Average row darkness stdev: ' + str(r_std)
    print 'Average column darkness stdev: ' + str(c_std)
    return [r_std, c_std]
    
def draw_horizontals(filename):
    [a, rows, cols] = read_image(filename)
    [r_std, c_std] = find_darkest(a)
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < 0.8*cols:
            line_rows.append(r)
    
    im = Image.open(filename)
    px = im.load()
    for r in line_rows:
        for c in range(cols-1):
            px[c,r] = 175
    hzname = 'hz_' + filename
    im.save(hzname)
    return line_rows

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
    
    avg_line = 0
    for line in line_rows:
        avg_line += line

    [a, rows, cols] = read_image(filename)
    im = Image.open(filename)
    im_width, im_height = im.size
    print im_height
    offset = im_height*0.04
    print offset
    line_final = int(float(avg_line)/len(line_rows))-offset
    line_final2 = line_final + 2*offset

    px = im.load()
    list_of_nw = []
    for i in range(cols):
        if px[i,line_final] < 25 or px[i, line_final2] < 25:
            
        #     draw = ImageDraw.Draw(im)
        #     draw.ellipse((i-r, line-r, i+r, line+r), fill=0)
        # elif px[i, line_final2] < 25:
        #     draw = ImageDraw.Draw(im)
        #     draw.ellipse((i-r, line-r, i+r, line+r), fill=0)
    im.show()
    # dotname = 'dot_' + filename
    # im.save(dotname)

def draw_verticals(filename):
    [a, rows, cols] = read_image(filename)
    [r_std, c_std] = find_darkest(a)
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < 0.8*rows:
            line_cols.append(c)
    
    im = Image.open(filename)
    px = im.load()
    for c in line_cols:
        for r in range(rows-1):
            px[c,r] = 175
    vtname = 'vt_' + filename
    im.save(vtname)

def darkness(line):
    (rows, cols) = line.shape
    dkSum = 0
    for i in range(rows):
        for j in range(cols):
            dkSum += line[i,j]
    return dkSum/float(line.size)

if __name__ == '__main__':
    #resize('cp_Doyung_Zoher_Test.jpg')
    #line_rows = draw_horizontals('small_cp_Doyung_Zoher_Test.jpg')
    #component_finder(line_rows, 'hz_small_cp_Doyung_Zoher_Test.jpg')
