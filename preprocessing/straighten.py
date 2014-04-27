# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop
import numpy
from PIL import Image

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
    
def read_image_im(im):
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
    return line_rows
    
def draw_horizontals_im(im):
    ''' draws straight lines over the strongest horizontals in an image
        input: image object
        output: image object with superimposed lines
    '''
    # get image array and pixel source
    a = read_image_im(im)
    (rows, cols) = a.shape
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

    # return
    return im
    
def draw_horizontals_a(a, a2):
    ''' draws straight lines over the strongest horizontals in an image
        input: image object
        output: image object with superimposed lines
    '''
    # get image array and pixel source
    (rows, cols) = a.shape

    # loop through rows checking for darkness
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < 0.4*255*cols: # hard-coded, as of now
           for c in range(cols):
                a2[r,c] = 0
    return a2
    
def contains_non_binary(a):
    (rows, cols) = a.shape
    for r in range(rows):
        for c in range(cols):
            if a[r,c] not in [1,0]: return True
    return False
    
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
    return line_cols
    
def draw_lines(filename):
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
            
    # remove similar values in line_rows
    main_rows = []
    for row in line_rows:
        accountedFor = False
        for i in range(row+50):
            if i in main_rows:
                accountedFor = True
        if accountedFor == False:
            main_rows.append(i)
            
    # draw dark rows
    for r in main_rows:
        for c in range(cols-1):
            bw_pix[c,r] = 175
            
    # loop through columns checking for darkness
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < 0.8*rows: # hard-coded, as of now
            line_cols.append(c)
            
    # remove similar values in line_rows
    main_cols = []
    for col in line_cols:
        accountedFor = False
        for i in range(col+50):
            if i in main_cols:
                accountedFor = True
        if accountedFor == False:
            main_cols.append(i)
            
    # draw dark columns
    for c in main_cols:
        for r in range(rows-1):
            bw_pix[c,r] = 175
            
    lnname = 'ln_' + filename
    im.save(lnname)
    return [main_rows, main_cols]
    
def intersections(line_rows, line_cols):    
    i = []
    for row in line_rows:
        for col in line_cols:
            i.append((row, col))
    return i   

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

def resize(filename):
    """Used to resize the image because the giant whiteboard picture Doyung and I were working with had a really long run-time"""
    im = Image.open(filename)
    half = 0.25
    out = im.resize( [int(half * s) for s in im.size] )
    small_name = filename
    out.save(small_name)


def component_finder(line_rows, filename, original):
    """Using the resized image and the horizontal lines the draw_horizontals function gives us, we find the average of those lines.
    We then offset those lines by 20px and we look for instances of non-white pixels, indicating an irregularity in the line. For 
    visualization purposes, we draw circles at each instance of non-white pixels but we will eventually aim to crop around each
    component and use the classifier to identify it"""
    
    im = Image.open(filename)
    im2 = Image.open(original)
    width, height = im.size
    bw_pix = im.load()
    avg_line = 0
    for line in line_rows:  #line_rows comes from Sarah's draw_horizontals function
        avg_line += line
    if avg_line == 0:
        return "There is no horizontal line drawn here because the image may not be straight enough"
    offset = 0.15*height
    line = int(float(avg_line)/len(line_rows))
    line_final = int((float(avg_line)/len(line_rows))-offset)
    line_final2 = int(line_final + 2*offset)
    
    a = read_image(filename)
    (rows, cols) = a.shape
    # (rows, cols) = im.shape

    non_white = []
    for i in range(cols):
        if bw_pix[i,line_final] < 25 or bw_pix[i,line_final2] < 25:
            non_white.append(i)
    
    component = []
    all_components = []
    component_counter = 0
    for i in range(len(non_white)-1):
        if non_white[i+1] - non_white[i] > 0.12*width:
            component_counter += 1
            component.append(non_white[i])
            all_components.append(component)
            component = [] 
        else:
            component.append(non_white[i])

    if len(component) != 0:
        all_components.append(component)

    for i in range(len(all_components)):
        name = 'component_' + str(i)
        box = (int(all_components[i][0] - 0.02*width), int(line_final-0.2*height), int(all_components[i][len(all_components[i])-1]+0.02*width), int(line_final2+0.2*height))
        region = im2.crop(box)
        region.save(name + ".jpg")

def draw_circuit(component_id_list):
    im = Image.open('resistor.png')
    width, height = im.size
    num_of_images = len(component_id_list)*2 +1 
    fin_segment = Image.new('L', (num_of_images*width , height), color=255)
    x_coord = 0
    j = 0
    for i in range(1, num_of_images+1):
        if i%2 != 0:
            fin_segment.paste(Image.open('line.png'), box=(x_coord*700, 0))
            x_coord += 1
        if i%2 == 0:
            fin_segment.paste(Image.open(component_id_list[j]+'.png'), box = (x_coord*700, 0))
            x_coord += 1
            j += 1
    fin_segment.show()
    



if __name__ == '__main__':
    # line_rows = draw_horizontals('test_2.jpg')
    # component_finder(line_rows,'hz_test_2.jpg', 'cp_test_2.jpg')
    ryan_list = ['resistor', 'capacitor']
    draw_circuit(ryan_list)

    
    # stashed changes
    # [r,c] = draw_lines('IMAG0722.jpg')
    # print intersections(r,c)
