# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop
import numpy
from PIL import Image, ImageDraw

# from bw_componentrecognition import ComponentClassifier

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
        output: x-coordinates of lines that are drawn on the strongest horizontals in image
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
    
def draw_horizontals_im(segment):
    ''' draws straight lines over the strongest horizontals in an image
        input: image object
        output: image object with superimposed lines
    '''
    # get image array and pixel source
    a = read_image_im(segment.image)
    (rows, cols) = a.shape
    bw_pix = segment.image.load()
    
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < 0.8*cols: # hard-coded, as of now
            line_rows.append(r)
            
    # draw dark rows
    # for r in line_rows:
    #     for c in range(cols-1):
    #         bw_pix[c,r] = 175

    return line_rows
    
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
    ''' for testing, not for functionality
        checks whether an array contains things that aren't 1 or 0
        returns boolean
    '''
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
    #for c in line_cols:
    #    for r in range(rows-1):
    #        bw_pix[c,r] = 175
            
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
        if val < 0.7*cols: # hard-coded, as of now
            line_rows.append(r)
            
    # remove similar values in line_rows
    main_rows = []
    this_row = []
    for row in line_rows:
        if len(this_row) == 0 or (row - this_row[-1]) <= 0.05*cols:
            this_row.append(row)
        else:
            main_rows.append(sum(this_row)/len(this_row))
            this_row = []
    if len(this_row) != 0:
        main_rows.append(sum(this_row)/len(this_row))
        
    # draw dark rows
    #for r in main_rows:
    #   for c in range(cols-1):
    #       bw_pix[c,r] = 175
            
    # loop through columns checking for darkness
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < 0.7*rows: # hard-coded, as of now
            line_cols.append(c)
        
        
    # remove similar values in line_rows
    main_cols = []
    this_col = []
    for col in line_cols:
        if len(this_col) == 0 or (col - this_col[-1]) <= 0.05*rows:
            this_col.append(col)
        else:
            main_cols.append(sum(this_col)/len(this_col))
            this_col = []
    if len(this_col) != 0:
        main_cols.append(sum(this_col)/len(this_col))
            
    # draw dark columns
    #for c in main_cols:
    #   for r in range(rows-1):
    #       bw_pix[c,r] = 175
    
    lnname = 'ln_' + filename
    im.save(lnname)
    return [im, rows, cols, main_rows, main_cols]
    
def intersections(rows, cols):    
    i = []
    for row in rows:
        for col in cols:
            i.append((row, col))
    return i
    
def get_segments(im, rows, cols, main_rows, main_cols):    
    segments = []
    
    for row in main_rows:
        t = row-int(0.05*rows)
        b = row+int(0.05*rows)     
        for i in range(len(main_cols)-1):
            l = main_cols[i]
            r = main_cols[i+1]
            #box = (main_cols[i]+int(0.05*cols), t, main_cols[i+1]-int(0.05*cols), b)
            box = (int(l+0.025*rows), t, int(r-0.025*rows), b)
            cropped = im.crop(box)
            cropped.save('segment_h_' + str(len(segments)) + '.jpg')
            segments.append(Segment(cropped, (l,row), (r, row)))
            
    for col in main_cols:
        l = col-int(0.05*cols)
        r = col+int(0.05*cols)
        for i in range(len(main_rows)-1):
            t = main_rows[i]
            b = main_rows[i+1]
            #box = (l, main_rows[i]+int(0.05*rows), r, main_rows[i+1]-int(0.05*rows))
            box = (l, int(t+0.025*cols), r, int(b-0.025*cols))
            cropped = im.crop(box)
            cropped = cropped.rotate(90)
            cropped.save('segment_v_' + str(len(segments)) + '.jpg')
            segments.append(Segment(cropped, (col, t), (col, b)))
    
    return segments
    

class Segment():
    def __init__(self, image, start, end):
        self.image = image
        self.start = start
        self.end = end
    
    def is_horizontal(self):
        return self.start[1] == self.end[1]

    def length(self):
        if self.is_horizontal():
            return self.end[0]-self.start[0]
        else:
            return self.end[1]-self.start[1]

    def finding_components(self, component_id_list):
        self.component_id_list = component_id_list
    

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


def component_finder(line_rows, segment):
    """Using the resized image and the horizontal lines the draw_horizontals function gives us, we find the average of those lines.
    We then offset those lines by 20px and we look for instances of non-white pixels, indicating an irregularity in the line. For 
    visualization purposes, we draw circles at each instance of non-white pixels but we will eventually aim to crop around each
    component and use the classifier to identify it"""
    
    im = segment.image
    width, height = im.size
    bw_pix = im.load() # could be from original one too
    
    # locate main dark line
    avg_line = 0
    for line in line_rows:  #line_rows comes from Sarah's draw_horizontals function
        avg_line += line
    if avg_line == 0:
        return "There is no horizontal line drawn here because the image may not be straight enough"
        
    # make offset lines
    offset = 0.2*height
    line = int(float(avg_line)/len(line_rows))
    line_final = int(line - offset)
    line_final2 = int(line + offset)
    print line, offset, line_final, line_final2
    print height
    non_white = []

    for i in range(width):
        if bw_pix[i,line_final] < 25 or bw_pix[i,line_final2] < 25:
            non_white.append(i)

    # find components using deviations
    component = []
    all_components = []
    component_counter = 0
    for i in range(len(non_white)-1):
        # have found end of component - add to list of all components and clear component list
        if non_white[i+1] - non_white[i] > 0.1*width:  #0.5*width  # assumption about component spacing
            component_counter += 1
            component.append(non_white[i])
            all_components.append(component)
            component = [] 
        # not the end of the component yet - keep going
        else:
            component.append(non_white[i])
    print component_counter

    # if last leftover component isn't empty (won't have been added b/c array hasn't had a chance to clear), add it to all components too
    if len(component) != 0:
        all_components.append(component)

    all_comps_cropped = []
    # name components, crop them out, and save them
    for i in range(len(all_components)):
        name = 'component_' + str(i)
        box = (int(all_components[i][0] - 0.02*width), 0, int(all_components[i][len(all_components[i])-1]+0.02*width), height)
        region = im.crop(box)
        region.show()
        all_comps_cropped.append(region)

    return all_comps_cropped

def draw_segment(component_id_list):
    im = Image.open('resistor.png')
    width, height = im.size
    num_of_images = len(component_id_list)*2+1 
    fin_segment = Image.new('L', (num_of_images*width , height), color=255)
    x_coord = 0
    j = 0
    for i in range(1, num_of_images+1):
        if i%2 != 0:
            fin_segment.paste(Image.open('line.png'), box=(x_coord*width, 0))
            x_coord += 1
        if i%2 == 0:
            fin_segment.paste(Image.open(component_id_istl[j]+'.png'), box = (x_coord*width, 0))
            x_coord += 1
            j += 1
    fin_segment.save(filename)

def draw_segment2(segment):
    fin_segment = Image.new('RGBA', (segment.length(), 700))
    len_of_images = segment.length()/700
    x_coord = 0
    width = 700
    layer = Image.open('line.png')
    layer = layer.convert('RGBA')
    for i in range(len_of_images):
        fin_segment.paste(layer, box=(x_coord*width, 0), mask=layer)
        x_coord += 1
    fin_segment.paste(layer, box=(segment.length()-700, 0), mask=layer)

    all_comps = segment.component_id_list
    x_coord = 1
    for i in range(len(all_comps)):
        fin_segment.paste(Image.open(all_comps[i]+'.png'), box=((x_coord * segment.length()/(len(all_comps)+1))-350, 0))
        x_coord += 1

    fin_segment.save('test.GIF', transparency=0)
    segment.image = fin_segment
        
def final_draw(segments):
    fin_width = []
    fin_height = []

    for segment in segments:
        if segment.is_horizontal():
            fin_width.append(segment.length())
        else:
            fin_height.append(segment.length())

    fin_width = max(fin_width)+len(segments)*350
    fin_height = max(fin_height)+len(segments*350)
    fin_image = Image.new('L', (fin_width, fin_height), color=255)

    for segment in segments:
        if segment.is_horizontal():
            layer = segment.image
            fin_image.paste(layer, box=(segment.start[0]+350, segment.start[1]), mask=layer)
        else:
            layer = segment.image.rotate(-90)
            fin_image.paste(layer, box=(segment.start[0], segment.start[1]+350), mask=layer)

    fin_image.show()

if __name__ == '__main__':
    #line_rows = draw_horizontals('test_1.jpg')
    #print component_finder(line_rows,'hz_test_1.jpg', 'cp_test_1.jpg')
    
    # stashed changes
    [im, rows, cols, main_rows, main_cols] = draw_lines('intersection-test.jpg')
    # [im, rows, cols, main_rows, main_cols] = draw_lines('grid.jpg')
    # print intersections(main_rows, main_cols)
    segments = get_segments(im, rows, cols, main_rows, main_cols)

    segment = segments[1]
    line_rows = draw_horizontals_im(segment)
    all_comps = component_finder(line_rows, segment)
    # fin_segments = []
    # for s in segments:
    #     print 'Start point: ' + str(s.start)
    #     print 'End point: ' + str(s.end)
    #     print s.length()
    #     print s.is_horizontal()
    #     print '---'
    #     line_rows = draw_horizontals_im(s)
    #     all_comps = component_finder(line_rows, s)
    #     s.finding_components(['resistor', 'capacitor'])
    #     draw_segment2(s)
    #     fin_segments.append(s)

    # final_draw(fin_segments)



