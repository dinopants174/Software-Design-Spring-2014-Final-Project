# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:53:26 2014

@author: swalters
"""

import ImageCropper
import numpy

################################################################################################

'''METHODS AND CLASSES: (** denotes main functionality)
    :: line finding
        -> find_horizontals (find main dark rows in an image)
        -> find_lines (find main dark rows and cols in an image)
        -> remove_similar (reduce list by averaging groups of similar numbers)
    
    :: segmentation
        -> **get_segments (crops image into segments defined by main dark rows and cols)
        Segment class -> stores start point and end point and associated image; eventually also list of components
    
    :: component finding
        -> **component_finder (locates components in a segment's image)
'''

###############################################################################################

''' ------ LINE FINDING ------ '''
def find_horizontals(im, t=0.8):
    ''' identifies strongest horizontals in an image
        input: image object, (optional threshold between 0 and 1)
        output: list of dark rows
    '''
    # get image array
    a = ImageCropper.pix_to_array(im.load())
    (rows, cols) = a.shape
    
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < t*cols: # hard-coded, as of now
            line_rows.append(r)

    # return list of single-pixel dark horizontals (hopefully with length 1, for just 1 segment)
    main_rows = remove_similar(line_rows, cols)
    return main_rows  
    
def find_lines(im, t=0.7):
    ''' identifies strongest horizontals and verticals in an image
        input: image object, (optional threshold between 0 and 1)
        output: list of lists: [dark rows, dark cols]
    '''
    (cols, rows) = im.size
    a = ImageCropper.pix_to_array(im.load(), rows, cols)
    
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < t*cols: # less than (t*100)% of the row is light
            line_rows.append(r)
    main_rows = remove_similar(line_rows, cols) # pick out 1px-wide line for each dark horizontal
            
    # loop through columns checking for darkness
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < t*rows: # less than (t*100)% of the col is light
            line_cols.append(c)
    main_cols = remove_similar(line_cols, rows) # again, 1px-wide line for each grouping

    # return list of lists: single-pixel dark horizontals and single-pixel dark verticals
    return [main_rows, main_cols]
    
def remove_similar(items, max_length):
    ''' reduces similar groups of items in a list to averages of the members in the group
        inputs: list of items, maximum possible list length (width or height of image)
        output: reduced list
    '''
    main_items = []
    this_item = []
    
    for i in items:
        if len(this_item) == 0 or (i - this_item[-1]) <= 0.05*max_length: # adding to new group
            this_item.append(i)
        else: # group is done; add it to list of group averages and clear it
            main_items.append(sum(this_item)/len(this_item))
            this_item = []

    if len(this_item) != 0: # catch last group
        main_items.append(sum(this_item)/len(this_item))
        
    return main_items
    
    
''' ------ SEGMENTATION ------ '''  
def get_segments(im):    
    ''' identifies dark rows and dark cols in image using find_lines, then crops all segments (defined as lines between intersections) out
        input: image object
        output: list of segment objects
        example: +---+---+  contains 7 segments - 4 horizontal and 3 vertical.
                 |   |   |
                 +---+---+
    '''
    # identify dark rows and dark cols to segment around
    [main_rows, main_cols] = find_lines(im) 
    
    segments = []
    (cols, rows) = im.size
    
    # move across each row cropping between dark column intersections
    for row in main_rows:
        # top and bottom boundaries consistent across row; dependent on image size
        t = row-int(0.05*rows)
        b = row+int(0.05*rows)    
        
        # left and right boundaries defined by one column intersection and the next
        for i in range(len(main_cols)-1):
            l = main_cols[i]
            r = main_cols[i+1]
            box = (int(l+0.025*rows), t, int(r-0.025*rows), b) # 0.025 crops off ends - not clean b/c of intersecting lines
            cropped = im.crop(box)
            segments.append(Segment(cropped, (row, l), (row, r))) # Segment object stores endpoints
    
    # move down each column cropping between dark row intersections
    for col in main_cols:
        # left and right boundaries consistent down column
        l = col-int(0.05*cols)
        r = col+int(0.05*cols)
        
        # top and bottom boundaries defined by one row intersection and the next
        for i in range(len(main_rows)-1):
            t = main_rows[i]
            b = main_rows[i+1]
            box = (l, int(t+0.025*cols), r, int(b-0.025*cols)) # 0.025 crops off ends, again
            cropped = im.crop(box)
            cropped = cropped.rotate(90) # make vertical image horizontal for upcoming component finding
            segments.append(Segment(cropped, (t, col), (b, col))) # again, Segment object stores endpoints
    
    return segments

class Segment():
    ''' initialized with start point, end point, and associated hand-drawn b/w segment image;
        will later add list of components and modify image to cleanly generated segment
    '''
    def __init__(self, image, start, end):
        self.image = image
        self.start = start
        self.end = end
    
    def is_horizontal(self):
        return self.start[0] == self.end[0]

    def length(self):
        if self.is_horizontal():
            return self.end[1]-self.start[1]
        else:
            return self.end[0]-self.start[0]

    def finding_components(self, component_id_list):
        self.component_id_list = component_id_list


''' ------ COMPONENT FINDING ------ '''
def component_finder(segment):
    ''' locates components in a segment's image
        input: segment object
        output: list of component images
    '''
    # set up image, dark row(s), width/height, pixel object
    im = segment.image
    line = find_horizontals(im)[0]
    (width, height) = im.size
    bw_pix = im.load()
        
    # make offset lines
    offset = 0.4*height
    above = int(line - offset)
    below = int(line + offset)

    # find deviations
    non_white = []
    for i in range(width):
        if bw_pix[i,above] < 25 or bw_pix[i,below] < 25:
            non_white.append(i)
    
    # find components using deviations
    component = []
    all_components = []
    component_counter = 0
    for i in range(len(non_white)-1):
        # have found end of component - add to list of all components and clear component list
        if non_white[i+1] - non_white[i] > 0.2*width:  #0.5*width  # assumption about component spacing
            component_counter += 1
            component.append(non_white[i])
            all_components.append(component)
            component = [] 
        # not the end of the component yet - keep going
        else:
            component.append(non_white[i])

    # if last leftover component isn't empty (won't have been added b/c array hasn't had a chance to clear), add it to all components too
    if len(component) != 0:
        all_components.append(component)

    all_comps_cropped = []
    
    # name components and crop them out
    for i in range(len(all_components)):
        # define bounding box
        l = int(all_components[i][0] - 0.02*width)
        r = int(all_components[i][len(all_components[i])-1]+0.02*width)
        t = int(above-0.15*height)
        b = int(below+0.15*height)
        box = (l, t, r, b)
        
        region = im.crop(box)
        all_comps_cropped.append(region)

    return all_comps_cropped