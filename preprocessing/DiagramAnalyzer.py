# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:53:26 2014

@author: swalters
"""

import ImageCropper
import numpy
from PIL import Image

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
    (cols, rows) = im.size
    a = ImageCropper.pix_to_array(im.load(), rows, cols)
    
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < t*cols: # hard-coded, as of now
            line_rows.append(r)

    # return list of lists - [center, distribution] for each located horizontal
    row_analysis = remove_similar(line_rows, cols)
    return row_analysis  
    
def find_lines(im, t=0.8):
    ''' identifies strongest horizontals and verticals in an image
        input: image object, (optional threshold between 0 and 1)
        output: list of lists: [dark rows, dark cols]
    '''
    (cols, rows) = im.size
    a = ImageCropper.pix_to_array(im.load(), rows, cols)
    
    ## ROWS
    # loop through rows checking for darkness
    line_rows = []
    for r in range(rows):
        val = numpy.sum(a[r,:])
        if val < t*cols: # less than (t*100)% of the row is light
            line_rows.append(r)
            
    # get just centers from row_analysis - distributions useful for find_horizontals, but not here.
    row_analysis = remove_similar(line_rows, cols) # pick out centers and distributions for each dark horizontal
    main_rows = []
    for item in row_analysis:
        main_rows.append(item[0]) 
        
    ## COLS
    # loop through columns checking for darkness
    line_cols = []
    for c in range(cols):
        val = numpy.sum(a[:,c])
        if val < t*rows: # less than (t*100)% of the col is light
            line_cols.append(c)
            
    # get just centers from col_analysis - distributions useful for find_horizontals, but not here.
    col_analysis = remove_similar(line_cols, rows)
    main_cols = []
    for item in col_analysis:
        main_cols.append(item[0])

    ## RETURN list of lists: single-pixel dark horizontals and single-pixel dark verticals
    return [main_rows, main_cols]
    
def remove_similar(items, max_length):
    ''' reduces similar groups of items in a list to averages of the members in the group
        inputs: list of items, maximum possible list length (width or height of image)
        output: reduced list of lists -> [center, distribution] where distribution is 1/2 of range
        example: [60, 61, 62, 63, 64, ..., 99, 100, 400, 401, 402, 403, ..., 439, 440]
            gets separated into [[60, 61, 62, 63, 64..., 99, 100], [400, 401, 402, 403, ..., 439, 440]]
            such that [[80, 20], [420,20]] is returned 
                -> 80 and 420 are averages of [60...100] and [400...440] sublist
                -> (100-60)/2 = 20 and (440-400)/2 = 20
    '''
    main_items = []
    this_item = []
    
    for i in items:
        if len(this_item) == 0 or (i - this_item[-1]) <= 0.05*max_length: # adding to new group
            this_item.append(i)
        else: # group is done; add it to list of group averages and clear it
            avg = sum(this_item)/len(this_item)
            distr = this_item[-1]-this_item[0]/2
            main_items.append([avg, distr])
            this_item = []

    if len(this_item) != 0: # catch last group
        avg = sum(this_item)/len(this_item)
        distr = this_item[-1]-this_item[0]/2
        main_items.append([avg, distr])
        
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
            print cropped
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
    
    
''' ------ DRAWING ------ '''
def draw_segment(segment):
    width = 700
    
    # 
    fin_segment = Image.new('RGBA', (segment.length(), width))
    line = Image.open('line.png').convert('RGBA')
    
    tiles = segment.length()/width
    current_tile = 0
    
    for i in range(tiles):
        fin_segment.paste(line, box=(current_tile*width, 0), mask=line)
        current_tile += 1
    fin_segment.paste(line, box=(segment.length()-width, 0), mask=line)

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

    segments = get_segments(Image.open('TestImages/intersection-test.jpg'))
    print segments