# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:30:05 2014

@author: swalters
"""

import crop

def straighten(filename):
    [bw_pix, rows, cols] = crop.im_to_size_px(filename)
    a = crop.pix_to_array(bw_pix, rows, cols)
    
def find_darkest(a):
    dk_rows = []
    dk_cols = []
    (rows, cols) = a.shape
    for r in range(rows):
        dk_rows.append(255-darkness(a[r,:]))
    for c in range(cols):
        dk_cols.append(255-darkness(a[:,c]))
    return [dk_rows, dk_cols] # do stdistr stuff

def darkness(line):
    (rows, cols) = line.shape
    dkSum = 0
    for i in range(rows):
        for j in range(cols):
            dkSum += line[i,j]
    return dkSum/float(line.size)

if __name__ == '__main__':
    straighten('lines.png')