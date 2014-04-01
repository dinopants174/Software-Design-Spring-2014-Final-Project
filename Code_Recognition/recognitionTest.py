# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:50:07 2014

@author: swalters
"""

#http://stackoverflow.com/questions/18777873/convert-rgb-to-black-or-white

import Image

from pytesser import *
im = Image.open('handwriting.jpg')
gray = im.convert('L')
bw = gray.point(lambda x: 0 if x<135 else 255, '1')
bw.save('result.png')
text = image_to_string(bw)
print text