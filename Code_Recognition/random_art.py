# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:59:22 2014

@author: zoherghadyali
"""

from random import randint
from math import *
import Image

def build_random_function(min_depth, max_depth):
    """ This function creates a random function using a list of strings that
    provide a description of what the mathematical function would do. Example:
    'sin_pi'['x'] should evaluate to sin(pi*x). Function takes as input the max
    and min depth, randomly generates a depth in that range, and recursively
    calls the function until dpeth is 1. """
    
    l = ["x","y"]
    funcs = ["sin_pi", "cos_pi", "square", "prod", "avg"]
    depth = randint(min_depth, max_depth)
    if depth == 1:
        return [l[randint(0,1)]]    #base case is depth is 1, returns either "x" or "y"
    elif depth>1:
        rand_func = funcs[randint(0,4)] #only the first three functions in the funcs list require one "x" or "y"
        if rand_func in funcs[0:3]:
            return [rand_func, build_random_function(depth-1, depth-1)] #so if rand_func is one of these functions, recursion calls function once
        elif rand_func in funcs[3:]: #the second two functions in the funcs list require two parameters
            return [rand_func, build_random_function(depth-1, depth-1), build_random_function(depth-1, depth-1)] #so if rand_func is one of these functions, calls function twice


def evaluate_random_function(f, x, y):
    """ This functions takes as input the randomly generated function, which at this
    point is just a nested list of lists, and x and y values to evaluate the function
    with. The function outputs a scalar that is the evaluation of the function. It scans
    the list of lists to find one of the functions, then recursively evaluates the embedded
    list until it gets to base case. """

    if f[0] == "x":
        return x
    elif f[0] == "y":
        return y        #this is base case, keeps delving into list until f[0]=="x" at which point it returns x
    elif f[0] == "sin_pi":
        return sin(pi*evaluate_random_function(f[1], x, y)) #if first element in list is sin_pi, then calls function to evaluate the next index in list
    elif f[0] == "cos_pi":
        return cos(pi*evaluate_random_function(f[1], x, y))
    elif f[0] == "square":
        return evaluate_random_function(f[1], x, y)**2  #if first element in list is prod or avg, need two parameters, calls function to evaluate the next index and index after that
    elif f[0] == "prod":
        return evaluate_random_function(f[1], x, y) * evaluate_random_function(f[2], x, y)
    elif f[0] == "avg":
        return (evaluate_random_function(f[1], x, y) + evaluate_random_function(f[2], x, y))/2.0


def remap_interval(val, input_interval_start, input_interval_end, output_interval_start, output_interval_end):
    """ Maps the input value that is in the interval [input_interval_start, input_interval_end]
        to the output interval [output_interval_start, output_interval_end].  The mapping
        is an affine one (i.e. output = input*c + b). This function uses equation of form    
                  (b-a)(x - min)
          f(x) =  --------------  + a
                    max - min
        where a is the min and b is the max of the output interval, min and max are the min
        and max of input interval, and x is the value """
    
    num = float((output_interval_end - output_interval_start)*(val - input_interval_start))
    denom = input_interval_end - input_interval_start    
    res = num/denom + output_interval_start
    return res    
    

def create_image():
    """ This function takes no input but should display an image that maps the red,
    green, and blue randomly generated equations for each pixel. So three equations
    are built for each channel. Then for each x value, function loops through all the
    y-values and plots each (x,y) pixel pair as a function of red, blue, and green """
    
    red_ch = build_random_function(1,3) #produces three randomly generated equations
    green_ch = build_random_function(5,7)
    blue_ch = build_random_function(9,12)
    
    im = Image.new("RGB", (700,700)) #loads an image of 350 x 350 px
    pixels = im.load()  #makes img into pixel map so can plot each pixel
    
    for x in range(700):    #loops through every x-value from 0 to 349
        x_val = remap_interval(x, 0, 699, -1, 1)    #remaps all x-values to [-1,1]
        for y in range(700):    #loops through every y-value from 0 to 349 for each x-value
            y_val = remap_interval(y, 0, 699, -1, 1)    #remaps all y-values to [-1,1]
            red_val = evaluate_random_function(red_ch, x_val, y_val)    #evaluates each random function for each x and y
            green_val = evaluate_random_function(green_ch, x_val, y_val)
            blue_val = evaluate_random_function(blue_ch, x_val, y_val)
            red = remap_interval(red_val, -1, 1, 0, 255)    #remaps to [0, 255] for RGB scale
            green = remap_interval(green_val, -1, 1, 0, 255)
            blue = remap_interval(blue_val, -1, 1, 0, 255)
            pixels[int(x),int(y)] = (int(red), int(green), int(blue))   #maps each pixel by its red, blue, and green values
    im.show()   #shows image
            
if __name__ == '__main__':
#    f = build_random_function(1,3)
#    print f
#    print evaluate_random_function(f,5,5)
#    print remap_interval(75, 50, 100, 0, 1)
    create_image()