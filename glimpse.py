import numpy as np

def takeGlismple(input, heightStart, widthStart, size):
    height = len(input)-size
    width = len(input[0])-size
    rowStart = int(heightStart*height)
    colStart = int(widthStart*width)
    return[ i[colStart:colStart+size] for i in input[rowStart:rowStart+size]]