import os
import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from detectBlobs import detectBlobs
from drawBlobs import drawBlobs

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

# Evaluation code for blob detection
# Your goal is to implement scale space blob detection using LoG  

imageName = 'sunflowers.jpg'
numBlobsToDraw = 1000
imName = imageName.split('.')[0]

datadir = os.path.join('..', 'data', 'blobs')
im = imread(os.path.join(datadir, imageName))


blobs = detectBlobs(im)  # dummy placeholder

drawBlobs(im, blobs, numBlobsToDraw)

