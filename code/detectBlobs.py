import numpy as np
import cv2
import scipy
from skimage.color import rgb2gray
from scipy import ndimage
import matplotlib.pyplot as plt

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

#show grayscale image
def showGrayScaleImage(im):
    plt.imshow(im[:,:], cmap='gray')
    plt.show()

#Find maximal response from LoG filter across scales
def find_max_scale(img_height, img_width, scale_maximal_response): 
    max_scale = np.zeros(scale_maximal_response.shape)
    for i in range(img_height): 
        for j in range(img_width):
           index = np.unravel_index(np.argmax(scale_maximal_response[i,j,:]), scale_maximal_response[i,j,:].shape)
           max_scale[i,j,index[0]] =  scale_maximal_response[i,j,index[0]]
    return max_scale

def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image
    im = rgb2gray(im)
    im = im.astype(np.float64)
    img_height, img_width = im.shape
    
    #create scale space
    scale_space = []
    sigma = 2
    k = 1.1
    n = 10
    for i in range(n):
        gaussian_laplace_im = ndimage.gaussian_laplace(input=im, sigma=sigma, mode='constant', cval=0)
        gaussian_laplace_im = np.power(sigma,2) * gaussian_laplace_im
        #showGrayScaleImage(gaussian_laplace_im)
        scale_space.append(gaussian_laplace_im)
        sigma = k* sigma
    #Reformatting
    scale_space = np.transpose(scale_space, (1,2,0))
    scale_maximal_response = np.zeros((img_height,img_width,n))

    #non maximal suppression at each pixel
    for i in range(n):
       scale_maximal_response[:,:,i] = ndimage.maximum_filter(scale_space[:,:,i],size = 3, mode = 'constant', cval = 0)

    #get maximal response across scales.
    scale_maximal_response = find_max_scale(img_height,img_width,scale_maximal_response)
    scale_maximal_response = np.multiply(scale_maximal_response, (scale_maximal_response == scale_space))   

    #threshold to detect blobs
    threshold = 0.02
    circles = []
    
    #Calculate radii of circles from maximal response from LoG filter.
    for i in range(n):
        index = np.argwhere(scale_maximal_response[:,:,i]>threshold)
        radius = np.power(2, 0.5) * sigma * np.power(k , i)
        circles.append((i,index,radius))
    
    #reformat for sending back output in desired format.
    blobs = []
    for item in circles:
        indices = item[1]
        radius = item[2]
        for blob in indices:
            score = scale_maximal_response[blob[0],blob[1],item[0]]
            #only include blobs if score is greater than zero.
            if score > 0:
                blobs.append([blob[1],blob[0],radius,0,score])
    
    print(np.shape(np.array(blobs)))
    return np.array(blobs)