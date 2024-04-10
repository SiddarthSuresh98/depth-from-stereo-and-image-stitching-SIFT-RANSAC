import numpy as np
import skimage.io as sio
import os
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def second_order_statistics(ix, iy, ws):
    img_height,img_width = image.shape
    #Precomputing different values required in second moment matrix. 
    A = np.power(ix,2)
    B = np.power(iy,2)
    C = ix*iy
    R = np.zeros((image.shape))
    k = 0.03

    for i in range(img_height):
        for j in range(img_width):
            #Generating window with current pixel at center
            top = max(0,i-ws//2)
            bottom = min(img_height, i+ws//2)
            left = max(0, j-ws//2)
            right = min(img_width,j+ws//2)
            #Generating second moment matrix for the patch
            patch_A = np.sum(A[top:bottom,left:right].flatten())
            patch_B = np.sum(B[top:bottom,left:right].flatten())
            patch_C = np.sum(C[top:bottom,left:right].flatten())
            M = np.zeros((2,2))
            M[0,0] = patch_A
            M[0,1] = patch_C
            M[1,0] = patch_C
            M[1,1] = patch_B
            #Calulating R value for the patch
            R[i,j] = np.linalg.det(M) - k*(np.power(np.matrix.trace(M),2))
    return R 

# read image
im_dir = "../data/disparity"
image_file = "cones_im2.png"
image = sio.imread(os.path.join(im_dir, image_file))
# convert image to gray
image = rgb2gray(image)

# compute differentiations along x and y axis respectively
# x-diff
#--------- add your code here ------------------#
#sobel filter used to generate differentiations
Ix = ndimage.sobel(image, 0)

# y-diff
#--------- add your code here ------------------#
#sobel filter used to generate differentiations
Iy = ndimage.sobel(image, 1)

# set window size
#--------- modify this accordingly ------------------#
ws = 5

heatMapImg = second_order_statistics(Ix, Iy, ws)

plt.imshow(heatMapImg)
plt.colorbar()
savedir = "../output/harris/"
savefile = "cones_im2.png"
if not os.path.isdir(savedir):
    os.makedirs(savedir)
plt.imsave(os.path.join(savedir, 'ws-'+str(ws)+'_'+savefile), heatMapImg)
plt.show()


# save
savedir = "../output/harris/"
savefile = "cone_im2.jpg"

if not os.path.isdir(savedir):
    os.makedirs(savedir)
plt.imsave(os.path.join(savedir, 'ws-'+str(ws)+'_'+savefile), heatMapImg)
