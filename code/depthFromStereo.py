# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm

#Function to show grayscale image
def showGrayScaleImage(im):
    plt.imshow(im[:,:], cmap='gray')
    plt.show()

#Function to calculate abolute difference as distance measure
def getAD(img1, img2):
    return np.sum(np.abs(img1-img2))

#As the images are parallel, consider only equipolar windows from the two images to calculate minimum absolute difference
def compareEquipolarPatches(idx, img1Patch, img2, ws):
    x = idx[0]
    minD = 9999999
    minY = 9999999
    for j in range(img2.shape[1]-ws+1):
        patch_right = img2[x:x+ws, j:j+ws]
        ad = getAD(img1Patch, patch_right)
        if ad < minD:
            minD = ad
            minY = j
    return minY

def depthFromStereo(img1, img2, ws):
    gs_img1 = rgb2gray(img1)
    gs_img2 = rgb2gray(img2)
    img_height, img_width = gs_img1.shape
    disparity_map = np.zeros((img_height, img_width))
    #Generate windows of size ws
    for x in tqdm(range(ws, img_height-ws+1)):
        for y in range(ws, img_width-ws+1):
            im1Patch = gs_img1[x:x + ws, y:y + ws]
            #get index of minimum distance
            min_index = compareEquipolarPatches((x,y), im1Patch, gs_img2, ws)
            #compute disparity
            disparity_map[x, y] = abs(min_index - y)
    return disparity_map