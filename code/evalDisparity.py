import matplotlib.pyplot as plt
from utils import imread
from depthFromStereo import depthFromStereo
import os

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

read_path = "../data/disparity/"
im_name1 = "teddy_im2.png" 
im_name2 = "teddy_im6.png"
#Read test images
img1 = imread(os.path.join(read_path, im_name1))
img2 = imread(os.path.join(read_path, im_name2))

#Compute depth
depth = depthFromStereo(img1, img2, 10)

#Show result
plt.imshow(depth, cmap = 'hot')
plt.show()
save_path = "../output/disparity/"
save_file = "teddy.png"
if not os.path.isdir(save_path):
    os.makedirs(save_path)
plt.imsave(os.path.join(save_path, save_file), depth)
