import numpy as np
import cv2
import time

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def findAffineTransformation(source_xys, dest_xys):
    A = []
    B = []
    A.clear()
    B.clear()
    for j in range(len(source_xys)):
        s_x,s_y = source_xys[j]
        d_x,d_y = dest_xys[j]
        A.append([s_x,s_y,0,0,1,0])
        A.append([0,0,s_x,s_y,0,1])
        B.append(d_x)
        B.append(d_y)
    A = np.array(A)
    B = np.array(B)
    #least squares solution for AX=B
    X = np.linalg.lstsq(A,B)
    #print(X)
    return X[0]
    

def getInliers(source_xys, dest_xys, affineTransformation):
    #threshold for inlier selection
    threshold = 10
    inliers = []
    inliers.clear()
    #print(affineTransformation.shape)
    for i in range(len(source_xys)):
        A = []
        A.clear()
        s_x,s_y = source_xys[i]
        A.append([s_x,s_y,0,0,1,0])
        A.append([0,0,s_x,s_y,0,1])
        #find transformed coordinates
        t_x,t_y = np.dot(A , affineTransformation)
        d_x,d_y = dest_xys[i]
        #distance between transformed coordinates and destination coordinates.
        dist = np.power((np.power((d_x-t_x),2) + np.power((d_y-t_y),2)), 0.5)
        if(dist<threshold):
            inliers.append((s_x,s_y,d_x,d_y,i))
        #print(dist) 
        #print(t_x,t_y)
    #print(inliers)
    return inliers



def ransac(matches, blobs1, blobs2):
    num_iterations = 20
    max_inliers = 0
    T = np.zeros(6,)
    final_affine_transformation = np.zeros((2,3))
    inliers_idx = []
    for i in range(num_iterations):
        source_xys = []
        dest_xys = []
        source_xys.clear()
        dest_xys.clear()
        #3 is the minimal number of points required.
        while len(source_xys)<3:
            random_match = np.random.randint(0,len(matches))
           # print(random_match)
            #print(matches[random_match])
            if (matches[random_match] != -1):
                source_xys.append((blobs1[random_match][0], blobs1[random_match][1]))
                dest_xys.append((blobs2[matches[random_match]][0],blobs2[matches[random_match]][1]))
        #print("Source:", source_xys)
        #print("Destination:", dest_xy)
        
        #Find the affine transformation
        X = findAffineTransformation(source_xys,dest_xys)


        #find inliers
        source_xys.clear()
        dest_xys.clear()
        for j in range(len(matches)):
            if(matches[j] != -1):
                source_xys.append((blobs1[j][0], blobs1[j][1]))
                dest_xys.append((blobs2[matches[j]][0],blobs2[matches[j]][1]))
        
        #get inliers
        inliers = getInliers(source_xys,dest_xys, X)

        #print(len(inliers))

        #find best affine transformation iteratively based on max inliers count
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            source_xys.clear()
            dest_xys.clear()
            inliers_idx.clear()
            for i in inliers:
                source_xys.append((i[0], i[1]))
                dest_xys.append((i[2],i[3]))
                inliers_idx.append(i[4])
            #find affine transformation for all inliers
            T = findAffineTransformation(dest_xys,source_xys)

        #Reshape affine transformation from least squares solution
        final_affine_transformation[0][0] = T[0]
        final_affine_transformation[0][1] = T[1]
        final_affine_transformation[0][2] = T[4]
        final_affine_transformation[1][0] = T[2]
        final_affine_transformation[1][1] = T[3]
        final_affine_transformation[1][2] = T[5]

    # print(inliers_idx)
    print(final_affine_transformation)

    return inliers_idx,final_affine_transformation