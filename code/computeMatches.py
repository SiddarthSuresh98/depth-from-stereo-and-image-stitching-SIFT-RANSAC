import numpy as np
from scipy.spatial.distance import cdist
from utils import showMatches
# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    # implement this
    matches = []
    threshold = 0.8
    for i in range(f1.shape[0]):
        #SSD between SIFT
        ssd = np.sum(np.power(f1[i,:]-f2,2), axis = 1)
        indices = np.argsort(ssd)
        ssd = np.sort(ssd)
        ratio = ssd[0]/ssd[1]
        #Only include matches if ssd score ratio lesser than threshold.
        if ratio < threshold:
            matches.append(indices[0])
        else:
            matches.append(-1)
    #print(np.shape(np.array(matches)))
    return np.array(matches)




