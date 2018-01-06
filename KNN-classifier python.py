# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:28:05 2017

@author: wrm
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import numpy as np

###Here is the calculation process for the nearest distances###
X1 = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=3).fit(X1)
result1= nbrs.kneighbors(X1)
print (result1)

X2=np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt=KDTree(X2,leaf_size=30,metric='euclidean')
result2=kdt.query(X2,k=3)
print (result2)

X3=np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
bal_tree=BallTree(X3,leaf_size=30)
result3=bal_tree.query(X3,k=3)
print (result3)

###From the sklearn package you could find that there are three different 
###algorithms that could be used to calculate the nearest neighbour. That is
###NearestNeighbors, KDTree and BallTree. From the three embedded function 
###you could find the the positions as well as the distances

from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print(clf.predict([[1.0, -1]]))


