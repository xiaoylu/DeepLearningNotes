Short Python Code for KNN and k-means
===

K-nearest-neighbors
```
# knn implementation 
import numpy
import collections

def knn(X, y, x0, k):
    # get labels of the k nearest neighbors
    distance = sorted([(numpy.linalg.norm(X[i,:] - x0), y[i]) \
                for i in range(X.shape[0]) \
                ])[:k]
    # vote for the prediction
    cnt = collections.Counter([v for dist, v in distance])
    return 0 if cnt[0] > cnt[1] else 1
    
X = numpy.random.randn(10, 3)
y = numpy.random.randint(0, 2, size=10)
x0 = numpy.array([0.1, -0.1, 0.5])

knn(X, y, x0, 5)
```

K-means
```
import numpy.linalg as LA
import numpy as np

def kmeans(X, k):
	
    # Initialize centroids randomly
    c = np.random.randn(3, k)
    tmp = np.copy(c)
    nIter = 0
    
    while nIter < 100:
        tmp = np.copy(c)
        
        clusters = np.zeros((3, k))
        n = np.zeros((1, k))
        # Assign labels to each datapoint based on centroids
        for i in range(X.shape[1]):
            j = np.argmin(LA.norm(X[:,i:i+1] - c, axis = 0))
            clusters[:, j] += X[:, i] 
            n[0, j] += 1
        
        # update centroids (raise error if emtpy cluster, n[0,j]==0?)
        c = clusters / n
        
        # if convergence, break
        if LA.norm(tmp - c) < 1e-3:
            break
        
        nIter += 1
        
    print "Centroid at (per row)"
    print c.T
    return c
    
# 20 samples, represented as dimension-3 col vectors
# two clusters with multivariate normal distribution at (0, 1) and (2, 1)
X = np.hstack((np.random.randn(3, 10), np.random.randn(3, 10) + 2))

kmeans(X, 2)
```
