Short Python Code for KNN and k-means
===

K-nearest-neighbors
```
# knn implementation 
import numpy
import collections

def knn(X, y, x0, k):
    distance = sorted([(numpy.linalg.norm(X[i,:] - x0), y[i]) \
                for i in range(X.shape[0]) \
                ])[:k]
    cnt = collections.Counter([v for dist, v in distance])
    return 0 if cnt[0] > cnt[1] else 1
    
X = numpy.random.randn(10, 3)
y = numpy.random.randint(0, 2, size=10)
x0 = numpy.array([0.1, -0.1, 0.5])

knn(X, y, x0, 5)
```

K-means
```
```
