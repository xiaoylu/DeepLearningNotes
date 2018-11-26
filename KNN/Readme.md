K Nearest Neighbors
===

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

In EM, you randomly initialize your model parameters, then you alternate between (E) assigning values to hidden variables, based on parameters and (M) computing parameters based on fully observed data.

E-Step: Coming up with values to hidden variables, based on parameters. If you work out the math of chosing the best values for the class variable based on the features of a given piece of data in your data set, it comes out to "for each data-point, chose the centroid that it is closest to, by euclidean distance, and assign that centroid's label." The proof of this is within your grasp! See lecture.

M-Step: Coming up with parameters, based on full assignments. If you work out the math of chosing the best parameter values based on the features of a given piece of data in your data set, it comes out to "take the mean of all the data-points that were labeled as c."
