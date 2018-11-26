Information-theoretical Measures
===

Kullback–Leibler divergence
---
KL divergence is a measure of how one probability distribution is different from a second, reference probability distribution.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a32176917e2304cf7c3a1e59220bf303d7f136c6)

Mutual Information
---
Definition:

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8da24e3338c5cadd04dd823feb3fbd85d95c611)

MI is the amount of information gained from observing another variable

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/209285ec1c887eaef3321b960b115857d8b1c099)

which is the entropy of `Y` minus the conditional entropy of `Y` given `X`.

Conditional Entropy
---
CE is the amount of uncertainty remaining about Y after X is known.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/c200b367c0f09c8d1faad3319c6c393d3ebbe539)

Variation of Information
---
VI is a measure of the distance between two clusterings. Unlike the mutual information, however, the variation of information is a true metric, in that it obeys the triangle inequality.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e87f2a8ba0eb9a98cab84243f14ac3298a7cd10f)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/VennDiagramIncludingVI.svg/1142px-VennDiagramIncludingVI.svg.png)
