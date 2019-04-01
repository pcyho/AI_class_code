import numpy as np
from scipy import linalg

np.arange(10)
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a = np.arange(10)
a**2
#array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81], dtype=int32)

b = np.array([[1, 2], [3, 4]])
# array([[1, 2],
#      [3, 4]])

linalg.det(b)
# -2.0
