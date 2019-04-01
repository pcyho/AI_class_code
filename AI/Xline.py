# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:38:02 2019

@author: xcledjwfr
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

x = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(x, y)
fig = plt.figure()
ax = Axes3D(fig)
x1 = [6, 8, 10, 14, 18]
y1 = [2, 1, 0, 2, 0]
z1 = model.predict(x)

ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

print(model.score(x, y))
