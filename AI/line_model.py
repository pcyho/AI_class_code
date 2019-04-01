# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:17:58 2019

@author: xcledjwfr
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = [13854, 12213, 11009, 10655, 9503]
x = np.reshape(x, newshape=(5, 1)) / 10000.0
y = [21332, 20162, 19138, 18621, 18016]
y = np.reshape(y, newshape=(5, 1)) / 10000.0

lr = LinearRegression()
lr.fit(x, y)

print(lr.score(x, y))

y_hat = lr.predict(x)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(title='line_model', ylabel='y_hat', xlabel='x')

ax.scatter(x, y, label='data')
ax.plot(x, y_hat, label='result')
ax.legend()
plt.show()
