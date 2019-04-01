# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:14:57 2019

@author: xcledjwfr
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal

# 打开图片文件
imgfile = 'E:\\file\\IMG_4493.JPG'
# 关于convert的用法参见 https://blog.csdn.net/chris_pei/article/details/78261922
I = Image.open(imgfile).convert('L')

# 转化为矩阵
img = np.array(I)

f = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])
i = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])
# 进行卷积
img_new = signal.convolve2d(img, f, mode='same')
img_new2 = signal.convolve2d(img_new, i, mode='same')
output = Image.fromarray(255 - img_new2)

# 显示图片
plt.figure()
plt.subplot(2, 1, 1)
plt.axis('off')
plt.imshow(I, cmap=cm.gray)
plt.subplot(2, 1, 2)
plt.imshow(output, cmap=cm.gray)
plt.axis('off')
plt.show()
