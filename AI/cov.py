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
imgfile = 'C:\\Users\\17269\\Desktop\\微信图片_20190421205547.jpg'
# 关于convert的用法参见 https://blog.csdn.net/chris_pei/article/details/78261922
I = Image.open(imgfile).convert('L')

# 转化为矩阵
img = np.array(I)
img_white = img > 161
img_black = img <= 161
img[img_black] = 0
img[img_white] = 255
output = Image.fromarray(img)

# 对侧面进行加权求和
row = 255 - np.mean(img, axis=1)
clum = 255 - np.mean(img, axis=0)

# 求取大于平均的最大值和最小值
row_size = row < np.mean(row)
size = np.array(range(0, len(row)))
size[row_size] = np.mean(row)
row_max = np.max(size)
row_min = np.min(size)
print(row_max, row_min)

# 显示图片
plt.figure()
plt.subplot(2, 2, 1)
plt.axis('off')
plt.imshow(I, cmap=cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(output, cmap=cm.gray)
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(Image.fromarray(img[row_min:row_max, :]), cmap=cm.gray)
plt.subplot(2, 2, 4)
plt.plot([x for x in range(len(clum))], np.ones([len(clum), 1]) * np.mean(clum))
plt.plot([x for x in range(len(clum))], clum)
plt.show()
