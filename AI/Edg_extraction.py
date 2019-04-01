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


class Edge_extraction():
    """
    用于提取一张图片的边缘信息，
    mig：图片
    type：算子
    """

    def __init__(self, img, type=None, label_X=None, label_Y=None, mode=True, ** other):
        self.img = img
        self.type = type
        self.mode = mode
        if isinstance(self.type, str):  # 三种算法
            if self.type == 'Prewitt':
                label_X = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
                label_Y = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])
            elif self.type == 'Sobel':
                label_X = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
                label_Y = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])
            elif self.type == 'Laplace':
                label_X = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])
                label_Y = np.ones((3, 3))
            else:
                print("type should be a str in ['Prewitt','Sobel','Laplace']")
                raise SyntaxError
        # 自己输入卷积矩阵
        elif self.type == None and isinstance(label_X, list) and isinstance(label_Y, list):
            print('label_X and label_Y should be a list')
            raise SyntaxError
        else:
            raise SyntaxError
        self.label_X = label_X
        self.label_Y = label_Y

    def caculateing(self):
        """
        get the img to array
        """
        I = np.array(Image.open(self.img).convert('L'))
        I1 = signal.convolve2d(I, self.label_X, mode='same')
        I2 = signal.convolve2d(I1, self.label_Y, mode='same')

        return Image.fromarray(255 - I2) if self.mode == True else Image.fromarray(I2)

    def show(self):
        """
        to show a image
        """
        output = self.caculateing()
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(Image.open(self.img), cmap=cm.gray)
        plt.subplot(1, 2, 2)
        plt.imshow(output, cmap=cm.gray)
        plt.axis('off')
        plt.show()
