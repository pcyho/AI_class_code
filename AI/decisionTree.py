# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CreateTree():
    def caculateHash(self, DataSet):
        """
        用于计算每个属性的信息熵，返回信息熵的值
        """
        numEntries = len(DataSet)
        labelList = {}
        for featVec in DataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelList.keys():
                labelList[currentLabel] = 0
            labelList[currentLabel] += 1
        shannonEnt = 0.0

        for key in labelList:
            prob = float(labelList[key]) / numEntries
            tmp = prob * math.log(prob, 2)
            shannonEnt -= tmp
        return shannonEnt

    def chooseBestFeatureToSplit(dataSet, labels):
        """
        计算最优节点划分方式
        """
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = caculateHash(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        bestSplitDict = {}

    def createTree(self, dataSet, labels, data_full, labels_full):
        """
        循环递归产生决策树
        """
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return majorityCnt(classList)
        bestFeat = chooseBestFeatureToSplit(dataSet, labels)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            """
            以字符串定义离散变量
            """
            currentLabel = labels_full.index(labels[bestFeat])
            featValuesFull = [example[currentLabel] for example in data_full]
            uniqueValsFull = set(featValuesFull)
        del(labels[bestFeat])  # 当对该节点划分完之后删除该节点
        for value in uniqueVals:
            subLabels = labels[:]
            if type(dataSet[0][bestFeat]).__name__ == 'str':
                uniqueValsFull.remove(value)
            myTree[bestFeatLabel][value] = createTree(splitDataSet(
                dataSet, bestFeat, value), subLabels, data_full, labels_full)
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            for value in uniqueValsFull:
                myTree[bestFeatLabel][value] = majorityCnt(classList)
        return myTree

    def __init__(self):
        """
        构造函数，用于开始程序
        """
        df = pd.read_csv('watermelon_4_3.csv')
        data = df.values[:, 1:]
        data_full = data[:]
        labels = df.columns.values[1:-1].tolist()
        labels_full = labels[:]
        myTree = CreateTree(data, labels, data_full, labels_full)
