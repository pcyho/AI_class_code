# -*- coding: utf-8 -*-

import math
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import *


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        tmp = prob * math.log(prob, 2)
        shannonEnt -= tmp
    return shannonEnt


# 对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 返回划分点的值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
# DataSet=全部样本，axis=特征标号，value=每一个划分点的值
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:  # 若该样本连续参数大于样本划分点
                reducedFeatVec = featVec[:axis]  # 除去该连续特征的其他特征
                reducedFeatVec.extend(featVec[axis + 1:])  # 同上
                retDataSet.append(reducedFeatVec)  # 传出除去该连续特征的样本
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1  # 样本数据长度
    baseEntropy = calcShannonEnt(dataSet)  # 计算信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}  # 最佳划分点的列表
    for i in range(numFeatures):  # 遍历每一个特征
        featList = [example[i] for example in dataSet]  # 取出每一个训练的样本的对应特征
        # 对连续型特征进行处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':  # 取出密度（连续特征）
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)  # 对样本进行排序
            splitList = []  # 对数据样本划分后的集合
            for j in range(len(sortfeatList) - 1):  # j为样本编号
                # 两两做和除以2，作为划分点
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)
            bestSplitEntropy = 10000
            slen = len(splitList)  # 计算后的数据长度
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                value = splitList[j]  # 取出每一个划分点值
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(
                    dataSet, i, value, 0)  # 返回一个除去该连续特征的列表
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))  # 该列表占全部样本的比例
                newEntropy += prob0 * calcShannonEnt(subDataSet0)  # 计算新的划分点
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:  # 若划分结果小于预期
                    bestSplitEntropy = newEntropy
                    bestSplit = j  # 得到最佳划分点
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)  # 将该特征的值转化为集合
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:  # 取出每一个特征值
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


# 主程序，递归产生决策树
def createTree(dataSet, labels, data_full, labels_full):
    classList = [example[-1] for example in dataSet]  # 获取每个样本的密度
    if classList.count(classList[0]) == len(classList):  # 若样本全部属于同一类别
        return classList[0]  # 返回根节点
    if len(dataSet[0]) == 1:  # 若所有属性都一样
        return majorityCnt(classList)  # 返回该属性最多的
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)  # 选择生成子节点
    bestFeatLabel = labels[bestFeat]  # 导出最优划分标签
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]  # 根据划分点选出数据
    uniqueVals = set(featValues)  # 创建选出数据的集合
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])  # 找出选出数据特征的下标
        featValuesFull = [example[currentlabel]
                          for example in data_full]  # 创建选出数据特征的列表
        uniqueValsFull = set(featValuesFull)  # 转化为集合
    del (labels[bestFeat])  # 删除该标签
    # 针对bestfeat的每个取值划分子树
    for value in uniqueVals:  # 对选出的每个数据的集合遍历
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)  # 删除已经筛选过的特征
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
            dataSet, bestFeat, value), subLabels, data_full, labels_full)  # 迭代筛选最优划分离散变量
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)  # 生成新节点
    return myTree


#################################################################################################
# 生成决策树的代码
df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:].tolist()  # 训练集列表
data_full = data[:]  # 副本？？
labels = df.columns.values[1:-1].tolist()  # 每个训练集的标签属性
labels_full = labels[:]
myTree = createTree(data, labels, data_full, labels_full)
print(myTree)
###################################################################################################
# 画图代码
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center",
                            bbox=nodeType, arrowprops=arrow_args)


# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) /
              2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff,
                                       plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


createPlot(myTree)
