# -*- coding:utf-8 -*-
# !/usr/bin/env python3
import operator
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class DecisionTree(object):
    def __init__(self):
        pass

    def createDataSet(self):
        ''' 创建数据集

        Paraments:

        Returns:
            dataSet                 -   生成的数据集
            labels                  -   数据集对应的标签
        Modify:
            2018-07-23
        Authon:
            Li Shan
        '''
        dataSet = [[0, 0, 0, 0, 'no'],
                   [0, 0, 0, 1, 'no'],
                   [0, 1, 0, 1, 'yes'],
                   [0, 1, 1, 0, 'yes'],
                   [0, 0, 0, 0, 'no'],
                   [1, 0, 0, 0, 'no'],
                   [1, 0, 0, 1, 'no'],
                   [1, 1, 1, 1, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [2, 0, 1, 2, 'yes'],
                   [2, 0, 1, 1, 'yes'],
                   [2, 1, 0, 1, 'yes'],
                   [2, 1, 0, 2, 'yes'],
                   [2, 0, 0, 0, 'no']]
        labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
        return dataSet, labels

    def splitDataSet(self, dataSet, index, feature):
        ''' 分割数据集
        Paraments:
            dataSet                         -   输入待分割的数据集
        Returns:
            subDataSet                      -   分割成功的数据集
        Modify:
            2018-07-23
         Author:
            Li Shan
        '''
        subDataSet = []
        for data in dataSet:
            if data[index] == feature:
                d = data[:index] + data[index+1:]
                subDataSet.append(d)
        return subDataSet

    def chooseBestFeature(self, dataSet):
        ''' 选择最优特征

        Paraments:
            dataSet                     -   输入的数据集
        Returns:
            bestFeature                 -   最优特征的索引值
        Modify:
            2018-0723
        Author:
            Li Shan
        '''
        featureCount = len(dataSet[0]) - 1
        baseShannonEntropy = self.calcShannonEntropy(dataSet)
        infoGain = {}
        for featureIndex in range(featureCount):
            featureLists = [example[featureIndex] for example in dataSet]
            featureUnique = set(featureLists)
            conditionEntropy = 0.0
            for feature in featureUnique:
                subDataSet = self.splitDataSet(dataSet, featureIndex, feature)
                prob = len(subDataSet) / len(featureLists)
                conditionEntropy += prob * self.calcShannonEntropy(subDataSet)
            infoGain[featureIndex] = baseShannonEntropy - conditionEntropy
        infoGain = sorted(infoGain.items(), key=operator.itemgetter(1), reverse=True)
        return infoGain[0][0]

    def calcShannonEntropy(self, dataSet):
        ''' 计算经验熵

        Paraments:
            dataSet                     -   输入的数据集
        Returns:

        Modefy:
            2018-07-23
        Authon:
            Li Shan
        '''
        resultClass = {}
        resultCount = len(dataSet)
        shannonEntropy = 0.0
        for data in dataSet:
            label = data[-1]
            resultClass[label] = resultClass.get(label, 0) + 1
        for k,v in resultClass.items():
            prob = v / resultCount
            shannonEntropy -= prob * math.log(prob, 2)
        return shannonEntropy

    def createTree(self, dataSet, labels):
        bestFeature = self.chooseBestFeature(dataSet)
        key = labels[bestFeature]
        myTree = {key:{}}

        print(myTree)

if __name__ == '__main__':
    dt = DecisionTree()
    dataSet, labels = dt.createDataSet()
    bestFeature = dt.chooseBestFeature(dataSet)
    myTree = dt.createTree(dataSet, labels)