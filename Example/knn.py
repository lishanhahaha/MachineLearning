#-*-coding:utf-8-*-

import os
import time
import operator

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN


TEST_DIGITS = r'F:\MLiA_SourceCode\machinelearninginaction\Ch02\digits\testDigits'
TRAIN_DIGITS = r'F:\MLiA_SourceCode\machinelearninginaction\Ch02\digits\trainingDigits'

def createDataSetAndLabels():
    ''' 创建数据集及相应的标签

    Parameters:

    Returns
        dataSet:        创建的数据集
        labels:         创建的数据集对应的标签
    Modify:
        2018-07-06
    '''
    dataSet = np.array([[1.0,1.0], [1.0,1.1], [0,0], [0,0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return dataSet, labels

def classifyKNN(testData, dataSet, labels, k):
    ''' kNN算法分类器

    Parameters:
        testData        -   用于分类的测试集
        dataSet         -   用于训练的数据集
        labels          -   分类标签
        k               -   kNN算法参数，选择距离最小的k个点
    Returns:
        classResult[0][0]     -   分类结果
    Modify:
        2018-07-06
    '''
    # 计算距离
    row = dataSet.shape[0]
    diffMat = np.tile(testData, (row,1)) - dataSet
    distances = ( (diffMat ** 2).sum(axis=1)) ** 0.5
    # 计算最小距离
    classCount = {}
    sortedDistIndex = distances.argsort()       # 计算升序所对应的索引号
    for i in range(k):
        lbl = labels[sortedDistIndex[i]]
        classCount[lbl] = classCount.get(lbl, 0) + 1
    # 计算降序并分类出结果
    classResult = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classResult[0][0]

def file2matrix(filename):
    ''' 用于将文件中的手写数字集转换成矩阵

    Parameters:
        filename            -   文件路径
    Returns:
        dataSet             -   返回文件里面的矩阵数据集
    Modify:
        2018-07-06
    '''
    f = open(filename)
    dataSet = np.zeros((1, 1024))
    for i in range(32):
        data = f.readline()
        for j in range(32):
            dataSet[0,32*i+j] = int(data[j])
    f.close()
    return dataSet

def handwriteClassTest():
    ''' 用于测试手写分类数字
    Parameters:

    Returns:

    Modify:
        2018-07-06
    '''
    # 训练数据集
    listTrainFile = os.listdir(TRAIN_DIGITS)
    numTrainFile = len(listTrainFile)
    trainLabels = []
    trainDataSet = np.zeros((numTrainFile,1024))
    for i in range(numTrainFile):
        lbl = listTrainFile[i].split('_')
        trainLabels.append(int(lbl[0]))
        trainDataSet[i,:] = file2matrix(TRAIN_DIGITS+os.sep+listTrainFile[i])
    # 测试训练集
    listTestFile = os.listdir(TEST_DIGITS)
    numTestFile = len(listTestFile)
    errorCount = 0
    for i in range(numTestFile):
        lbl = listTestFile[i].split('_')
        realResult = int(lbl[0])
        testDataSet = file2matrix(TEST_DIGITS+os.sep+listTestFile[i])
        testResult = int(classifyKNN(testDataSet, trainDataSet, trainLabels, 3))
        print('真实结果：%s      分类结果：%s' % (realResult, testResult))
        if (realResult != testResult):
            errorCount += 1
    errorRatio = round(errorCount/numTestFile*100, 2)
    print('错误率：%s %%' % (errorRatio), errorCount, numTestFile, type(errorCount), type(numTestFile))

def handwriteClassKNN():
    trainFileList = os.listdir(TRAIN_DIGITS)
    trainFileNum = len(trainFileList)
    labels = []
    trainDataSet = np.zeros((trainFileNum, 1024))
    for i in range(trainFileNum):
        lbl = trainFileList[i].split('_')
        labels.append(int(lbl[0]))
        trainDataSet[i,:] = file2matrix(TRAIN_DIGITS+os.sep+trainFileList[i])
    # 运用kNN算法进行分类
    testFileList = os.listdir(TEST_DIGITS)
    testFileNum = len(testFileList)
    neigh = kNN(n_neighbors=3, weights='uniform', algorithm='auto')
    neigh.fit(trainDataSet, labels)
    errorCount = 0
    for i in range(testFileNum):
        lbl = testFileList[i].split('_')
        realResult = int(lbl[0])
        testDataSet = file2matrix(TEST_DIGITS+os.sep+testFileList[i])
        classResult = int(neigh.predict(testDataSet))
        print('真实值：%s       预测值：%s' % (realResult, classResult))
        if (realResult != classResult):
            errorCount += 1
    errorRatio = errorCount/testFileNum*100
    print('错误率：%s %%' % (errorRatio))


if __name__ == '__main__':
    startTime = time.time()
    '''dataSet, labels = createDataSetAndLabels()
    classResult = classifyKNN([1,0.1], dataSet, labels, 2)
    print(classResult)'''
    #handwriteClassTest()
    handwriteClassKNN()
    endTime = time.time()
    print('测试时间：%s s' % (endTime - startTime))
