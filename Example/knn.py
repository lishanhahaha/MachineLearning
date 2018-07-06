import os
import operator
import numpy as np


TEST_DIGITS = r'C:\Users\Administrator\Desktop\demo\digits\testDigits'
TRAIN_DIGITS = r'C:\Users\Administrator\Desktop\demo\digits\trainingDigits'

def file2matrix(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createClassSet():
    dataSet = np.array([[1.0,1.0], [1.0,1.1], [0,0], [0,0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return dataSet, labels


def handleWriteClassify():
    hwLabels = []
    trainingFileList = os.listdir(TRAIN_DIGITS)  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = file2matrix(TRAIN_DIGITS + os.sep + fileNameStr)
    test_files = os.listdir(TEST_DIGITS)  # iterate through the test set
    error_count = 0.0
    num_test = len(test_files)
    for i in range(num_test):
        fileNameStr = test_files[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = file2matrix(TEST_DIGITS + os.sep + fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): error_count += 1.0
    print(error_count/float(num_test))



if __name__ == '__main__':
    handleWriteClassify()
