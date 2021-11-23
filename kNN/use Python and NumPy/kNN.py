'''
@author: wepon
@github: https://github.com/wepe
@blog:   http://blog.csdn.net/u012162613
'''
#!/usr/bin/python
#-*-coding:utf-8-*-
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    '''
    :param  Matrix inX: test case
    :param Matrix dataSet: known classes that used to compare with the test one.
    :param List labels: different classes
    :param Int k: the number of neighbour used to check
    :return: class
    '''
    dataSetSize = dataSet.shape[0]
    # return the row number if shape[1] return column number
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # array.sum(axis=1)按行累加，axis=0为按列累加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # L2 norm is calculated above
    sortedDistIndicies = distances.argsort()
    # argsort() sort the data in increasing order and return the index
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # reverse = True decrease, reverse = False increase
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():

    hwLabels = []
    trainingFileList = listdir('trainingDigits')          
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]                  
        fileStr = fileNameStr.split('.')[0]                
        classNumStr = int(fileStr.split('_')[0])          
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
     
    testFileList = listdir('testDigits')       
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()
