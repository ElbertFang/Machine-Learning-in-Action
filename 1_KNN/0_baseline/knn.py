import numpy as np
import operator

def createData():
    data = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return data,labels

def classify(test_data, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    difMat = np.tile(test_data, (dataSize, 1)) - dataSet
    sqMat = difMat**2
    distance = sqMat.sum(axis=1)**0.5
    sortDistance = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortClass = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortClass[0][0]