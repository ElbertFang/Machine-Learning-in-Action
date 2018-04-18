import numpy as np
import operator
from os import listdir

def img2vec(file):
    returnVec = np.zeros((1,1024))
    file = open(file)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(line[j])
    return returnVec

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

def test(train_data, test_data):
    labels = []
    train_list = listdir(train_data)
    test_list = listdir(test_data)
    m = len(train_list)
    n = len(test_list)
    train_mat = np.zeros((m,1024))
    for i in range(m):
        file_str = train_list[i]
        file_label = int(file_str.split('.')[0].split('_')[0])
        labels.append(file_label)
        train_mat[i,:] = img2vec(train_data+'/%s'%file_str)
    error = 0.0
    for i in range(n):
        file_str = test_list[i]
        file_label = int(file_str.split('.')[0].split('_')[0])
        file_vec = img2vec(test_data+'/%s'%file_str)
        file_result = classify(file_vec, train_mat, labels, 3)
        if (file_result != file_label):
            print("The result of %s is wrong."%file_str)
            print("The classifier return : %d, and the real answer is %d." % (file_result, file_label))
            error += 1.0
    print("\nThe total number of error is %d.\n" % error)
    print("The total error rate is %f." % (error/float(m)))