import numpy as np
import operator

def readData(filename):
    file = open(filename)
    file_data = file.readlines()
    num_lines = len(file_data)
    returnMat = np.zeros((num_lines, 3))
    label_vector = []
    index = 0
    for line in file_data:
        line = line.strip().split('\t')
        returnMat[index,:] = line[0:3]
        label_vector.append(int(line[-1]))
        index+=1
    return returnMat, label_vector

def autoNorm(dataSet):
    min = dataSet.min(0)
    max = dataSet.max(0)
    ranges = max - min
    normData = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normData = dataSet - np.tile(min, (m,1))
    normData = normData/np.tile(ranges, (m,1))
    return normData, ranges, min

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

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percenrage of time spent playing video games ?  "))
    ffMiles = float(input("frequent filer miles earned per year ?  "))
    iceCream = float(input("liters of ice cream consumed per year ?  "))
    data, labels = readData('datingTestSet2.txt')
    normMat, ranges, min = autoNorm(data)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-min)/ranges,normMat, labels, 3)
    print("You will probaly like this person : ",resultList[classifierResult-1])


def test(file):
    ratio = 0.1
    data, labels = readData(file)
    normMat, ranges, min = autoNorm(data)
    m = normMat.shape[0]
    numTest = int(m * ratio)
    errorCount = 0.0
    for i in range(numTest):
        classifyResult = classify(normMat[i,:], normMat[numTest:m,:],labels[numTest:m],3)
        print("The classifier return : %d , and the ground truth is %d"%(classifyResult, labels[i]))
        if(classifyResult != labels[i]):
            errorCount += 1.0
    print("The total error rate is %f"%(errorCount/float(numTest)))