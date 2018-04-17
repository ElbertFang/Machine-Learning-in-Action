import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def creatVoc(dataset):
    voc_set = set([])
    for i in dataset:
        voc_set = voc_set | set(i)
    return list(voc_set)

def docVec(voc_list, input_doc):
    return_voc = [0]*len(voc_list)
    for i in input_doc:
        if i in voc_list:
            return_voc[voc_list.index(i)] = 1
        else:
            print("The word \"%s\" is not in my vocabulary !"%i)
    return return_voc

def trainNB(train_mat, train_label):
    num_docs = len(train_mat)
    num_words = len(train_mat[0])
    pc = sum(train_label)/float(num_docs)
    p0 = np.ones(num_words)
    p1 = np.ones(num_words)
    p0num = 2.0
    p1num = 2.0
    for i in range(num_docs):
        if train_label[i] == 1:
            p1 += train_mat[i]
            p1num += sum(train_mat[i])
        else:
            p0 += train_mat[i]
            p0num += sum(train_mat[i])
    p0vec = np.log(p0/p0num)
    p1vec = np.log(p1/p1num)
    return p0vec, p1vec, pc

def classify(vec,p0,p1,pc):
    p0 = sum(vec*p0)+np.log(1-pc)
    p1 = sum(vec*p1)+np.log(pc)
    if(p1>p0):
        return 1
    else:
        return 0

def test(doc):
    train_data,train_label = loadDataSet()
    voc_list = creatVoc(train_data)
    train_mat = []
    for i in train_data:
        train_mat.append(docVec(voc_list, i))
    p0,p1,pc = trainNB(train_mat, train_label)
    doc_vec = np.array(docVec(voc_list, doc))
    print(doc," classified as : ",classify(doc_vec,p0,p1,pc))
    return 0
