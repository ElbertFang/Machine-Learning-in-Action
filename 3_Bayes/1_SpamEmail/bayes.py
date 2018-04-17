import numpy as np

def createVoc(dataset):
    voc_set = set([])
    for i in dataset:
        voc_set = voc_set | set(i)
    return list(voc_set)

def docVec(voc_list, input_doc):
    return_voc = [0]*len(voc_list)
    print_list = []
    for i in input_doc:
        if i in voc_list:
            return_voc[voc_list.index(i)] += 1
        else:
            print_list.append(i)
    # if len(print_list)>0:
    #     print("The words \"",print_list,"\" is not in my vocabulary !")
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

def textParse(doc_string):
    import re
    token_list = re.split(r'\W*', doc_string)
    return [token.lower() for token in token_list if(len(token)>2)]

def spamTest():
    doc_list = []; class_list = []; full_list = []
    for i in range(1,26):
        word_list = textParse(open("email/spam/%d.txt"%i).read())
        doc_list.append(word_list)
        full_list.extend(word_list)
        class_list.append(1)
        word_list = textParse(open("email/ham/%d.txt"%i).read())
        doc_list.append(word_list)
        full_list.extend(word_list)
        class_list.append(0)
    voc_list = createVoc(full_list)
    traing_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(traing_set)))
        test_set.append(traing_set[rand_index])
        del(traing_set[rand_index])
    train_mat = []
    train_class = []
    for i in traing_set:
        train_mat.append(docVec(voc_list, doc_list[i]))
        train_class.append(class_list[i])
    p0,p1,pc = trainNB(np.array(train_mat),np.array(train_class))
    error_count = 0
    for i in test_set:
        word_vec = docVec(voc_list, doc_list[i])
        if classify(np.array(word_vec),p0,p1,pc)!=class_list[i]:
            error_count+=1
    print("The error rate is : ",float(error_count)/len(test_set))