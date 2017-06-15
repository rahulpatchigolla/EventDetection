import numpy as np
import sklearn as sk
import random
import csv
import re
import collections
# from geniatagger import GeniaTagger
# tagger = GeniaTagger("/home/sunil/packages/geniatagger-3.0.2/geniatagger")
from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec

tokenizer = WordPunctTokenizer()
import pickle


def preProcess(sent):
    newsent = []
    for word in sent:
        if not word == "TRIGGER" and not word == "ARGUEMENT":
            word = word.lower()
        word = word.replace('/', ' ')
        #	sent = sent.replace('dg','')
        #	sent = sent.replace('(','')
        #	sent = sent.replace(')','')
        #	sent = sent.replace('[','')
        #	sent = sent.replace(']','')
        word = word.replace('.', '')
        #	sent = sent.replace(',',' ')
        #	sent = sent.replace(':','')
        #	sent = sent.replace(';','')
        word = word.split(' ')
        word = word[0]
        newsent.append(word)
    # sent = re.sub('\d', 'dg',sent)
    return newsent


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1


def makePaddedList(sent_contents, maxl, pad_symbol='<pad>'):
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        for i in range(lenth):
            t.append(sent[i])
        for i in range(lenth, maxl):
            t.append(pad_symbol)
        T.append(t)

    return T


def makeWordList(sent_lista, sent_listb):
    sent_list = sent_lista + sent_listb
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = collections.OrderedDict()  # orederd dictionary
    i = 1
    wl['unkown'] = 0
    for w, f in wf.items():
        wl[w] = i
        i += 1
    return wl


def mapWordToId(sent_contents, word_dict):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_dict[w])
        T.append(t)
    return T


def mapLabelToId(sent_labels, label_dict):
    #	print"sent_lables", sent_lables
    #	print"label_dict", label_dict
    # return [label_dict[label] for label in sent_lables]
    rval = []
    for label in sent_labels:
        if label_dict.has_key(label):
            rval.append(label_dict[label])
        else:
            label1 = label[1:]
            label2 = label[:len(label) - 1]
            if label_dict.has_key(label1):
                rval.append(label_dict[label1])
            elif label_dict.has_key(label2):
                rval.append(label_dict[label2])
            else:
                print "Not found label"
                exit(0)
    return rval


#	return [int (label != 'false') for label in sent_lables]


def dataRead(fname):
    print ("Input File Reading")
    fp = open(fname, 'r')
    samples = fp.read().strip().split('\n\n')
    sent_lengths = []  # 1-d array
    sent_contents = []  # 2-d array [[w1,w2,....] ...]
    sent_lables = []  # 1-d array
    entity1_list = []  # 2-d array [[e1,e1_t] [e1,e1_t]...]
    entity2_list = []  # 2-d array [[e1,e1_t] [e1,e1_t]...]
    for sample in samples:
        sent, entities, relation = sample.strip().split('\n')
        #		if len(sent.split()) > 100:
        #			continue
        e1, e1_t, e2, e2_t = entities.split('\t')
        sent_contents.append(sent.lower())
        entity1_list.append([e1, e1_t])
        entity2_list.append([e2, e2_t])
        sent_lables.append(relation)

    return sent_contents, entity1_list, entity2_list, sent_lables


def makeFeatures(sent_list, entity1_list, entity2_list):
    print ('Making Features')
    word_list = []
    d1_list = []
    d2_list = []
    type_list = []
    count = 0
    for sent, entity1, entity2 in zip(sent_list, entity1_list, entity2_list):
        count += 1
        sent = preProcess(sent)
        # print sent
        s1 = sent.index('TRIGGER')
        s2 = sent.index('ARGUEMENT')
        # distance1 feature
        d1 = []
        for i in range(len(sent)):
            if i < s1:
                d1.append(str(i - s1))
            elif i > s1:
                d1.append(str(i - s1))
            else:
                d1.append('0')
        # distance2 feature
        d2 = []
        for i in range(len(sent)):
            if i < s2:
                d2.append(str(i - s2))
            elif i > s2:
                d2.append(str(i - s2))
            else:
                d2.append('0')
        newword1 = preProcess([entity1])
        newword2 = preProcess([entity2])
        sent[s1] = newword1[0]
        sent[s2] = newword2[0]
        word_list.append(sent)
        d1_list.append(d1)
        d2_list.append(d2)
        '''if count<=100:
			print sent
			print d1
			print d2'''
    return word_list, d1_list, d2_list


def readWordEmb(word_dict, fname, embSize=50):
    print ("Reading word vectors")
    wv = []
    wl = []
    with open(fname, 'r') as f:
        for line in f:
            vs = line.split()
            #			print (len(vs))
            if len(vs) != embSize + 1:
                continue
            vect = list(map(float, vs[1:]))
            #			print (vect[0:5] )
            wv.append(vect)
            wl.append(vs[0])
        #	print ("wv",wv[0:10])
    wordemb = []
    count = 0
    for word, id in word_dict.items():
        if word in wl:
            wordemb.append(wv[wl.index(word)])
        else:
            count += 1
            wordemb.append(np.random.rand(embSize))
        #	print (wordemb)
    # wordemb = np.asarray(map(float, wordemb))
    wordemb = np.asarray(wordemb, dtype='float32')
    print ("number of unknown word in word embedding", count)
    return wordemb


def loadWordEmbeddings(wordVocab, embSize):
    print "Loading Word Embeddings..."
    print "Total Words:", len(wordVocab)
    model = Word2Vec.load_word2vec_format('/home/rahul/PycharmProjects/PubMed-w2v.bin', binary=True)
    wordemb = []
    count = 0
    for word in wordVocab.keys():
        if model.__contains__(word):
            wordemb.append(model[word])
        else:
            count += 1
            wordemb.append(np.random.rand(embSize))
        #	print (wordemb)
    # wordemb = np.asarray(map(float, wordemb))
    # wordemb[wordVocab.index('<pad>')] = np.zeros(embSize)
    wordemb = np.asarray(wordemb, dtype='float32')
    print ("number of unknown word in word embedding", count)
    return wordemb


def findLongestSent(Tr_word_list, Te_word_list):
    combine_list = Tr_word_list + Te_word_list
    a = max([len(sent) for sent in combine_list])
    return a


def findSentLengths(tr_te_list):
    lis = []
    for lists in tr_te_list:
        lis.append([len(l) for l in lists])
    return lis


def paddData(listL, maxl):  # W_batch, d1_tatch, d2_batch, t_batch)
    rlist = []
    for mat in listL:
        mat_n = []
        for row in mat:
            lenth = len(row)
            t = []
            for i in range(lenth):
                t.append(row[i])
            for i in range(lenth, maxl):
                t.append(0)
            mat_n.append(t)
        rlist.append(np.array(mat_n))
    return rlist


def makeBalence(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
    sent_contents = [];
    entity1_list = [];
    entity2_list = [];
    sent_lables = [];
    other = []
    clas = []
    for sent, e1, e2, lab in zip(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
        if lab == 'false':
            other.append([sent, e1, e2, lab])
        else:
            clas.append([sent, e1, e2, lab])

    random.shuffle(other)

    neg = other[0: 3 * len(clas)]
    l = neg + clas
    for sent, e1, e2, lab in l:
        sent_contents.append(sent)
        entity1_list.append(e1)
        entity2_list.append(e2)
        sent_lables.append(lab)
    return sent_contents, entity1_list, entity2_list, sent_lables


def readData(datafile):
    fp = open(datafile, "r")
    content = fp.read()
    content = content.strip('\n').split('\n\n')
    docWordList = []
    docEntityList = []
    docword1List = []
    docword2List = []
    #docParseList = []
    docLabelList = []
    count = 0
    for lines in content:
        count += 1
        lines = lines.split('\n')
        wordList = lines[0].strip('#').split('#')
        entityList = lines[1].strip('#').split('#')
        try:
                assert (len(wordList) == len(entityList))
        except:
        	print len(wordList),len(entityList)
        	print wordList
        	print entityList
        	exit(0)
        word1, word2 = lines[2].strip('\n').split('\t')
        #parseList = lines[3].strip('#').split('#')
        label = lines[3].strip("\n")
        docWordList.append(wordList)
        docEntityList.append(entityList)
        docword1List.append(word1)
        docword2List.append(word2)
        #docParseList.append(parseList)
        docLabelList.append(label)
        '''if count<=100:
			print wordList
			print entityList
			print  word1
			print word2
			print label'''
    return docWordList, docword1List, docword2List, docLabelList, docEntityList

