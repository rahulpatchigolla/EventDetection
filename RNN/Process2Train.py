import numpy as np
import theano
import theano.tensor as T
import keras
import os
from os import listdir
from nltk.tokenize import PunktSentenceTokenizer
import cPickle
import numpy

from nltk.tokenize import word_tokenize
from Utils import *
#from Event.MyModelFile import
from TriggerModel import *
from copy import deepcopy
from SentenceExtractor import *
from random import shuffle
processedpathtrain="./Preprocessed_Corpus_train/"
corpuspathtrain="./Corpus/standoff/test/train/"
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus/standoff/test/test/"
#This file generates train data for other argument models mentioned in the paper
if __name__ == "__main__":
    dirDicttrain = readDir(corpuspathtrain)
    dirDicttest = readDir(corpuspathtest)
    train_set = dirDicttrain.keys()
    test_set = dirDicttest.keys()
    fpdump=open("train.txt","w")
    wordVocabtrain = cPickle.load(open(processedpathtrain + "wordVocab.pkl", 'rb'))
    entityVocabtrain = cPickle.load(open(processedpathtrain + "entityVocab.pkl", 'rb'))
    eventVocabtrain = cPickle.load(open(processedpathtrain + "eventVocab.pkl", 'rb'))
    wordVocabtest = cPickle.load(open(processedpathtest + "wordVocab.pkl", 'rb'))
    entityVocabtest = cPickle.load(open(processedpathtest + "entityVocab.pkl", 'rb'))
    eventVocabtest = cPickle.load(open(processedpathtest + "eventVocab.pkl", 'rb'))
    wordVocab = mergeVocabs(wordVocabtrain, wordVocabtest)
    eventVocab = mergeVocabs(eventVocabtrain, eventVocabtest, removeRareEvents=False)
    entityVocab = mergeVocabs(entityVocabtrain, entityVocabtest)
    for key in train_set:
        print "\n\n\n*******************", key, "*******************\n\n\n"
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(open(processedpathtrain + key + ".pkl", 'rb'))
        fp = open(corpuspathtrain + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, _, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,eventIdDict, eventIndexDict,content, sentIndex, eventVocab,wordVocab,removeRareTriggers=False,concatenate=True)
        for index in range(0,len(sentenceList)):
            sentence=sentenceList[index]
            entities=entityList[index]
            print sentence
            print entities
            assert len(sentence)==len(entities)
            arguements=arguementList[index]
            for index1 in range(0,len(sentence)):
                word=sentence[index1]
                entity=entities[index1]
                wordarguements=arguements[index1]
                if eventVocab.has_key(entity):
                    _, input4 = truncateLableswithRules(wordarguements, entities, entityVocab, entity)
                    assert len(wordarguements) > 0
                    for index2 in range(0,len(sentence)):
                        word1=sentence[index1]
                        word2=sentence[index2]
                        if (index1==index2) or (index2 not in input4):
                            continue
                        if entities[index2]=="None":
                            if wordarguements[index2]=="No":
                                continue
                            else:
                                print word1,entities[index1],word2,entities[index2],wordarguements[index2]
                        dumpsentence=deepcopy(sentence)
                        dumpsentence[index1]="TRIGGER"
                        dumpsentence[index2]="ARGUEMENT"
                        assert len(dumpsentence)==len(entities)
                        fpdump.write(convertToSentence(dumpsentence,"#"))
                        fpdump.write("\n")
                        fpdump.write(convertToSentence(entities,"#"))
                        fpdump.write("\n")
                        fpdump.write(word1+"\t"+word2)
                        fpdump.write("\n")
                        fpdump.write(wordarguements[index2])
                        fpdump.write("\n\n")
    fpdump.close()
    '''fp = open("train.txt", "r")
    content = fp.read()
    content = content.split('\n\n')
    for lines in content:
        lines=lines.split('\n')
        print lines[0]
        print lines[1]
        wordline=lines[0].split('#')
        entityline=lines[1].split('#')
        print wordline
        print entityline
        print len(wordline)
        print len(entityline)
        assert(len(wordline)==len(entityline))
    exit(0)'''
