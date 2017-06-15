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
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus/standoff/test/test/"
#This file is used for creating the event annotations from the predicted triggers and arguments
if __name__ == "__main__":
    dirDicttest = readDir(corpuspathtest)
    test_set = dirDicttest.keys()
    wordVocabtrain = {}
    entityVocabtrain = {}
    eventVocabtrain = {}
    wordVocabtest = cPickle.load(open(processedpathtest + "wordVocab.pkl", 'rb'))
    entityVocabtest = cPickle.load(open(processedpathtest + "entityVocab.pkl", 'rb'))
    eventVocabtest = cPickle.load(open(processedpathtest + "eventVocab.pkl", 'rb'))
    wordVocab = mergeVocabs(wordVocabtrain, wordVocabtest)
    eventVocab = mergeVocabs(eventVocabtrain, eventVocabtest, removeRareEvents=True)
    entityVocab=mergeVocabs(entityVocabtest,eventVocab)
    del entityVocab["None"]
    print entityVocab
    labelVocab = {"No": 0, "Theme": 1, "AtLoc": 2, "ToLoc": 3, "FromLoc": 4, "Site": 5, "Cause": 6, "Instrument": 7}
    shuffle(test_set)
    for key in test_set:
        #key="PMID-12558942"
        print "\n\n\n*******************", key, "*******************\n\n\n"
        extractor = sentenceExtractor()
        predictedtriggertest=extractor.loadPredictedTriggers(key)
        predictedarguementtest=extractor.loadPredictedArguements(key)
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(processedpathtest + key + ".pkl", 'rb'))
        fp = open(corpuspathtest + key + ".txt", 'r')
        content = fp.read()
        sentenceList, entityList, _, _,indexList= extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                 eventIdDict, eventIndexDict, content,
                                                                                 sentIndex, eventVocab, wordVocab,
                                                                                 removeRareTriggers=True,
                                                                                 concatenate=True,getindexlist=True,loadpredTriggers=True,key=key)
        arguementList=extractor.loadPredictedArguements(key)
        triggerList=extractor.loadPredictedTriggers(key)

        extractor.dumpPredictedEvents(key,sentenceList,entityList,arguementList,indexList,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,eventVocab,triggerList,entityVocab)
        #exit(0)