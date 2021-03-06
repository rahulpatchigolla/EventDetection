import numpy as np
import theano
import theano.tensor as T
import os
from os import listdir
from nltk.tokenize import PunktSentenceTokenizer
import cPickle
import numpy
from nltk.tokenize import word_tokenize
from Utils import *
from TriggerModel import *
from SentenceExtractor import *
from random import shuffle
processedpathtrain="./Preprocessed_Corpus_train/"
corpuspathtrain="./Corpus_filtered/train/"
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus_filtered/test/"
ScoreTrain=[]
ScoreTest=[]
#This File performs contains the training procedure of the model
# This method tests and dumps test files triggers in a file for evaluation after every epoch
def testAndDumpTriggers(test_set,globalScore):
    error = 0.0
    sentenceCount = 0
    fppred = open("testPredictionsTrigger.txt", "w")
    predTriggersTotal=[]
    print "Testing"
    for key in test_set:
        #print "\n\n\n*******************", key, "*******************\n\n\n"
        predTriggersDoc=[]
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(processedpathtest + key + ".pkl", 'rb'))
        fp = open(corpuspathtest + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, eventList, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                         eventIdDict, eventIndexDict,
                                                                                         content, sentIndex, eventVocab,
                                                                                         wordVocab,removeRareTriggers=True)
        for index in range(0, len(sentenceList)):
            predTriggersSent=[]
            sentence = sentenceList[index]
            entities = entityList[index]
            events = eventList[index]
            input1, input2, labels = compute(sentence, entities, model, events)
            pred_labels = get_predictions(input1, input2,dropouttest)
            er = test_model(input1, input2,dropouttest,labels)
            error += er
            sentenceCount += 1
            actual_labels = labels
            pred_labels=postProcessPredictedLabels(pred_labels,inveventVocab)
            for j in range(0, len(pred_labels)):
                fppred.write(invWordVocab[input1[j]] + "/" + inventityVocab[input2[j]] + " " + inveventVocab[
                    actual_labels[j]] + " " + pred_labels[j])
                #fppred.write(invWordVocab[input1[j]]+"/"+inventityVocab[input2[j]]+" "+inveventVocab[actual_labels[j]] + " " + inveventVocab[pred_labels[j]])
                fppred.write("\n")
                predTriggersSent.append(pred_labels[j])
                #predTriggersSent.append(inveventVocab[pred_labels[j]])
            fppred.write("\n")
            predTriggersDoc.append(predTriggersSent)
        predTriggersTotal.append(predTriggersDoc)
    fppred.close()
    postProcessResults("testPredictionsTrigger.txt")
    curScore=getF1Score("Other","testPredictionsTrigger.txt")
    print "F1Score on test data",curScore
    ScoreTest.append(curScore)
    if curScore > globalScore:
        updateGlobalFile("testPredictionsTrigger.txt","bestTestPredictionsTrigger.txt")
        globalScore=curScore
        extractor = sentenceExtractor()
        for key,predTriggersDoc in zip(test_set,predTriggersTotal):
            extractor.dumpPredictedTriggers(key,predTriggersDoc)
    return globalScore
if __name__ == "__main__":
    print "Event Trigger Detection..."
    dirDicttrain = readDir(corpuspathtrain)
    dirDicttest=readDir(corpuspathtest)
    # Constructing the final Vocabularies
    wordVocabtrain = cPickle.load(open(processedpathtrain + "wordVocab.pkl", 'rb'))
    entityVocabtrain = cPickle.load(open(processedpathtrain + "entityVocab.pkl", 'rb'))
    eventVocabtrain = cPickle.load(open(processedpathtrain + "eventVocab.pkl", 'rb'))
    wordVocabtest = cPickle.load(open(processedpathtest + "wordVocab.pkl", 'rb'))
    entityVocabtest = cPickle.load(open(processedpathtest + "entityVocab.pkl", 'rb'))
    eventVocabtest = cPickle.load(open(processedpathtest + "eventVocab.pkl", 'rb'))
    wordVocab=mergeVocabs(wordVocabtrain,wordVocabtest)
    eventVocab=mergeVocabs(eventVocabtrain,eventVocabtest,removeRareEvents=True)
    #print eventVocab
    entityVocab = mergeVocabs(entityVocabtrain, entityVocabtest)
    #print entityVocab
    inveventVocab = {v: k for k, v in eventVocab.items()}
    invWordVocab = {v: k for k, v in wordVocab.items()}
    inventityVocab = {v: k for k, v in entityVocab.items()}
    eventCount=0
    actualCount=0
    L1_reg=0.001
    L2_reg = 0.0001
    learning_rate=0.01
    nepochs=50
    globalScore=0
    dropouttrain=np.asarray(0.2,dtype=theano.config.floatX)
    dropouttest = np.asarray(0.0,dtype=theano.config.floatX)
    x1=T.ivector('x1')
    x2 = T.ivector('x2')
    x3=T.dscalar('x3')
    y=T.ivector('y')
    rng = numpy.random.RandomState(1234)
    model = MyTriggerModel(rng=rng, wordVocab=wordVocab, entityVocab=entityVocab, eventVocab=eventVocab, embSizeWord=200,
                    embSizeEntity=50, RnnHiddenDim=250, FFhiddLayerDim=150,input1=x1,input2=x2,dropout=x3)
    cost = (
        model.negative_log_likelihood(y)
        + L1_reg * model.L1
        + L2_reg * model.L2_sqr
    )
    gparams = [T.grad(cost, param) for param in model.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model.params, gparams)
        ]
    #updates=adam(cost,model.params)
    train_model = theano.function(
        inputs=[x1,x2,x3,y],
        outputs=cost,
        updates=updates,
    )
    test_model = theano.function(
        inputs=[x1,x2,x3,y],
        outputs=model.errors(y)
    )
    get_predictions=theano.function(
        inputs=[x1,x2,x3],
        outputs=model.predict()
    )
    train_set = dirDicttrain.keys()
    test_set = dirDicttest.keys()
    #print train_set
    #print test_set
    for key in test_set:
        if dirDicttrain.has_key(key):
            print key
            exit(0)
    print "Training..."
    for i in range(0, nepochs):
        fppred=open("trainPredictionsTrigger.txt", "w")
        loss=0.0
        sentenceCount=0
        error=0.0
        shuffle(train_set)
        shuffle(test_set)
        for key in train_set:
            #print "\n\n\n*******************", key, "*******************\n\n\n"
            #key="PMID-2255190"
            # Retrive the appropriate processed data  and text for the sentence extractor to work
            sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict=cPickle.load(open(processedpathtrain+key+".pkl",'rb'))
            #print eventIdDict
            #print eventIndexDict
            #print "lenghts",len(eventIdDict),len(eventIndexDict)
            fp=open(corpuspathtrain+key+".txt",'r')
            content=fp.read()
            extractor=sentenceExtractor()
            # Extracting the list of entities,events and words using sentence extractor
            sentenceList, entityList, eventList,arguementList=extractor.entitiesAndEvents(entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex,eventVocab,wordVocab,removeRareTriggers=True)
            for index in range(0, len(sentenceList)):
                #print "sentence",index
                sentence = sentenceList[index]
                entities = entityList[index]
                events = eventList[index]
                # Computes the indexes of the lists extracted above
                input1Sent,input2Sent,labels=compute(sentence,entities,model,events)
                l =train_model(input1Sent,input2Sent,dropouttrain,labels)
                er=test_model(input1Sent,input2Sent,dropouttest,labels)
                pred_labels = get_predictions(input1Sent, input2Sent,dropouttest)
                actual_labels = labels
                # Dumps every sentence for getting the F1-Score
                for j in range(0, len(pred_labels)):
                    fppred.write(invWordVocab[input1Sent[j]] + "/" + inventityVocab[input2Sent[j]] + " " +
                        inveventVocab[actual_labels[j]] + " " + inveventVocab[pred_labels[j]])
                    fppred.write("\n")
                fppred.write("\n")
                loss+=l
                error+=er
                sentenceCount+=1
        fppred.close()
        # It involves merging procedure for multi word triggers  to get the F1-score
        postProcessResults("trainPredictionsTrigger.txt")
        curScore = getF1Score("Other","trainPredictionsTrigger.txt")
        print "F1Score on training data", curScore
        print "loss iteration:",i,loss/sentenceCount
        ScoreTrain.append(curScore)
        globalScore=testAndDumpTriggers(test_set,globalScore)
    fp = open("scoresTrainTrigger.txt", "w")
    for value in ScoreTrain:
        fp.write(str(value) + "\n")
    fp.close()
    fp = open("scoresTestTrigger.txt", "w")
    for value in ScoreTest:
        fp.write(str(value) + "\n")
    fp.close()











