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
#from Event.MyModelFile import
from ArguementModel import *
from SentenceExtractor import *
from random import shuffle
processedpathtrain="./Preprocessed_Corpus_train/"
corpuspathtrain="./Corpus_filtered/train/"
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus_filtered/test/"
ScoreTrain=[]
ScoreTest=[]
#This File performs contains the training procedure of the argument identificationmodel
# This method tests and dumps test files arguments in a file for evaluation after every epoch
def testAndDumpArguements(test_set,globalScore):
    error = 0.0
    sentenceCount = 0
    fppred = open("testPredictionsArguements.txt", "w")
    predArguementsTotal=[]
    for key in test_set:
        #print "\n\n\n*******************", key, "*******************\n\n\n"
        predArguementsDoc = []
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(processedpathtest + key + ".pkl", 'rb'))
        fp = open(corpuspathtest + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, _, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                 eventIdDict, eventIndexDict,
                                                                                 content, sentIndex, eventVocab,
                                                                                 wordVocab, removeRareTriggers=True,
                                                                                 concatenate=True,loadpredTriggers=True,key=key)
        sentenceList, entityList, arguementList, sentenceIdList, sentenceRootList, sentenceChunkList, sentencePosList, sentenceParentList, sentenceParseList = extractor.loadParseTrees(key, sentenceList, entityList, arguementList, "test/")
        for index in range(0, len(sentenceList)):
            predArguementsSent=[]
            sentence = sentenceList[index]
            entities = entityList[index]
            arguements = arguementList[index]
            sentenceIds = sentenceIdList[index]
            dependencies = sentenceParseList[index]
            parents = sentenceParentList[index]
            for index1 in range(0, len(sentence)):
                predArguementsWord=[]
                #word = sentence[index1]
                entity = entities[index1]
                wordarguements = arguements[index1]

                if eventVocab.has_key(entity):
                    depPathvertices, depPathEdges,_ = extractor.getShortestDependencyPath(sentenceIds, sentence,dependencies, parents,str(index1 + 1),entities)
                    #distances = generateArguementFeatures(sentence, index1)
                    distances = generateDepArguementFeatures(sentence, depPathvertices, index1)
                    if len(wordarguements)==0:
                        wordarguements=["No"]*len(sentence)
                    assert len(wordarguements)==len(sentence)
                    wordarguements, input4 = truncateLableswithRules(wordarguements, entities, entityVocab, entity)
                    input1, input2, input3, labels = compute(sentence, entities, distances, wordarguements, model)
                    #labels, input4 = truncateLabels(labels, input2, inventityVocab)
                    input1, input2, input3, input4, labels, flag = checkLengthAndUpdateInput(input1, input2, input3,
                                                                                             input4, labels, wordVocab,
                                                                                             entityVocab, distanceVocab,
                                                                                             labelVocab, index1)
                    if len(input4)>0:
                        #pred_labels = get_predictions(index1, input1, input2,input3,input4,dropouttest)
                        viterbi_max, viterbi_argmax = get_predictions(index1, input1, input2, input3, input4,dropouttest)
                        first_ind = np.argmax(viterbi_max[-1])
                        viterbi_pred = backtrack(first_ind, viterbi_argmax)
                        pred_labels = np.array(viterbi_pred)
                        er = test_model(index1, input1, input2, input3, input4,dropouttest,labels)
                    else:
                        pred_labels=[]
                        er=0
                    error += er
                    sentenceCount += 1
                    actual_labels = labels
                    for j in range(0, len(pred_labels)-flag):
                        fppred.write(invWordVocab[input1[input4[j]]]+"/"+inventityVocab[input2[input4[j]]]+" "+invlabelVocab[actual_labels[j]] + " " + invlabelVocab[pred_labels[j]])
                        predArguementsWord.append(invlabelVocab[pred_labels[j]])
                        fppred.write("\n")
                    fppred.write("\n")
                predArguementsSent.append(predArguementsWord)
            predArguementsDoc.append(predArguementsSent)
        predArguementsTotal.append(predArguementsDoc)

    fppred.close()
    curScore=getF1Score("No","testPredictionsArguements.txt")
    print "F1Score on test data",curScore
    ScoreTest.append(curScore)
    if curScore > globalScore:
        updateGlobalFile("testPredictionsArguements.txt","bestTestPredictionsArguements.txt")
        globalScore=curScore
        extractor=sentenceExtractor()
        for key,predArguementsDoc in zip(test_set,predArguementsTotal):
            extractor.dumpPredictedArguements(key,predArguementsDoc)
    return globalScore
if __name__ == "__main__":
    print "Event Arguement Detection..."
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
    labelVocab={"No":0,"Theme":1,"AtLoc":2,"ToLoc":3,"FromLoc":4,"Site":5,"Cause":6,"Instrument":7}
    entityVocab = mergeVocabs(entityVocabtrain, entityVocabtest)
    entityVocab=mergeVocabs(entityVocab,eventVocab)
    distanceVocab=createDistanceVocab(100)
    inveventVocab = {v: k for k, v in eventVocab.items()}
    invWordVocab = {v: k for k, v in wordVocab.items()}
    inventityVocab = {v: k for k, v in entityVocab.items()}
    invlabelVocab={v: k for k, v in labelVocab.items()}
    invdistanceVocab={v: k for k, v in distanceVocab.items()}
    train_set = dirDicttrain.keys()
    test_set = dirDicttest.keys()
    L1_reg = 0.001
    L2_reg = 0.0001
    learning_rate = 0.01
    nepochs = 30
    globalScore=0
    dropouttrain = np.asarray(0.2, dtype=theano.config.floatX)
    dropouttest = np.asarray(0.0, dtype=theano.config.floatX)
    x0=T.iscalar('x0')
    x1=T.ivector('x1')
    x2=T.ivector('x2')
    x3=T.ivector('x3')
    x4=T.ivector('x4')
    x5=T.dscalar('x5')
    y=T.ivector('y')
    rng = numpy.random.RandomState(1234)
    model=MyArguementModel(rng,wordVocab,entityVocab,labelVocab,distanceVocab,embSizeWord=200,embSizeEntity=100,embSizeDistance=50,RnnHiddenDim=150,FFhiddenLayerDim=300,inputWord=x0,input1=x1,input2=x2,input3=x3,input4=x4,dropout=x5)
    cost = (
        model.negative_log_likelihood(y)
        #+ L1_reg * model.L1
        #+ L2_reg * model.L2_sqr
    )
    gparams = [T.grad(cost, param) for param in model.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(model.params, gparams)
        ]
    # updates=adam(cost,model.params)
    train_model = theano.function(
        inputs=[x0, x1, x2, x3,x4,x5,y],
        outputs=cost,
        updates=updates,
    )
    test_model = theano.function(
        inputs=[x0, x1, x2, x3,x4,x5,y],
        outputs=model.errors(y)
    )
    get_predictions = theano.function(
        inputs=[x0,x1,x2,x3,x4,x5],
        outputs=model.predict()
    )
    for key in test_set:
        if dirDicttrain.has_key(key):
            print key
            exit(0)
    print "Training..."
    for i in range(0, nepochs):
        fppred = open("trainPredictionsArguements.txt", "w")
        shuffle(train_set)
        shuffle(test_set)
        loss=0.0
        error=0.0
        sentenceCount = 0
        for key in train_set:
            #print "\n\n\n*******************", key, "*******************\n\n\n"
            # Retrive the appropriate processed data  and text for the sentence extractor to work
            sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(open(processedpathtrain + key + ".pkl", 'rb'))
            fp = open(corpuspathtrain + key + ".txt", 'r')
            content = fp.read()
            extractor = sentenceExtractor()
            # Extracting the list of entities,arguments and words using sentence extractor
            sentenceList, entityList, _, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                     eventIdDict, eventIndexDict,
                                                                                     content, sentIndex, eventVocab,
                                                                                     wordVocab, removeRareTriggers=True,
                                                                                     concatenate=True)
            sentenceList, entityList, arguementList, sentenceIdList, sentenceRootList, sentenceChunkList, sentencePosList, sentenceParentList, sentenceParseList = extractor.loadParseTrees(key, sentenceList, entityList, arguementList, "train/")
            for index in range(0, len(sentenceList)):
                sentence = sentenceList[index]
                entities = entityList[index]
                arguements = arguementList[index]
                sentenceIds = sentenceIdList[index]
                dependencies = sentenceParseList[index]
                parents = sentenceParentList[index]
                '''print sentence
                print entities
                print sentenceIds
                print dependencies
                print parents'''
                for index1 in range(0, len(sentence)):
                    #word = sentence[index1]
                    entity = entities[index1]
                    wordarguements = arguements[index1]
                    if eventVocab.has_key(entity):
                        depPathvertices, depPathEdges,_ = extractor.getShortestDependencyPath(sentenceIds, sentence,dependencies, parents,str(index1 + 1),entities)
                        #distances=generateArguementFeatures(sentence,index1)
                        distances=generateDepArguementFeatures(sentence,depPathvertices,index1)
                        '''print sentence
                        print entities
                        print distances
                        print wordarguements'''
                        wordarguements, input4 = truncateLableswithRules(wordarguements, entities, entityVocab, entity)
                        # Computes the indexes of the lists extracted above
                        input1,input2,input3,labels=compute(sentence,entities,distances,wordarguements,model)
                        #labels,input4=truncateLabels(labels,input2,inventityVocab)
                        input1,input2,input3,input4,labels,flag=checkLengthAndUpdateInput(input1,input2,input3,input4,labels,wordVocab,entityVocab,distanceVocab,labelVocab,index1)
                        #print labels,input4
                        if len(input4)>0:
                            l=train_model(index1,input1,input2, input3,input4,dropouttrain,labels)
                            er = test_model(index1, input1, input2, input3 ,input4,dropouttest,labels)
                            #pred_labels = get_predictions(index1,input1, input2,input3,input4,dropouttest)
                            viterbi_max, viterbi_argmax=get_predictions(index1,input1, input2,input3,input4,dropouttest)
                            first_ind = np.argmax(viterbi_max[-1])
                            viterbi_pred = backtrack(first_ind, viterbi_argmax)
                            pred_labels = np.array(viterbi_pred)
                            #print pred_labels
                        else:
                            l=0
                            pred_labels=[]
                            er=0
                        actual_labels=labels
                        # Dumps every sentence for getting the F1-Score
                        for j in range(0, len(pred_labels)-flag):
                            fppred.write(
                                invWordVocab[input1[input4[j]]] + "/" + inventityVocab[input2[input4[j]]] + " " + invlabelVocab[
                                    actual_labels[j]] + " " + invlabelVocab[pred_labels[j]])
                            fppred.write("\n")
                        fppred.write("\n")
                        loss += l
                        error +=er
                        sentenceCount += 1
        fppred.close()
        curScore=0.0
        curScore = getF1Score("No", "trainPredictionsArguements.txt")
        print "F1Score on training data", curScore
        print "loss iteration:", i, loss / sentenceCount
        ScoreTrain.append(curScore)
        globalScore = testAndDumpArguements(test_set, globalScore)
    fp = open("ScoresTrainArguements.txt", "w")
    for value in ScoreTrain:
        fp.write(str(value) + "\n")
    fp.close()
    fp = open("ScoresTestArguements.txt", "w")
    for value in ScoreTest:
        fp.write(str(value) + "\n")
    fp.close()




