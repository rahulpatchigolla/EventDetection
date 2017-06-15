import numpy as np
import theano
import theano.tensor as T
import os
from os import listdir
from nltk.tokenize import PunktSentenceTokenizer
import cPickle
import string
import numpy
import sys
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
distanceDict={}
NoDict={}
#Creates random numpy datastructures
def randomMatrix(r, c,):
    W_bound = numpy.sqrt(6. / (r + c))
    #W_bound = 1.
    return np.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)
def randomArray(size):
    return np.random.uniform(
        low=-np.sqrt(6. / np.sum(size)),
        high=np.sqrt(6. / np.sum(size)),
        size=size).astype(theano.config.floatX)
#Converts the words into appropriate indexes based on the vocabulary for trigger identification
def wordsToIndexes(words,vocab,wordflag,eventflag):
    punctuations=string.punctuation
    wordIndexes=[]
    for word in words:
        if wordflag == True:
            while len(word) > 0 and word[-1] in punctuations:
                word = word[:-1]
            while len(word) > 0 and word[0] in punctuations:
                word = word[1:]
            word = word.lower()
        if vocab.has_key(word):
            index=vocab[word]
            wordIndexes.append(index)
        elif wordflag==False and eventflag==False:
            index=vocab['None']
            wordIndexes.append(index)
        else:
            if not eventflag==True:
                print word
                print "Look up error!!!"
                exit()
            else:
                print "Event error!!!"
                exit()
    return wordIndexes
#Converts the words into appropriate indexes based on the vocabulary for argument identification
def wordsToIndexes1(words,vocab,flag1,flag2):
    punctuations = string.punctuation
    wordIndexes = []
    if flag1==True:
        for word in words:
            word=word.split(' ')
            word=word[0]
            while len(word) > 0 and word[-1] in punctuations:
                word = word[:-1]
            while len(word) > 0 and word[0] in punctuations:
                word = word[1:]
            word = word.lower()
            if vocab.has_key(word):
                index = vocab[word]
                wordIndexes.append(index)
            else:
                print word ,"Look up error"
                exit(0)
    else:
        for word in words:
            if vocab.has_key(word):
                index=vocab[word]
                wordIndexes.append(index)
            else:
                word1=word[1:]
                word2=word[0:len(word)-1]
                if vocab.has_key(word1):
                    index = vocab[word1]
                    wordIndexes.append(index)
                elif vocab.has_key(word2):
                    index=vocab[word2]
                    wordIndexes.append(index)
                else:
                    if flag2==False:
                        print word,"Look up error"
                        exit(0)
                    else:
                        index=vocab["None"]
                        wordIndexes.append(index)
    return wordIndexes

#loads the word embeddings from the binary file mentioned in my paper
def loadWordEmbeddings(wordVocab,embSize):
    print "Loading Word Embeddings..."
    print "Total Words:", len(wordVocab)
    model = Word2Vec.load_word2vec_format('/home/rahul/PycharmProjects/PubMed-w2v.bin', binary=True)
    wordemb = []
    count = 0
    for word in wordVocab:
        if model.__contains__(word):
            wordemb.append(model[word])
        else:
            count += 1
            #print word
            wordemb.append(np.random.rand(embSize))
            #	print (wordemb)
    # wordemb = np.asarray(map(float, wordemb))
    wordemb[wordVocab["UNK"]] = np.zeros(embSize)
    wordemb = np.asarray(wordemb, dtype=theano.config.floatX)
    print ("number of unknown word in word embedding", count)
    return wordemb
#method for reading files from the directory
def readDir(path):
    dirList=listdir(path)
    dirDict = {}
    for file in dirList:
        id = file.split('.')
        if (not dirDict.has_key(id[0])):
            dirDict[id[0]] = 0
    return dirDict
#method for getting F1-score from using connll script
def getF1Score(otherWord,filename):
    import commands
    output = commands.getoutput("perl connlleval.pl -l -r -o "+otherWord+"<"+filename)
    output = output.split('\n')
    output = output[-1]
    output = output.split(' ')
    return  float(output[-2])
#method to update the current result with the best result
def updateGlobalFile(file,globalfile):
    f=open(file,"r")
    fw=open(globalfile,"w")
    for line in f.readlines():
        fw.write(line)
    f.close()
    fw.close()
#method for merging train and test vocabularies
def mergeVocabs(trainVocab,testVocab,removeRareEvents=False):
    rareEvents = {"Phosphorylation": "", "Synthesis": "", "Transcription": "", "Catabolism": "",
                  "Dephosphorylation": "","Remodeling":""}
    for key in testVocab.keys():
        if not trainVocab.has_key(key):
            trainVocab[key]=0
    i=0
    if removeRareEvents==True:
        for key in trainVocab.keys():
            if rareEvents.has_key(key):
                del trainVocab[key]
    for key in trainVocab.keys():
        trainVocab[key]=i
        i+=1
    return trainVocab
#method to convert a list into a string using the appropriate delimiter
def convertToSentence(sentenceList,delimiter):
    sentence=""
    for word in sentenceList:
        sentence+=word
        sentence+=delimiter
    return sentence
#method to generate normal distance between trigger word and the entities
def generateArguementFeatures(sentence,indexWord):
    distances=[]
    for index in range(0,len(sentence)):
        distances.append(abs(indexWord-index))
    return distances
#method to generate dependency path based distance between trigger word and the entities
def generateDepArguementFeatures(sentence,depPathVertices,indexWord):
    distances=[]
    assert len(depPathVertices)==len(sentence)
    for index in range(0,len(depPathVertices)):
        if index<=indexWord:
            distances.append(-len(depPathVertices[index]))
        else:
            distances.append(len(depPathVertices[index]))
    return distances
# method to create distance vocabulary
def createDistanceVocab(maxval):
    distanceVocab={}
    index=0
    for i in range(0,maxval):
        distanceVocab[i]=index
        index+=1
    for i in range(1,maxval):
        distanceVocab[-i]=index
        index+=1
    return distanceVocab
# removing invalid arguments w.r.t the trigger word
def truncateLabels(input,entities,vocab):
    rval=[]
    indexList=[]
    index=0
    for label,entity in zip(input,entities):
        if not vocab[entity]=="None":
            rval.append(label)
            indexList.append(index)
        index+=1
    return rval,indexList
# add back the removed arguments w.r.t the trigger word
def reconstructLabels(input,entities):
    rval=[]
    index=0
    for entity in entities:
        if entity=="None":
            rval.append("No")
        else:
            rval.append(input[index])
            index+=1
    return rval
# add back the removed arguments w.r.t the trigger word with more filtering rules
def reconstructLabelsWithRulesProcedure(input,entities,dictionary):
    rval=[]
    index=0
    for entity in entities:
        if dictionary.has_key(entity):
            rval.append(input[index])
            index+=1
        else:
            rval.append("No")
    return rval
def reconstructLabelsWithRules(input,entities,vocab,triggerlabel):
    anotomicalTriggerDict = {'Cell_proliferation': 0, 'Development': 0, 'Blood_vessel_development': 0, 'Growth': 0,
                             'Death': 0, 'Breakdown': 0, 'Remodeling': 0}
    anotomicalEntityDict = {'Organism': 0, 'Organism_subdivision': 0, 'Anatomical_system': 0, 'Organ': 0,
                            'Multi-tissue_structure': 0, 'Tissue': 0, 'Cell': 0, 'Cellular_component': 0,
                            'Developing_anatomical_structure': 0, 'Organism_substance': 0,
                            'Immaterial_anatomical_entity': 0, 'Pathological_formation': 0}
    molecularTriggerDict = {'Synthesis': 0, 'Gene_expression': 0, 'Transcription': 0, 'Catabolism': 0,
                            'Phosphorylation': 0, 'Dephosphorylation': 0}
    molecularEntityDict = {'Drug_or_compound': 0, 'Gene_or_gene_product': 0}

    allEntityDict = {'Organism': 0, 'Organism_subdivision': 0, 'Anatomical_system': 0, 'Organ': 0,
                     'Multi-tissue_structure': 0, 'Tissue': 0, 'Cell': 0, 'Cellular_component': 0,
                     'Developing_anatomical_structure': 0, 'Organism_substance': 0, 'Immaterial_anatomical_entity': 0,
                     'Pathological_formation': 0, 'Drug_or_compound': 0, 'Gene_or_gene_product': 0}
    remNonRecursiveDict = {'Binding': 0, 'Planned_process': 0, 'Localization': 0}
    recursveTriggerDict = {'Regulation': 0, 'Positive_regulation': 0, 'Negative_regulation': 0}
    if anotomicalTriggerDict.has_key(triggerlabel):
        if triggerlabel == 'Cell_proliferation':
            return reconstructLabelsWithRulesProcedure(input, entities, {'Cell': 0})
        else:
            return reconstructLabelsWithRulesProcedure(input, entities, anotomicalEntityDict)
    elif molecularTriggerDict.has_key(triggerlabel):
        return reconstructLabelsWithRulesProcedure(input, entities, molecularEntityDict)
    elif remNonRecursiveDict.has_key(triggerlabel):
        return reconstructLabelsWithRulesProcedure(input, entities, allEntityDict)
    elif recursveTriggerDict.has_key(triggerlabel):
        return reconstructLabelsWithRulesProcedure(input, entities, vocab)
    else:
        print triggerlabel
        print "Not a valid trigger label"
        exit(0)
# removing invalid arguments w.r.t the trigger word with more filtering rules
def truncateWithRuleProceduce(input,entities,ruleDict):
    rval=[]
    indexList=[]
    index=0
    for label,entity in zip(input,entities):
        if ruleDict.has_key(entity) and not entity=='None':
            rval.append(label)
            indexList.append(index)
        index+=1
    return rval,indexList

def truncateLableswithRules(input,entities,vocab,triggerlabel):
    #print "merged",entities
    anotomicalTriggerDict={'Cell_proliferation': 0,'Development': 0, 'Blood_vessel_development': 0,'Growth': 0,'Death': 0,'Breakdown': 0,'Remodeling': 0}
    anotomicalEntityDict={'Organism':0,'Organism_subdivision':0,'Anatomical_system':0,'Organ':0,'Multi-tissue_structure':0,'Tissue':0,'Cell':0,'Cellular_component':0,'Developing_anatomical_structure':0,'Organism_substance':0,'Immaterial_anatomical_entity':0,'Pathological_formation':0}
    molecularTriggerDict={'Synthesis': 0,'Gene_expression': 0,'Transcription': 0,'Catabolism': 0,'Phosphorylation': 0,'Dephosphorylation': 0}
    molecularEntityDict={'Drug_or_compound':0,'Gene_or_gene_product': 0}
    allEntityDict={'Organism':0,'Organism_subdivision':0,'Anatomical_system':0,'Organ':0,'Multi-tissue_structure':0,'Tissue':0,'Cell':0,'Cellular_component':0,'Developing_anatomical_structure':0,'Organism_substance':0,'Immaterial_anatomical_entity':0,'Pathological_formation':0,'Drug_or_compound':0,'Gene_or_gene_product': 0}
    remNonRecursiveDict={'Binding': 0,'Planned_process': 0,'Localization': 0}
    recursveTriggerDict={ 'Regulation': 0, 'Positive_regulation': 0,'Negative_regulation': 0}
    if anotomicalTriggerDict.has_key(triggerlabel):
        if triggerlabel=='Cell_proliferation':
            return truncateWithRuleProceduce(input,entities,{'Cell':0})
        else:
            return truncateWithRuleProceduce(input,entities,anotomicalEntityDict)
    elif molecularTriggerDict.has_key(triggerlabel):
        return truncateWithRuleProceduce(input,entities,molecularEntityDict)
    elif remNonRecursiveDict.has_key(triggerlabel):
        return truncateWithRuleProceduce(input,entities,allEntityDict)
    elif recursveTriggerDict.has_key(triggerlabel):
        return truncateWithRuleProceduce(input,entities,vocab)
    else:
        print triggerlabel
        print "Not a valid trigger label"
        exit(0)
# sub procedure for mergeMultiWordTriggers
def generateMergingList(events):
    newindex = 0
    indexList = []
    for index in range(0, len(events)):
        word = events[index]
        if index == 0:
            if word == "Other":
                indexList.append(newindex)
            else:
                indexList.append(newindex)
        else:
            if word == "Other":
                newindex += 1
                indexList.append(newindex)
            else:
                prevword = events[index - 1]
                if prevword == word :
                    indexList.append(newindex)
                else:
                    newindex += 1
                    indexList.append(newindex)
    return indexList
# merging multi word triggers for correct evalution sub procedure for post processing
def mergeMultiWordTriggers(words,actual,predicted):
    indexList=generateMergingList(actual)
    assert(len(words)==len(actual)==len(predicted)==len(indexList))
    newwords=[]
    newactual=[]
    newpredicted=[]
    newindex = 0
    newword = ""
    newactuallabel=""
    newpredictedlabel=""
    for word,actuallabel,predictedlabel,index in zip(words,actual,predicted,indexList):
        if newindex == index:
            if newword=="":
                newword=word
                newactuallabel=actuallabel
                newpredictedlabel=predictedlabel
            else:
                newword+="_"
                newword+=word
                if not newactuallabel==actuallabel:
                    print "Something is wrong!!"
                    exit(0)
                if newactuallabel==predictedlabel:
                    newpredictedlabel=newactuallabel
        else:
            newwords.append(newword)
            newactual.append(newactuallabel)
            newpredicted.append(newpredictedlabel)
            newword=word
            newactuallabel=actuallabel
            newpredictedlabel=predictedlabel
            newindex+=1
    newwords.append(newword)
    newactual.append(newactuallabel)
    newpredicted.append(newpredictedlabel)
    assert len(newwords)==len(newactual)==len(newpredicted)
    return newwords,newactual,newpredicted


# The postprocessing procedure where we merge multi word triggers for getting the correct F1-Score
def postProcessResults(filename):
    fp=open(filename,"r")
    finalLinesWords=[]
    finalLinesActual = []
    finalLinesPredicted = []
    finalLineWords=[]
    finalLineActual = []
    finalLinePredicted = []
    for line in fp:
        if line=="\n":
            finalLineWords,finalLineActual,finalLinePredicted=mergeMultiWordTriggers(finalLineWords,finalLineActual,finalLinePredicted)
            #print finalLineWords
            #print finalLineActual
            #print finalLinePredicted
            #print "\n"
            finalLinesWords.append(finalLineWords)
            finalLinesActual.append(finalLineActual)
            finalLinesPredicted.append(finalLinePredicted)
            finalLineWords = []
            finalLineActual = []
            finalLinePredicted = []
        else:
            line=line.strip('\n').split(' ')
            finalLineWords.append(line[0])
            finalLineActual.append(line[1])
            finalLinePredicted.append(line[2])

    fp.close()
    fpwrite=open(filename,"w")
    for line,actual,predicted in zip(finalLinesWords,finalLinesActual,finalLinesPredicted):
        for word,actuallabel,predictedlabel in zip(line,actual,predicted):
            fpwrite.write(word+" "+actuallabel+" "+predictedlabel+"\n")
        fpwrite.write("\n")
    fpwrite.close()
# Procedure where it corrects regulation labels if it is near a postive or negative regulation label
def postProcessPredictedLabels(labels,invvocab):
    labels=[invvocab[label] for label in labels]
    for index in range(0,len(labels)):
        label=labels[index]
        if label=="Regulation":
            tempindex=index+1
            replacelabel="Regulation"
            while tempindex<len(labels) and (labels[tempindex]=="Positive_regulation" or labels[tempindex]=="Negative_regulation"):
                replacelabel = labels[tempindex]
                tempindex+=1
            updateindex=index
            while updateindex<tempindex:
                labels[updateindex] = replacelabel
                updateindex+=1
    labels=labels[::-1]
    for index in range(0,len(labels)):
        label=labels[index]
        if label=="Regulation":
            tempindex=index+1
            replacelabel="Regulation"
            while tempindex<len(labels) and (labels[tempindex]=="Positive_regulation" or labels[tempindex]=="Negative_regulation"):
                replacelabel = labels[tempindex]
                tempindex+=1
            updateindex=index
            while updateindex<tempindex:
                labels[updateindex] = replacelabel
                updateindex+=1
    labels = labels[::-1]
    return labels
# subprocedure for viterbi procedure
def backtrack(first, argmax):
    tokens = [first]
    for i in xrange(argmax.shape[0]-1, -1, -1):
        tokens.append(argmax[i, tokens[-1]])
    return tokens[::-1]

# explicit handling for length one inputs
def checkLengthAndUpdateInput(input1,input2,input3,input4,labels,wordVocab,entityVocab,distanceVocab,labelVocab,index):
    if len(input4)==1:
        #print "appending"
        input1.append(wordVocab['UNK'])
        input2.append(entityVocab['None'])
        input3.append(distanceVocab[len(input1)]-index-1)
        input4.append(len(input1)-1)
        labels.append(labelVocab['No'])
        return input1, input2,input3, input4, labels,1
    else:
        return input1,input2,input3,input4,labels,0
