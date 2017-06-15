import cPickle
import string
from Utils import readDir
from nltk.tokenize import PunktSentenceTokenizer
from SentenceExtractor import *
trainpath="./Preprocessed_Corpus_train/"
#This file generates the pickle files(train data) which store the representational information regarding the data for sentence extractor to work
def ExtractEntities(key):
    entityIndexDict={}
    entityIdDict={}
    fileName=key
    fileName+=".a1"
    fp=open(fileName,"r")
    line=fp.readline()
    line=line.replace('\n','')
    while line:
        line=line.split('\t')
        #print line
        if(entityIdDict.has_key(line[0])):
            print "Duplicate key error!!! in entityIdDict"
            exit(0)
        else:
            entityIdDict[line[0]]=line[1]
        indexLine=line[1].split(' ')
        #print indexLine
        indexLine[2]+=" "
        indexLine[2]+=indexLine[0]
        if(entityIndexDict.has_key(int(indexLine[1]))):
            tempList=entityIndexDict[int(indexLine[1])]
            tempList.append(indexLine[2])
            entityIndexDict[int(indexLine[1])]=tempList
        else:
            tempList=[]
            tempList.append(indexLine[2])
            entityIndexDict[int(indexLine[1])]=tempList
        line=fp.readline()
        line=line.replace('\n','')
    fileName=key
    fp.close()
    fileName+=".a2"
    fp=open(fileName,"r")
    line=fp.readline()
    line=line.replace('\n','')
    while line:
        if line[0]=='T':
            line=line.split('\t')
            #print line
            if(entityIdDict.has_key(line[0])):
                print "Duplicate key error!!! in entityIdDict"
                exit(0)
            else:
                entityIdDict[line[0]]=line[1]
            indexLine=line[1].split(' ')
            #print indexLine
            indexLine[2]+=" "
            indexLine[2]+=indexLine[0]
            if(entityIndexDict.has_key(int(indexLine[1]))):
                tempList=entityIndexDict[int(indexLine[1])]
                tempList.append(indexLine[2])
                entityIndexDict[int(indexLine[1])]=tempList
            else:
                tempList=[]
                tempList.append(indexLine[2])
                entityIndexDict[int(indexLine[1])]=tempList
        if line[0]=='E':
            line=line.split('\t')
            triggerline=line[1].split(' ')
            triggerline=triggerline[0].split(':')
            entityIdDict[line[0]]=triggerline[1]

        line=fp.readline()
        line=line.replace('\n','')
    fp.close()
    return entityIdDict,entityIndexDict
def printEntitiesAndEvents(fileName,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex):
    for index in range(0,len(sentIndex)):
        print "*******************sentence*******************"
        entityList = []
        ListOfevents = []
        if index==0:
            print content[0:sentIndex[index]]
            for i in range(0,sentIndex[index]):
                if(entityIndexDict.has_key(i)):
                    tempList = entityIndexDict[i]
                    for entity in tempList:
                        entity = entity.split(' ')
                        startindex = i
                        endindex = int(entity[0])
                        res = content[startindex:endindex]
                        res += ":"
                        res += entity[1]
                        entityList.append(res)
                if (eventIndexDict.has_key(i)):
                    eventId = eventIndexDict[i]
                    eventId=eventId[0]
                    event = eventIdDict[eventId]
                    event = event.split(' ')
                    eventList=[]
                    for atom in event:
                        if len(atom) == 0:
                            continue
                        atom = atom.split(':')
                        key = atom[1]
                        if (key[0] == 'T'):
                            if (not entityIdDict.has_key(atom[1])):
                                print "entity Id not found !!!"
                                exit(0)
                            else:
                                entity = entityIdDict[atom[1]]
                                entity = entity.split(' ')
                                word = content[int(entity[1]):int(entity[2])]
                                word += "("
                                word += entity[0]
                                word += ")"
                        else:
                            None
                        eventList.append(atom[0]+":"+word)
                    ListOfevents.append(eventList)

        else:
            print content[sentIndex[index-1]:sentIndex[index]]
            for i in range(sentIndex[index-1], sentIndex[index]):
                if (entityIndexDict.has_key(i)):
                    tempList = entityIndexDict[i]
                    for entity in tempList:
                        entity = entity.split(' ')
                        startindex = i
                        endindex = int(entity[0])
                        res = content[startindex:endindex]
                        res += ":"
                        res += entity[1]
                        entityList.append(res)
                        if (eventIndexDict.has_key(i)):
                            eventId = eventIndexDict[i]
                            eventId=eventId[0]
                            event = eventIdDict[eventId]
                            event = event.split(' ')
                            eventList = []
                            for atom in event:
                                if len(atom) == 0:
                                    continue
                                atom = atom.split(':')
                                key = atom[1]
                                if (key[0] == 'T'):
                                    if (not entityIdDict.has_key(atom[1])):
                                        print "entity Id not found !!!"
                                        exit(0)
                                    else:
                                        entity = entityIdDict[atom[1]]
                                        entity = entity.split(' ')
                                        word = content[int(entity[1]):int(entity[2])]
                                        word += "("
                                        word += entity[0]
                                        word += ")"
                                else:
                                    None
                                eventList.append(atom[0] + ":" + word)
                            ListOfevents.append(eventList)
        print "Entities:"
        for entity in entityList:
            print entity
        print '\n'
        print "Events:"
        for event in ListOfevents:
            print event
        print '\n'
    cPickle.dump([sentIndex,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict],open("./Preprocessed_Corpus_train/"+fileName+".pkl","wb"))

def ExtractEvents(key):
    eventIndexDict={}
    eventIdDict={}
    eventTriggerDict={}
    fileName=key
    fileName+=".a2"
    fp=open(fileName)
    line=fp.readline()
    line=line.replace('\n','')
    while line:
        if line[0]=='E':
            line = line.split('\t')
            eventIdDict[line[0]]=line[1]
            trigger=line[1].split(' ')
            trigger=trigger[0].split(':')
            if(not entityIdDict.has_key(trigger[1])):
                print "Event not found in dictionary"
                exit(0)
            else:
                triggerval=entityIdDict[trigger[1]]
                triggerindex=triggerval.split(' ')
                if eventIndexDict.has_key(int(triggerindex[1])):
                    listevents=eventIndexDict[int(triggerindex[1])]
                    listevents.append(line[0])
                else:
                    listevents=[]
                    listevents.append(line[0])
                    eventIndexDict[int(triggerindex[1])]=listevents
        line=fp.readline()
        line=line.replace('\n','')
    fp.close()
    return eventIdDict,eventIndexDict
def getSentIndex(sentences):
    sentIndex=[]
    for sentence in sentences:
        sentIndex.append(len(sentence))
    for index in range(0,len(sentIndex)):
        if index==0:
            continue
        elif index==1:
            sentIndex[index]=sentIndex[index-1]+sentIndex[index]+2
        else:
            sentIndex[index]=sentIndex[index-1]+sentIndex[index]+1
    return sentIndex
def buildWordVocab(dirDict,path):
    vocab={}
    dummy={}
    for key in dirDict.keys():
        print key
        fileName=key+".txt"
        fp=open(path+fileName,"r")
        content=fp.read()
        extractor = sentenceExtractor()
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(trainpath + key + ".pkl", 'rb'))
        sentences, _, _, _ = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,eventIdDict, eventIndexDict,content, sentIndex, eventVocab,dummy,removeRareTriggers=True)
        punctuations=string.punctuation
        for sentence in sentences:
            for word in sentence:
                while len(word)>0 and word[-1] in punctuations:
                    word = word[:-1]
                while len(word)>0 and word[0] in punctuations:
                    word = word[1:]
                word=word.lower()
                if not vocab.has_key(word):
                    #print word
                    vocab[word]=0
    print "Word Vocabulary size",len(vocab)
    i=0
    for w in vocab.keys():
        vocab[w]=i
        i+=1
    vocab['UNK']=i
    return vocab
def buildEntityVocab(entityIdDict,entityVocab):
    for key,value in entityIdDict.items():
        value=value.split(' ')
        if key[0]=='T':
            if not entityVocab.has_key(value[0]):
                entityVocab[value[0]]=0
    return entityVocab

def buildEntityVocabFinal(entityVocab,eventVocab):
    i=0
    rareEventVocab={'Dissociation': 0,'Protein_processing':0,'Pathway': 0,'Cell_division': 0,'Metabolism': 0,'Translation': 0,'Ubiquitination': 0,'Acetylation': 0,'Reproduction': 0,'DNA_methylation': 0,'Protein_domain_or_region':0,'DNA_domain_or_region': 0}
    keys = eventVocab.keys() + rareEventVocab.keys()
    for key in keys:
        if (eventVocab.has_key(key) or rareEventVocab.has_key(key)) and not key=='Other':
            if entityVocab.has_key(key):
                #print "removing",key
                del entityVocab[key]
    for w in entityVocab.keys():
        entityVocab[w]=i
        i+=1
    entityVocab["None"]=i
    return entityVocab
def buildEventVocab():
    eventVocab={'Development':0,'Growth':1,'Remodeling':2,'Breakdown':3,
                'Death':4,'Cell_proliferation':5,'Blood_vessel_development':6,
                'Localization':7,'Binding':8,'Catabolism':9,'Synthesis':10,
                'Gene_expression':11,'Phosphorylation':12,'Dephosphorylation':13,
                'Transcription':14,'Regulation':15,'Positive_regulation':16,
                'Negative_regulation':17,'Planned_process':18,'Other':19
                }
    return eventVocab
if __name__ == "__main__":
    path="./Corpus_filtered/train/"
    dirDict=readDir(path)
    for key,value in dirDict.items():
        print key
    entityVocab={}
    #print wordVocab
    for key in dirDict.keys():
        #key="PMID-15975645"
        print "*******************",key,"*******************\n\n\n"
        fileName=key
        entityIndexDict={}
        entityIdDict={}
        eventIdDict={}
        eventIndexDict={}
        entityIdDict,entityIndexDict=ExtractEntities(path+fileName)
        eventIdDict, eventIndexDict = ExtractEvents(path+fileName)
        '''print "entityidDict"
        for key,value in entityIdDict.items():
            print key,":",value
        print "\nentityindexDict\n"
        for key,value in entityIndexDict.items():
            print key,":",value
        print "eventidDict"
        for key,value in eventIdDict.items():
            print key,":",value
        print "\neventindexDict\n"
        for key,value in eventIndexDict.items():
            print key,":",value'''
        entityVocab=buildEntityVocab(entityIdDict,entityVocab)
        fp=open(path+fileName+".txt","r")
        content=fp.read()
        #print content
        tokenizer = PunktSentenceTokenizer(content)
        sentences=tokenizer.tokenize(content)
        index=0
        #print sentences
        sentIndex=getSentIndex(sentences)
        '''print sentIndex,len(sentIndex)
        print len(content)'''
        printEntitiesAndEvents(fileName,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex)
        fp.close()
    eventVocab = buildEventVocab()
    entityVocab=buildEntityVocabFinal(entityVocab,eventVocab)
    wordVocab = buildWordVocab(dirDict, path)
    cPickle.dump(entityVocab, open("./Preprocessed_Corpus_train/entityVocab.pkl", "wb"))
    cPickle.dump(eventVocab, open("./Preprocessed_Corpus_train/eventVocab.pkl", "wb"))
    cPickle.dump(wordVocab, open("./Preprocessed_Corpus_train/wordVocab.pkl", "wb"))
    print "EntityVocabulary size:",len(entityVocab)
    print "EventVocabulary size",len(eventVocab)
    print "WordVocabulary size", len(wordVocab)
    print eventVocab
    print entityVocab
    print wordVocab