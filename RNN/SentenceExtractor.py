from Utils import *
from sets import ImmutableSet
import copy
import networkx as nx
class sentenceExtractor:
    def __init__(self):
        # vocabulary of rare events
        self.rareEvents = {"Phosphorylation": "", "Synthesis": "", "Transcription": "", "Catabolism": "",
                  "Dephosphorylation": "","Remodeling":""}

    # Extracting the list of entities,events,arguments,words and their indexes based on the processed data
    def entitiesAndEvents(self,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,content,sentIndex,eventVocab,wordVocab,removeRareTriggers=False,concatenate=False,loadpredTriggers=False,key="",getindexlist=False):
        sentenceList = []
        entityList = []
        eventList = []
        arguementList=[]
        docIndexList=[]
        '''print entityIdDict
        print "\n"
        print entityIndexDict
        print "\n"
        print sentIndex'''
        for index in range(0,len(sentIndex)):
            # Processing step for 1st line in the data
            if index==0:
                sentence=content[0:sentIndex[index]]
                indexList, wordList=self.assignIndextoWord(sentence,0,0,entityIndexDict,eventIndexDict)
                '''print indexList
                print wordList'''
                entities=["None"]*len(indexList)
                events=["Other"]*len(indexList)
                '''i = 0
                for index in indexList:
                    print content[int(index)], wordList[i]
                    i += 1'''
                for wordIndex in range(0,len(indexList)):
                    if entityIndexDict.has_key(indexList[wordIndex]):
                        entity = entityIndexDict[indexList[wordIndex]]
                        entity=entity[0]
                        entity=entity.split(' ')
                        startindex=indexList[wordIndex]
                        endindex=int(entity[0])
                        words=content[startindex:endindex]
                        words=words.split(' ')
                        j=wordIndex
                        for word in words:
                            entities[j]=entity[1]
                            j+=1
                for wordIndex in range(0,len(indexList)):
                    if eventIndexDict.has_key(indexList[wordIndex]):
                        eventId = eventIndexDict[indexList[wordIndex]]
                        eventId=eventId[0]
                        event = eventIdDict[eventId]
                        event = event.split(' ')
                        event=event[0]
                        event=event.split(':')
                        trigger=event[1]
                        if trigger[0]=='T':
                            if(not entityIdDict.has_key(trigger)):
                                print "entity Id not found !!!!"
                                exit(0)
                            else:
                                entity = entityIdDict[trigger]
                                entity = entity.split(' ')
                                startindex=indexList[wordIndex]
                                endindex=int(entity[2])
                                words=content[startindex:endindex]
                                words=words.split(' ')
                                j = wordIndex
                                for word in words:
                                    events[j] = event[0]
                                    j += 1
                        else:
                            print "Invalid format !!!"
                            exit(0)

            else:
                # Processing step for remaining lines
                sentence = content[sentIndex[index-1]:sentIndex[index]]
                if index==1:
                    indexList, wordList = self.assignIndextoWord(sentence, sentIndex[index-1], 2,entityIndexDict,eventIndexDict)
                else:
                    indexList, wordList = self.assignIndextoWord(sentence, sentIndex[index-1], 1,entityIndexDict,eventIndexDict)
                '''print indexList
                print wordList'''
                entities = ["None"] * len(indexList)
                events = ["Other"] * len(indexList)
                '''i = 0
                for index in indexList:
                    print content[int(index)], wordList[i]
                    i += 1'''
                for wordIndex in range(0, len(indexList)):
                    if entityIndexDict.has_key(indexList[wordIndex]):
                        entity = entityIndexDict[indexList[wordIndex]]
                        entity = entity[0]
                        entity = entity.split(' ')
                        startindex = indexList[wordIndex]
                        endindex = int(entity[0])
                        words = content[startindex:endindex]
                        words = words.split(' ')
                        j = wordIndex
                        for word in words:
                            entities[j] = entity[1]
                            j += 1
                for wordIndex in range(0,len(indexList)):
                    if eventIndexDict.has_key(indexList[wordIndex]):
                        eventId = eventIndexDict[indexList[wordIndex]]
                        eventId=eventId[0]
                        event = eventIdDict[eventId]
                        event = event.split(' ')
                        event=event[0]
                        event=event.split(':')
                        trigger=event[1]
                        if trigger[0]=='T':
                            if(not entityIdDict.has_key(trigger)):
                                print "entity Id not found !!!!"
                                exit(0)
                            else:
                                entity = entityIdDict[trigger]
                                entity = entity.split(' ')
                                startindex=indexList[wordIndex]
                                endindex=int(entity[2])
                                words=content[startindex:endindex]
                                words=words.split(' ')
                                j = wordIndex
                                for word in words:
                                    events[j] = event[0]
                                    j += 1

                        else:
                            print "Invalid format !!!"
                            exit(0)
                '''print wordList
                print entities
                print events
                print indexList'''
            events = self.removeRareTriggerWords(events, eventVocab, removeRareTriggers)
            arguements = []
            indexmap={}
            for wordIndex in range(0, len(indexList)):
                indexmap[indexList[wordIndex]]=wordIndex
            for wordIndex in range(0,len(indexList)):
                currentarguement=[]
                #currentarguement = ["No"] * len(indexList)
                if not events[wordIndex]=="Other":
                    currentarguement = ["No"] * len(indexList)
                    if not eventIndexDict.has_key(indexList[wordIndex]):
                        currentarguement=[]
                        arguements.append(currentarguement)
                        #print "Event crossing sentence!!!"
                        continue
                    eventId=eventIndexDict[indexList[wordIndex]]
                    for id in eventId:
                        event=eventIdDict[id]
                        event=event.split(' ')
                        #print event
                        event=event[1:]
                        #print event
                        for arg in event:
                            arg=arg.split(':')
                            #print arg
                            if len(arg)>1:
                                if arg[1][0]=='T':
                                    entity=entityIdDict[arg[1]]
                                    #print entity
                                    entity=entity.split(' ')
                                    startindex=int(entity[1])
                                    endindex=int(entity[2])
                                    for index in range(startindex,endindex):
                                        if indexmap.has_key(index):
                                            currentarguement[indexmap[index]]=arg[0]
                                else:
                                    event=eventIdDict[arg[1]]
                                    #print event
                                    event=event.split(' ')
                                    event=event[0]
                                    event=event.split(':')
                                    entity = entityIdDict[event[1]]
                                    entity=entity.split(' ')
                                    #print entity
                                    startindex=int(entity[1])
                                    endindex=int(entity[2])
                                    for index in range(startindex,endindex):
                                        if indexmap.has_key(index):
                                            currentarguement[indexmap[index]]=arg[0]
                arguements.append(currentarguement)
            sentenceList.append(wordList)
            entityList.append(entities)
            eventList.append(events)
            docIndexList.append(indexList)
            if len(arguements) != len(indexList):
                print len(arguements), len(indexList)
                print "Arguement size mismatch"
                exit(0)
            arguementList.append(arguements)
            if (not len(indexList)==len(wordList)) or (not len(indexList)==len(entities) or (not len(events)==len(wordList))):
                print "Some thing went wrong !!!!"
                exit(0)
        self.correctList(entityList,eventList,eventVocab)#should take care of the correction
        #self.printList(sentenceList,entityList,eventList,arguementList)
        #self.dumpTriggers(sentenceList,eventList,wordVocab,eventVocab)
        if concatenate==True:
            if loadpredTriggers==True:
                #print "Old",eventList
                eventList=self.loadPredictedTriggers(key)
                #print "New",eventList
                for events,sentence,sentIndexList in zip(eventList,sentenceList,docIndexList):
                    assert len(events)==len(sentence)==len(sentIndexList)
            #self.printList1(sentenceList, entityList, arguementList, indexList, docIndexList, eventVocab)

            sentenceList,entityList,arguementList,indexList,docIndexList=self.concatenateWordsAndEntities(sentenceList,entityList,eventList,arguementList,docIndexList,eventVocab)
            #self.printList(sentenceList,entityList,[],arguementList,flag=True)
        if getindexlist==True:
            return sentenceList, entityList, eventList, arguementList,docIndexList
        return sentenceList,entityList,eventList,arguementList

    # The custom word tokenising process based on the data is done here
    def assignIndextoWord(self,sentence,index,flag,entityIndexDict,eventIndexDict):
        indexList=[]
        wordList=sentence.strip().split(' ')
        i=flag
        for word in wordList:
            indexList.append(flag)
            flag+=len(word)+1
        for i in range(0,len(indexList)):
            indexList[i]+=index
        newwordList=[]
        newindexList=[]
        for word,index in zip(wordList,indexList):
            high = index + len(word) - 1
            low=index
            curr=low
            tempindexlist=[]
            tempwordlist=[]
            while curr<=high:
                if entityIndexDict.has_key(curr) or eventIndexDict.has_key(curr) or curr==low:
                    tempindexlist.append(curr)
                curr+=1
            wordindex=0
            for iterator in range(0,len(tempindexlist)):
                if iterator==len(tempindexlist)-1:
                    tempwordlist.append(word[wordindex:])
                else:
                    low=tempindexlist[iterator]
                    high=tempindexlist[iterator+1]-1
                    tempwordlist.append(word[wordindex:wordindex+high-low+1])
                    wordindex+=high-low+1
            assert(len(tempwordlist)==len(tempindexlist))
            newwordList+=tempwordlist
            newindexList+=tempindexlist
        indexList=newindexList
        wordList=newwordList
        newindexList=[]
        newwordList=[]
        for word,index in zip(wordList,indexList):
            high = index + len(word) - 1
            low = index
            curr = low
            tempindexlist = []
            tempwordlist = []
            if entityIndexDict.has_key(curr) or eventIndexDict.has_key(curr):
                length =0
                if entityIndexDict.has_key(curr):
                    length=self.getLength(curr,entityIndexDict)
                else:
                    length=self.getLength(curr,eventIndexDict)
                tempindexlist=[curr]
                tempwordlist=[word[0:length]]
                newindexList+=tempindexlist
                newwordList+=tempwordlist
            else:
                newindexList.append(index)
                newwordList.append(word)

        return newindexList,newwordList
    # removes rare events from the data and corrects the lists
    def correctList(self,entityList,eventList,eventVocab):
        for gindex in range(0,len(entityList)):
            entities=entityList[gindex]
            events=eventList[gindex]
            for index in range(0,len(entities)):
                if eventVocab.has_key(entities[index]) or self.rareEvents.has_key(entities[index]):
                    entities[index]="None"
    # Procedure for printing the things done by the sentence extractor
    def printList1(self,sentenceList,entityList,arguementsList,indexList,docIndexList,eventDict):
        for index in range(0,len(sentenceList)):
            print "\n******************sentence*********************\n"
            print "sentence:", sentenceList[index]
            print "entities:", entityList[index]
            print "docIndexes", docIndexList[index]
            #print "Indexes:", indexList[index]
            entities=entityList[index]
            sentence = sentenceList[index]
            arguements = arguementsList[index]
            for index1 in range(0,len(arguements)):
                if eventDict.has_key(entities[index1]):
                    print "trigger word:",sentence[index1],"event:",entities[index1]
                    print arguements[index1]

    # Procedure to remove rare trigger words
    def removeRareTriggerWords(self,events,eventVocab,removeRareTriggers):
        for index in range(0,len(events)):
            if removeRareTriggers==True:
                if self.rareEvents.has_key(events[index]) or not eventVocab.has_key(events[index]):
                    events[index]="Other"
            else:
                if not eventVocab.has_key(events[index]):
                    events[index]="Other"
        return events

    # procedure for merging entity list with event trigger list
    def generateMergingList(self,entities,events,arguements):
        newindex=0
        indexList=[]
        for index in range(0,len(entities)):
            if not events[index]=="Other":
                if not entities[index]=="None":
                    None
                    print "replacing",entities[index],"with",events[index]
                entities[index]=events[index]
        for index in range(0, len(entities)):
            word = entities[index]
            if index == 0:
                if word == "None":
                    indexList.append(newindex)
                else:
                    indexList.append(newindex)
            else:
                if word == "None":
                    newindex += 1
                    indexList.append(newindex)
                else:
                    prevword = entities[index - 1]
                    if prevword == word and (arguements[index-1]==arguements[index] or len(arguements[index])==0):
                        indexList.append(newindex)
                    else:
                        newindex+=1
                        indexList.append(newindex)
        return indexList
    # Proper structuring of arguments based on the data
    def restructureArguements(self,arguements,indexList):
        newarguements=[]
        for wordarguements in arguements:
            newwordarguements=[]
            newindex=0
            newwordarguement=""
            if len(wordarguements)==0:
                newarguements.append(wordarguements)
            else:
                for arguement,index in zip(wordarguements,indexList):
                    if newindex==index:
                        if newwordarguement=="":
                            newarguement=arguement
                        else:
                            if not newarguement==arguement:
                                print newarguement,arguement
                                print "Arguements not equal"
                                exit(0)
                    else:
                        newwordarguements.append(newarguement)
                        newarguement=arguement
                        newindex+=1
                newwordarguements.append(newarguement)
                assert len(newwordarguements)==indexList[-1]+1
                newarguements.append(newwordarguements)
        return newarguements

    # Procedure to concatenate words based on the entities and event triggers for argument relation extraction
    def concatenateWordsAndEntities(self,sentenceList,entityList,eventList,arguementList,docIndexList,eventVocab):
        newsentenceList=[]
        newentityList=[]
        newarguementList=[]
        newindexList=[]
        newdocIndexList=[]
        for sentence,entities,events,arguements,sentIndexes in zip(sentenceList,entityList,eventList,arguementList,docIndexList):
            #print "\n******************sentence*********************\n"
            indexList=self.generateMergingList(entities,events,arguements)
            newsentence=[]
            newentities=[]
            newarguements=[]
            newsentIndex = []
            newindex=0
            newword =""
            newentity=""
            newwordIndex=0
            newwordarguements=[]
            #print sentence
            #print indexList
            #print entities
            for word,entity,index,wordarguements,wordIndex in zip(sentence,entities,indexList,arguements,sentIndexes):
                if newindex==index:
                    if newword=="":
                        newword=word
                        newentity=entity
                        newwordarguements=wordarguements
                        newwordIndex=wordIndex
                    else:
                        newword+=" "
                        newword+=word
                        if not newentity==entity:
                            print "Entities not equal"
                            exit(0)
                        if not len(wordarguements)==0 and not wordarguements==newwordarguements:
                            print wordarguements
                            print newwordarguements
                            print "Arguements not equal"
                            exit(0)
                else:
                    newsentence.append(newword)
                    newentities.append(newentity)
                    newarguements.append(newwordarguements)
                    newsentIndex.append(newwordIndex)
                    newword=word
                    newentity=entity
                    newwordarguements=wordarguements
                    newwordIndex=wordIndex
                    newindex+=1
            newsentence.append(newword)
            newentities.append(newentity)
            newarguements.append(newwordarguements)
            newsentIndex.append(newwordIndex)
            assert len(newarguements)==len(newsentence)
            assert len(newsentIndex)==len(newsentence)
            newsentenceList.append(newsentence)
            newentityList.append(newentities)
            newindexList.append(indexList)
            newdocIndexList.append(newsentIndex)
            newarguements=self.restructureArguements(newarguements,indexList)
            newarguementList.append(newarguements)
            '''print sentence
            print newsentence
            print indexList
            print entities
            print newentities
            print arguements
            print newarguements'''
        return newsentenceList,newentityList,newarguementList,newindexList,newdocIndexList
    # Dumping predicted triggers into a pickle file
    def dumpPredictedTriggers(self,key,eventlist):
        cPickle.dump(eventlist, open("./Predicted_test_triggers/"+key+".pkl", "wb"))

    # Loading predicted triggers into a pickle file
    def loadPredictedTriggers(self,key):
        return cPickle.load(open("./Predicted_test_triggers/"+key+".pkl", 'rb'))

    # Dumping predicted arguments into a pickle file
    def dumpPredictedArguements(self,key,arguementlist):
        cPickle.dump(arguementlist, open("./Predicted_test_arguements/" + key + ".pkl", "wb"))

    # Loading predicted arguments into a pickle file
    def loadPredictedArguements(self,key):
        return cPickle.load(open("./Predicted_test_arguements/" + key + ".pkl", 'rb'))

    # Correcting the entity and event dictionaries appropirately
    def removeOverlaps(self,entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,eventVocab):
        for key in eventIndexDict.keys():
            if entityIndexDict.has_key(key):
                del entityIndexDict[key]
        for key,value in entityIdDict.items():
            value=value.split(' ')
            value=value[0]
            if eventVocab.has_key(value) or eventIdDict.has_key(key):
                del entityIdDict[key]
    # sub procedure for finding a new unique id which has not occured in the given article uptill now
    def findNewId(self,idDict,prefix):
        no=1
        while True:
            id=prefix+str(no)
            if not idDict.has_key(id):
                return id
            no+=1
    # sub procedure for creating the final annotation line for the trigger word
    def dumpTrigger(self,triggerWord,triggerType,index,idDict,indexDict):
        dumpLine = ""
        newId=self.findNewId(idDict,"T")
        dumpLine+=newId
        dumpLine += "\t"
        value= triggerType+" "+str(index)+" "+str(index+len(triggerWord))
        dumpLine+=value
        dumpLine+="\t"+triggerWord
        idDict[newId]=value
        indexDict[index]=[newId]
        return dumpLine,idDict,indexDict
    # sub procedure for finding the correct set of arguments for the given trigger word
    def findUniqueArguements(self,arguementWordList):
        argDict={}
        for index in range(0,len(arguementWordList)):
            arg=arguementWordList[index]
            if not  arg== "No":
                if argDict.has_key(arg):
                    argDict[arg].append(index)
                else:
                    argDict[arg]=[index]
        argDictList=[]
        for key in argDict.keys():
            argDictList.append((key,argDict[key]))
        return argDictList
    # sub procedure for getting all the possible events w.r.t the current trigger word
    def populatedArgDumpLines(self,index,curDumpLine,dumpLines,argDictList):
        if index==len(argDictList):
            #print curDumpLine
            dumpLines.append(curDumpLine)
        else:
            argument,curArgList=argDictList[index][0],argDictList[index][1]
            for curArgIndex in curArgList:
                curDumpLine.add(curArgIndex)
                self.populatedArgDumpLines(index+1,copy.deepcopy(curDumpLine),dumpLines,argDictList)
                curDumpLine.remove(curArgIndex)
    #  sub procedure for creating the final annotation line for an event
    def createFinalArgDumpLine(self,indexTrigger,indexList,entityList,dumpLine,argList,idDict,indexDict,eventDict,triggerWord,revEntityIdDict):
        finalDumpLine=""
        newId = self.findNewId(idDict, "E")
        finalDumpLine+=newId
        finalDumpLine+="\t"
        if indexDict.has_key(indexTrigger):
            value=triggerWord+":"+indexDict[indexTrigger][0]+" "
        else:
            print indexTrigger,"Trigger lookup error!!!"
            exit(0)
        for sentIndex in dumpLine:
            actIndex=indexList[sentIndex]
            if eventDict.has_key(actIndex):
                value += argList[sentIndex] + ":" + eventDict[actIndex] + " "
            elif indexDict.has_key(actIndex) and revEntityIdDict.has_key(actIndex):
                value+=argList[sentIndex]+":"+revEntityIdDict[actIndex]+" "
            else:
                return ""
        finalDumpLine+=value
        idDict[newId]=value
        eventDict[indexTrigger]=newId
        return finalDumpLine

    # procedure for creating the event annotations based on the given arguement relationships
    def dumpArguements(self,word,triggerWord,entityList,arguementWordList,index,indexList,idDict,indexDict,eventDict,revEntityIdDict):
        dumpLines=[]
        finaldumpLines=[]
        curDumpLine=set()
        oldIdDict=copy.deepcopy(idDict)
        oldEventDict=copy.deepcopy(eventDict)
        argDictList=self.findUniqueArguements(arguementWordList)
        self.populatedArgDumpLines(0,curDumpLine,dumpLines,argDictList)
        if triggerWord=="Binding":
            #print "Binding",arguementWordList
            dumpLines=self.mergeDumpLines(dumpLines,["Theme"],arguementWordList)
            #print dumpLines
        if triggerWord=="Planned_process":
            #print "Planned_process",arguementWordList
            dumpLines=self.mergeDumpLines(dumpLines,["Instrument"],arguementWordList)
        #print dumpLines
        for curindex in range(0,len(dumpLines)):
            dummy=self.createFinalArgDumpLine(index,indexList,entityList,dumpLines[curindex],arguementWordList,idDict,indexDict,eventDict,triggerWord,revEntityIdDict)
            if triggerWord=="Binding":
                dummy=self.updateDumpLine(dummy,"Theme")
            elif triggerWord=="Instrument":
                dummy = self.updateDumpLine(dummy, "Instrument")
            if not dummy=="":
                #print dummy
                finaldumpLines.append(dummy)
            else:
                #print triggerWord, index, "Event not found error!!!"
                return "",oldIdDict,indexDict,oldEventDict
        '''assert len(flagList)==len(dumpLines)
        for dumpLine,flag in zip(dumpLines,flagList):
            if flag==1:
                dummy=self.createFinalArgDumpLine(index,indexList,entityList,dumpLine,arguementWordList,idDict,indexDict,eventDict,triggerWord,revEntityIdDict)
                if dummy=="":
                    print triggerWord,index,"Event not found error!!!"
                    continue
                finaldumpLines.append(dummy)'''
        return finaldumpLines,idDict,indexDict,eventDict
    # Updating the given event annotations for specfic special events
    def updateDumpLine(self,dummy, keyword):
        if dummy=="":
            return ""
        newdummy = ""
        dummy = dummy.split('\t')
        newdummy += dummy[0]
        newdummy += "\t"
        values = dummy[1]
        values = values.split(' ')
        count = 0
        for value in values:
            oldvalue = value
            value = value.split(':')
            if value[0] == keyword:
                if count == 0:
                    newdummy += (oldvalue)
                else:
                    newdummy += (value[0] + str(count + 1) + ":" + value[1])
                count += 1
            else:
                newdummy += (oldvalue)
            newdummy += " "
        newdummy.strip(" ")
        return newdummy

    # Updating the given event annotations for specfic special events
    def mergeDumpLines(self,dumpLines,mergeList,arguementWordList):
        indexList=set()
        index=0
        for indexset in dumpLines:
            for index in indexset:
                if arguementWordList[index] in mergeList:
                    indexList.add(index)
        if len(indexList)==0:
            return dumpLines
        for index1 in range(0,len(dumpLines)):
            word=dumpLines[index1]
            for index in indexList:
                word.add(index)
            dumpLines[index1]=word
        finaldumpLines=[]
        for line in dumpLines:
            flag=0
            for line1 in finaldumpLines:
                if len(line.symmetric_difference(line1))==0:
                    flag=1
                    break
            if flag==0:
                finaldumpLines.append(line)
        #print finaldumpLines
        return finaldumpLines
    # procedural call for creating final event annnotations
    def dumpPredictedEvents(self,key,sentenceList,entityList,arguementList,indexList,entityIdDict, entityIndexDict, eventIdDict, eventIndexDict,eventVocab,triggerList,entityVocab):
        dumpList=[]
        punctuations = string.punctuation
        '''print entityIdDict
        print entityIndexDict
        print eventIdDict
        print eventIndexDict'''
        eventDict={}
        fp=open("./Predicted_test_events/"+key+".a2.t12","w")
        self.removeOverlaps(entityIdDict,entityIndexDict,eventIdDict,eventIndexDict,eventVocab)
        revEntityIdDict = {int(v.split(' ')[1]): k for k, v in entityIdDict.items()}
        '''print revEntityIdDict
        print entityIdDict
        print entityIndexDict
        print eventIdDict
        print eventIndexDict'''
        for sentence,entitySentList,arguementSentList,indexSentList,triggerSentList in zip(sentenceList,entityList,arguementList,indexList,triggerList):
            print "\n******************sentence*********************\n"
            print sentence
            print entitySentList
            #print triggerSentList
            #print indexSentList
            assert len(sentence)==len(entitySentList)
            print "Triggers"
            for word,arguementWordList,triggerWord,index in zip(sentence,arguementSentList,entitySentList,indexSentList):
                while len(word) > 0 and word[-1] in punctuations:
                    word = word[:-1]
                while len(word) > 0 and word[0] in punctuations:
                    word = word[1:]
                if eventVocab.has_key(triggerWord) and not triggerWord=="Other":
                    print triggerWord,arguementWordList
                    #arguementWordList=reconstructLabels(arguementWordList,entitySentList)
                    arguementWordList=reconstructLabelsWithRules(arguementWordList,entitySentList,entityVocab,triggerWord)
                    assert len(entitySentList)==len(arguementWordList)
                    #print triggerWord,arguementWordList
                    dumpLine,entityIdDict,entityIndexDict=self.dumpTrigger(word,triggerWord,index,entityIdDict,entityIndexDict)
                    print dumpLine
                    fp.write(dumpLine)
                    fp.write('\n')
            '''print entityIdDict
            print entityIndexDict
            print revEntityIdDict'''
            flaglist = [0] * len(sentence)
            print "Non Recursive Events"
            for curindex in range(0,len(sentence)):
                word=sentence[curindex]
                arguementWordList=arguementSentList[curindex]
                triggerWord=entitySentList[curindex]
                index=indexSentList[curindex]
                while len(word) > 0 and word[-1] in punctuations:
                    word = word[:-1]
                while len(word) > 0 and word[0] in punctuations:
                    word = word[1:]
                if eventVocab.has_key(triggerWord) and not triggerWord=="Other" and not triggerWord=="Regulation" and not triggerWord=="Positive_regulation" and not triggerWord=="Negative_regulation":
                    #arguementWordList=reconstructLabels(arguementWordList,entitySentList)
                    #print triggerWord,arguementWordList
                    arguementWordList = reconstructLabelsWithRules(arguementWordList, entitySentList, entityVocab,triggerWord)
                    assert len(entitySentList)==len(arguementWordList)
                    #print triggerWord,arguementWordList
                    dumpLines,entityIdDict,entityIndexDict,eventDict=self.dumpArguements(word,triggerWord,entityList,arguementWordList,index,indexSentList,entityIdDict,entityIndexDict,eventDict,revEntityIdDict)
                    if dumpLines=="":
                        print "Non recursive event error!!!!"
                        #exit(0)
                    else:
                        flaglist[curindex]=1
                    for dumpLine in dumpLines:
                        print dumpLine
                        fp.write(dumpLine)
                        fp.write('\n')
            curflagcount=0
            oldflagcount=-1
            print "Recursive Events"
            while(oldflagcount!=curflagcount):
                oldflagcount = curflagcount
                curflagcount=0
                for curindex in range(0, len(sentence)):
                    word = sentence[curindex]
                    arguementWordList = arguementSentList[curindex]
                    triggerWord = entitySentList[curindex]
                    index = indexSentList[curindex]
                    while len(word) > 0 and word[-1] in punctuations:
                        word = word[:-1]
                    while len(word) > 0 and word[0] in punctuations:
                        word = word[1:]
                    if eventVocab.has_key(triggerWord) and (triggerWord=="Regulation" or triggerWord=="Positive_regulation" or triggerWord=="Negative_regulation") and flaglist[curindex]==0:
                        #arguementWordList=reconstructLabels(arguementWordList,entitySentList)
                        #print triggerWord,arguementWordList
                        arguementWordList = reconstructLabelsWithRules(arguementWordList, entitySentList, entityVocab,triggerWord)
                        print arguementWordList
                        #arguementWordList = self.forceLabels(arguementWordList, entitySentList)
                        #print arguementWordList
                        assert len(entitySentList)==len(arguementWordList)
                        #print triggerWord,arguementWordList
                        dumpLines,entityIdDict,entityIndexDict,eventDict=self.dumpArguements(word,triggerWord,entityList,arguementWordList,index,indexSentList,entityIdDict,entityIndexDict,eventDict,revEntityIdDict)
                        if dumpLines == "":
                            #print "recursive event error!!!!"
                            None
                        else:
                            flaglist[curindex] = 1
                            for dumpLine in dumpLines:
                                print dumpLine
                                fp.write(dumpLine)
                                fp.write('\n')
                for flag,triggerWord in zip(flaglist,entitySentList):
                    if eventVocab.has_key(triggerWord) and not triggerWord=="Other":
                        if flag==0:
                            print triggerWord,flag
                            curflagcount+=1
                oldflagcount = curflagcount=0
            if oldflagcount>0:
                print "Cyclic event error!!!"
        fp.close()

    # sub procedure for assignIndextoWord
    def getLength(self,index,indexDict):
        low=index
        val=indexDict[index][0]
        val=val.split(" ")
        high=int(val[0])
        return high-low

    # loading the features generated by the gdep parser
    def loadParseTrees(self,key,sentenceList,entitylist,arguementList,prefix):
        newSentenceList=[]
        newEntityList=[]
        newArguementList=[]
        sentenceIdList=[]
        sentenceRootList=[]
        sentenceChunkList=[]
        sentencePosList=[]
        sentenceParentList=[]
        sentenceParseList=[]
        id, root, chunk, pos, parentid, parse=cPickle.load( open("./Parsed_output_"+prefix + key + ".pkl", "rb"))
        for sentence,sentids,sentroots,sentchunks,sentpos,sentparentids,sentparses,entities,arguements in zip(sentenceList,id,root,chunk,pos,parentid,parse,entitylist,arguementList):
            newsentence=[]
            newentities=[]
            newarguementsdup=[]
            newarguements=[]
            for word,entity,arguement in zip(sentence,entities,arguements):
                if not word=="":
                    newsentence.append(word)
                    newentities.append(entity)
                    newarguementsdup.append(arguement)
                else:
                    print key
                    exit(0)
            for arguements in newarguementsdup:
                if len(arguements)>0:
                    assert len(arguements)==len(sentence)
                    newarguement=[]
                    for word,arguement in zip(sentence,arguements):
                        if not word=="":
                            newarguement.append(arguement)
                    assert len(newarguement)==len(newsentence)
                    newarguements.append(newarguement)
                else:
                    newarguements.append([])

            '''print newsentence
            print newentities
            print sentids
            print sentroots
            print sentchunks
            print sentpos
            print sentparentids
            print sentparses
            print "\n\n"
            print len(newsentence)
            print len(newentities)
            print len(sentids)
            print len(sentroots)
            print len(sentchunks)
            print len(sentpos)
            print len(sentparentids)
            print len(sentparses)'''
            assert(len(newsentence)==len(sentids)==len(sentroots)==len(sentchunks)==len(sentpos)==len(sentparentids)==len(sentparses)==len(newentities)==len(newarguements))
            newSentenceList.append(newsentence)
            newEntityList.append(newentities)
            newArguementList.append(newarguements)
            sentenceIdList.append(sentids)
            sentenceRootList.append(sentroots)
            sentenceChunkList.append(sentchunks)
            sentencePosList.append(sentpos)
            sentenceParentList.append(sentparentids)
            sentenceParseList.append(sentparses)
        return newSentenceList,newEntityList,newArguementList,sentenceIdList,sentenceRootList,sentenceChunkList,sentencePosList,sentenceParentList,sentenceParseList

    # Procedure for getting shortest dependency path between two entities
    def getShortestDependencyPath(self,ids,words,dependencies,parents,triggerwordid,entities):
        edges=[]
        wordIdDict={}
        entityIdDict={}
        shortestpathsvertices=[]
        shortestpathverticesentities=[]
        shortestpathsedges=[]
        for id,word,entity in zip(ids,words,entities):
            wordIdDict[id]=word
            entityIdDict[id]=entity
        for id,word,dependency,parentid in zip(ids,words,dependencies,parents):
            #print word,id,parentid
            if not parentid=='0':
                edges.append((word+"-"+id,wordIdDict[parentid]+"-"+parentid))
        #print edges
        graph = nx.Graph(edges)
        for word,id in zip(words,ids):
            path=nx.shortest_path(graph,source=wordIdDict[triggerwordid]+"-"+triggerwordid,target=word+"-"+id)
            #print path
            entitypath=[]
            for vertex in path:
                id = vertex.rsplit('-',1)
                id = id[len(id) - 1]
                entitypath.append(entityIdDict[id])
            #print entitypath
            shortestpathsvertices.append(path)
            shortestpathverticesentities.append(entitypath)
        for path in shortestpathsvertices:
            edgePath=[]
            for vertexIndex in range(0,len(path)-1):
                vertex=path[vertexIndex]
                id=vertex.rsplit("-")
                id=id[len(id)-1]
                edgePath.append(dependencies[int(id)-1])
            shortestpathsedges.append(edgePath)
        return shortestpathsvertices,shortestpathsedges,shortestpathverticesentities
