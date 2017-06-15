from Process1Test import readDir
import os
from shutil import copyfile
eventCount={}
multiWordDict={}
multiwordcount=0
#This file is to change the train data according to my preprocessing requirements
def countTriggers(triggerDict,triggerEventDict):
    for key in triggerEventDict.keys():
        value=triggerDict[key]
        triggerWord=value[0]
        triggerWord=triggerWord.split(' ')
        triggerWord=triggerWord[0]
        if eventCount.has_key(triggerWord):
            eventCount[triggerWord]+=1
        else:
            eventCount[triggerWord] = 1
def adjustEventLine(eventLine,removedEvents):
    #print eventLine
    args=eventLine.strip('\n').strip(' ').split(' ')
    #print args
    newargs=[]
    for arg in args:
        arg=arg.split(':')
        if not removedEvents.has_key(arg[1]):
            newargs.append(arg[0]+":"+arg[1])
    neweventLine=""
    #print newargs
    for newarg in newargs:
        neweventLine+=newarg
        neweventLine+=" "
    neweventLine.strip(' ')
    neweventLine+="\n"
    #print neweventLine
    return neweventLine
def removeNegationAndSpeculation(dirDict,destpath,path):
    totalmultiwordtriggers=0
    for fileName in dirDict.keys():
        print fileName
        triggerDict={}
        eventDict={}
        triggerEventDict={}
        removedevents={}
        file=fileName
        file+=".a2"
        fp = open(path+file, "r")
        for line in fp:
            if line[0]=="T":
                line=line.split('\t')
                #print line
                assert len(line)==3
                triggerDict[line[0]]=[line[1],line[2]]
            elif line[0]=="E":
                line = line.split('\t')
                #print line
                assert len(line) == 2
                eventDict[line[0]] = line[1]
        fp.close()
        for key,value in eventDict.items():
            value=value.split(" ")
            assert len(value)>=0
            triggerWord=value[0]
            triggerId=triggerWord.split(":")
            triggerId = triggerId[1]
            assert triggerDict.has_key(triggerId)
            if triggerEventDict.has_key(triggerId):
                triggerEventDict[triggerId].append(key)
            else:
                triggerEventDict[triggerId]=[key]
        #print triggerDict
        #print eventDict
        #print triggerEventDict
        #oldval = len(triggerEventDict)
        countTriggers(triggerDict, triggerEventDict)
        fpwrite=open(destpath+file,"w")
        for key,events in triggerEventDict.items():
            triggerLine=triggerDict[key]
            fpwrite.write(key+"\t"+triggerLine[0]+"\t"+triggerLine[1])
            for event in events:
                eventLine=eventDict[event]
                eventLine=adjustEventLine(eventLine,removedevents)
                fpwrite.write(event+"\t"+eventLine)
        fpwrite.close()
        copyfile(path+fileName+".txt", destpath+fileName+".txt")
        copyfile(path + fileName + ".a1", destpath + fileName + ".a1")
if __name__ == "__main__":
    path = "./Corpus/standoff/test/train/"
    destpath="./Corpus_filtered/train/"
    dirDict = readDir(path)
    for file in dirDict.keys():
        print file
        file+=".txt"
        fp = open(path+file, "r")
        content=fp.read()
        fp.close()
        if content[0]=='[':
            content='@'+content[1:]
            index=1
            while True:
                if content[index]==']':
                    content=content[:index]+'.'+content[index+1:]
                    break
                index+=1
        content=content.replace("\n\n","  ")
        fp = open(path+file, "w")
        fp.write(content)
        fp.close()
    removeNegationAndSpeculation(dirDict,destpath,path)
    print eventCount
    #print multiWordDict

