from SentenceExtractor import *
import re
import subprocess
import cPickle

from copy import deepcopy

processedpathtrain="./Preprocessed_Corpus_train/"
corpuspathtrain="./Corpus/standoff/test/train/"
processedpathtest="./Preprocessed_Corpus_test/"
corpuspathtest="./Corpus/standoff/test/test/"
# This file generates the data for the dependency parser and finally stores the parse trees for training data
posVocab={}
parseVocab={}
def buildVocabs(words,vocab):
    for word in words:
        if not vocab.has_key(word):
            vocab[word]=0
def mergeParseWordsAndIdentifiers(words,tags,flag=True):
    mergedsentence=[]
    for i in range(0,len(tags)):
        word=words[i]
        tag=tags[i]
        if flag==True:
            word=word.rsplit('-',1)
            mergedsentence.append(word[0])
        else:
            mergedsentence.append(word)
        mergedsentence.append(tag)
    lastword=words[-1]
    if flag==True:
        lastword=lastword.rsplit('-',1)
        mergedsentence.append(lastword[0])
    else:
        mergedsentence.append(lastword)
    return mergedsentence
def removePunctuations(word):
    punctuations=string.punctuation
    while len(word) > 1 and word[-1] in punctuations:
        word = word[:-1]
    while len(word) > 1 and word[0] in punctuations:
        word = word[1:]
    word = word.lower()
    return word
def replaceWordWithEntites(sentence,entities):
    newsentence=[]
    vocab={'Dissociation': 'Dissociation', 'Pathological_formation': 'formation', 'Organism_subdivision': '', 'Protein_processing': 'process',
           'Drug_or_compound': 'Drug', 'Cell': 'Cell', 'Protein_domain_or_region': 'Protein', 'Multi-tissue_structure': 'Tissue',
           'Pathway': 'Pathway', 'Gene_or_gene_product': 'Gene', 'Cell_division': 'division', 'Immaterial_anatomical_entity': 'entity',
           'Metabolism': 'Metabolism', 'Translation': 'Translation', 'DNA_domain_or_region': 'DNA', 'Ubiquitination': 'Ubiquitination',
           'Developing_anatomical_structure': 'structure', 'Organ': 'Organ', 'Acetylation': 'Acetylation',
           'Cellular_component': 'cell', 'Reproduction': 'Reproduction', 'Organism_substance': 'Organism', 'Organism': 'Organism',
           'Anatomical_system': 'Anatomy', 'DNA_methylation': 'Methylation', 'Tissue': 'Tissue'}
    for word,entity in zip(sentence,entities):
        if not entity=="None" and vocab.has_key(entity):
                newsentence.append(entity)
        else:
            newsentence.append(word)
    return newsentence
if __name__ == "__main__":
    dirDicttrain = readDir(corpuspathtrain)
    dirDicttest = readDir(corpuspathtest)
    train_set = dirDicttrain.keys()
    test_set = dirDicttest.keys()
    wordVocabtrain = cPickle.load(open(processedpathtrain + "wordVocab.pkl", 'rb'))
    entityVocabtrain = cPickle.load(open(processedpathtrain + "entityVocab.pkl", 'rb'))
    eventVocabtrain = cPickle.load(open(processedpathtrain + "eventVocab.pkl", 'rb'))
    wordVocabtest = cPickle.load(open(processedpathtest + "wordVocab.pkl", 'rb'))
    entityVocabtest = cPickle.load(open(processedpathtest + "entityVocab.pkl", 'rb'))
    eventVocabtest = cPickle.load(open(processedpathtest + "eventVocab.pkl", 'rb'))
    wordVocab = mergeVocabs(wordVocabtest, wordVocabtest)
    eventVocab = mergeVocabs(eventVocabtest, eventVocabtest, removeRareEvents=True)
    entityVocab = mergeVocabs(entityVocabtest, entityVocabtest)
    fpwrite = open("ToParse.txt", "w")
    for key in train_set:
        print "\n\n\n*******************", key, "*******************\n\n\n"
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(processedpathtrain + key + ".pkl", 'rb'))
        fp = open(corpuspathtrain + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, _, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                 eventIdDict, eventIndexDict, content,
                                                                                 sentIndex, eventVocab, wordVocab,
                                                                                 removeRareTriggers=True,
                                                                                 concatenate=True)
        fpwrite.write(key)
        fpwrite.write("\n\n")
        for index in range(0,len(sentenceList)):
            sentence=sentenceList[index]
            entities=entityList[index]
            print sentence
            newsentence=""
            sentence = replaceWordWithEntites(sentence, entities)
            for word in sentence:
                word=word.split(' ')
                if len(word)>1:
                    newword=""
                    for part in word:
                        part=removePunctuations(part)
                        newword+=part
                        newword+="_"
                        break
                    newword=newword.strip('_')
                    newsentence+=newword
                else:
                    newword=removePunctuations(word[0])
                    newsentence+=newword
                newsentence+=" "
            newsentence.strip(" ")
            newsentence.strip(".")
            fpwrite.write(newsentence)
            fpwrite.write("\n")
        fpwrite.write("\n")
    fpwrite.close()
    os.chdir("./gdep-beta2")
    subprocess.call("./gdep < ../ToParse.txt > ParsedOutput.txt -nt",shell=True)
    os.chdir("..")
    #for key in train_set:
    #    print key
    print "Starting"
    fpread=open("./gdep-beta2/ParsedOutput.txt")
    fpwrite=""
    id=[]
    root=[]
    chunk=[]
    pos=[]
    parentid=[]
    parse=[]
    sentid = []
    sentroot = []
    sentchunk = []
    sentpos = []
    sentparentid = []
    sentparse = []
    previd=""
    while True:
        line =fpread.readline()
        if line=="":
            #print "Dumping",previd
            cPickle.dump([id, root, chunk, pos, parentid, parse],open("./Parsed_output_train/" + previd + ".pkl", "wb"))
            break
        else:
            if re.match("^1\tPMID-.*",line):
                #print line
                line =line.split('\t')
                #print line[1]
                if not previd=="":
                    #print "Dumping",previd
                    cPickle.dump([id,root,chunk,pos,parentid,parse],open("./Parsed_output_train/" + previd + ".pkl", "wb"))
                    id = []
                    root = []
                    chunk = []
                    pos = []
                    parentid = []
                    parse = []
                fpread.readline()
                previd=line[1]
            else:
                if line=="\n":
                    id.append(sentid)
                    root.append(sentroot)
                    chunk.append(sentchunk)
                    pos.append(sentpos)
                    parentid.append(sentparentid)
                    parse.append(sentparse)
                    buildVocabs(sentpos,posVocab)
                    buildVocabs(sentparse,parseVocab)
                    sentid = []
                    sentroot = []
                    sentchunk = []
                    sentpos = []
                    sentparentid = []
                    sentparse = []
                else:
                    line =line.split('\t')
                    sentid.append(line[0])
                    sentroot.append(line[2])
                    sentchunk.append(line[3])
                    sentpos.append(line[4])
                    sentparentid.append(line[6])
                    sentparse.append(line[7].strip('\n'))
    cPickle.dump(posVocab, open("./Parsed_output_train/posVocab.pkl", "wb"))
    cPickle.dump(parseVocab,open("./Parsed_output_train/parseVocab.pkl", "wb"))
    '''fpdump = open("train.txt", "w")
    for key in train_set:
        print key
        sentIndex, entityIdDict, entityIndexDict, eventIdDict, eventIndexDict = cPickle.load(
            open(processedpathtrain + key + ".pkl", 'rb'))
        fp = open(corpuspathtrain + key + ".txt", 'r')
        content = fp.read()
        extractor = sentenceExtractor()
        sentenceList, entityList, _, arguementList = extractor.entitiesAndEvents(entityIdDict, entityIndexDict,
                                                                                 eventIdDict, eventIndexDict, content,
                                                                                 sentIndex, eventVocab, wordVocab,
                                                                                 removeRareTriggers=True,
                                                                                 concatenate=True)
        sentenceList, entityList, arguementList, sentenceIdList,sentenceRootList, sentenceChunkList, sentencePosList, sentenceParentList, sentenceParseList=extractor.loadParseTrees(key,sentenceList,entityList,arguementList,"train/")

        for index in range(0, len(sentenceList)):
            sentence = sentenceList[index]
            entities = entityList[index]
            sentenceIds=sentenceIdList[index]
            dependencies=sentenceParseList[index]
            parents=sentenceParentList[index]
            print sentence
            print entities
            print sentenceIds
            print dependencies
            print parents
            assert len(sentence) == len(entities)==len(sentenceIds)==len(dependencies)
            arguements = arguementList[index]
            for index1 in range(0, len(sentence)):
                word = sentence[index1]
                entity = entities[index1]
                wordarguements = arguements[index1]
                depPathvertices, depPathEdges,depPathverticesentities = extractor.getShortestDependencyPath(sentenceIds, sentence, dependencies,parents, str(index1 + 1),entities)
                #print depPathvertices
                #print depPathEdges
                if eventVocab.has_key(entity):
                    _, input4 = truncateLableswithRules(wordarguements, entities, entityVocab, entity)
                    assert len(wordarguements) > 0
                    for index2 in range(0, len(sentence)):
                        word1 = sentence[index1]
                        word2 = sentence[index2]
                        if (index1 == index2) or (index2 not in input4):
                            continue
                        if entities[index2] == "None":
                            if wordarguements[index2] == "No":
                                continue
                            else:
                                None
                                #print word1, entities[index1], word2, entities[index2], wordarguements[index2]
                        dumpsentence = deepcopy(sentence)
                        dumpsentence[index1] = "TRIGGER"
                        dumpsentence[index2] = "ARGUEMENT"
                        assert len(dumpsentence) == len(entities)
                        fpdump.write(convertToSentence(dumpsentence, "#"))
                        fpdump.write("\n")
                        fpdump.write(convertToSentence(entities, "#"))
                        fpdump.write("\n")
                        fpdump.write(word1 + "\t" + word2)
                        fpdump.write("\n")
                        fpdump.write(convertToSentence(mergeParseWordsAndIdentifiers(depPathvertices[index2],depPathEdges[index2]),"#"))
                        fpdump.write("\n")
                        fpdump.write(convertToSentence(mergeParseWordsAndIdentifiers(depPathverticesentities[index2],["None"]*(len(depPathverticesentities[index2])-1),flag=False),"#"))
                        fpdump.write("\n")
                        fpdump.write(wordarguements[index2])
                        fpdump.write("\n\n")
    fpdump.close()'''




