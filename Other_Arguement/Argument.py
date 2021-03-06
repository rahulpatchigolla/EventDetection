from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import random
import re
import collections 
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle
import sys
sys.path.append("source")

from Utils import *
from RNNModel import *
#from CNNRNNModel import *
#from CNNModel import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

embSize = 200
d1_emb_size=10
d2_emb_size=10
type_emb_size=50
numfilter = 150
out_file = "results.txt"
num_epochs = 50
N = 1
batch_size=100
reg_para = 0.0
drop_out = 0.8

ftrain = "train.txt"
ftest = "test.txt"


Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables,Tr_sent_entities= readData(ftrain)
print ('Tr_sent_contents', len(Tr_sent_contents))
Tr_word_list, Tr_d1_list, Tr_d2_list= makeFeatures(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list)
Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables,Te_sent_entities = readData(ftest)
print ("Te_sent_contents", len(Te_sent_contents))
Te_word_list, Te_d1_list, Te_d2_list= makeFeatures(Te_sent_contents, Te_entity1_list, Te_entity2_list)

print ("train_size", len(Tr_word_list))
print ("test_size", len(Te_word_list))

train_sent_lengths, test_sent_lengths = findSentLengths([Tr_word_list, Te_word_list])
sentMax = max(train_sent_lengths + test_sent_lengths)
print ("max sent length", sentMax)
train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')

label_dict = {"No":0,"Theme":1,"AtLoc":2,"ToLoc":3,"FromLoc":4,"Site":5,"Cause":6,"Instrument":7}

word_dict = makeWordList(Tr_word_list, Te_word_list)
d1_dict = makeWordList(Tr_d1_list, Te_d1_list)
d2_dict = makeWordList(Tr_d2_list, Te_d2_list)
type_dict = makeWordList(Tr_sent_entities, Te_sent_entities)
print ("word dictonary length", len(word_dict) )
print ("entity dictionary lengh",len(type_dict))
# Word Embedding
wv = loadWordEmbeddings(word_dict,embSize)

# Mapping Train
W_train =   mapWordToId(Tr_word_list, word_dict)
d1_train = mapWordToId(Tr_d1_list, d1_dict)
d2_train = mapWordToId(Tr_d2_list, d2_dict)
T_train = mapWordToId(Tr_sent_entities, type_dict)

Y_t = mapLabelToId(Tr_sent_lables, label_dict)
Y_train = np.zeros( (len(Y_t), len(label_dict)) )
for i in range(len(Y_t)):
	Y_train[i][Y_t[i]] = 1.0

# Mapping Test
W_test =   mapWordToId(Te_word_list, word_dict)
d1_test = mapWordToId(Te_d1_list, d1_dict)
d2_test = mapWordToId(Te_d2_list, d2_dict)
T_test = mapWordToId(Te_sent_entities, type_dict)
Y_t = mapLabelToId(Te_sent_lables, label_dict)
Y_test = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_test[i][Y_t[i]] = 1.0
#padding
W_train, d1_train, d2_train, T_train, W_test, d1_test, d2_test, T_test = paddData([W_train, d1_train, d2_train, T_train, W_test, d1_test, d2_test, T_test], sentMax)
print ("train", len(W_train))
print ("test", len(W_test))
'''print "Dumping into pickle file"
with open('train_test_rnn_data.pickle', 'wb') as handle:
	pickle.dump(W_train, handle)
	pickle.dump(d1_train, handle)
	pickle.dump(d2_train, handle)
	pickle.dump(T_train, handle)
	pickle.dump(Y_train, handle)
	pickle.dump(train_sent_lengths, handle)

	pickle.dump(W_test, handle)
	pickle.dump(d1_test, handle)
	pickle.dump(d2_test, handle)
	pickle.dump(T_test, handle)
	pickle.dump(Y_test, handle)
	pickle.dump(test_sent_lengths, handle)

	pickle.dump(wv, handle)
	pickle.dump(word_dict, handle)
	pickle.dump(d1_dict, handle)
	pickle.dump(d2_dict, handle)
	pickle.dump(type_dict, handle)
	pickle.dump(label_dict, handle)
	pickle.dump(sentMax, handle)
exit(0)
with open('train_test_rnn_data.pickle', 'rb') as handle:
	W_train = pickle.load(handle)
	d1_train= pickle.load(handle)
	d2_train= pickle.load(handle)
	T_train = pickle.load(handle)
	Y_train = pickle.load(handle)
	train_sent_lengths = pickle.load(handle)
	W_test = pickle.load(handle)
	d1_test = pickle.load(handle)
	d2_test = pickle.load(handle)
	T_test = pickle.load(handle)
	Y_test = pickle.load(handle)
	test_sent_lengths = pickle.load(handle)

	wv = pickle.load(handle)
	word_dict= pickle.load(handle)
	d1_dict = pickle.load(handle)
	d2_dict = pickle.load(handle)
	type_dict = pickle.load(handle)
	label_dict = pickle.load(handle)
	sentMax = pickle.load(handle)'''

#vocabulary size
word_dict_size = len(word_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
label_dict_size = len(label_dict)

fp = open(out_file, 'w')

for reg_para, drop_out in [(0.001, 0.8)]: #, (0.001,0.8)

	rnn = Relation(label_dict_size, 		# class labels size
			word_dict_size, 		# word embedding size
			d1_dict_size, 			# position embedding size
			d2_dict_size,
			type_dict_size, 		# type emb. size
			sentMax, 			# length of sentence
			wv,				# word embedding
			d1_emb_size=d1_emb_size, 	# emb. length
			d2_emb_size=d2_emb_size,
			type_emb_size=type_emb_size,
			num_filters=numfilter, 		# number of hidden nodes in RNN
			w_emb_size=embSize, 		# dim. word emb
			l2_reg_lambda=reg_para,		# l2 reg
			)

	def test_step(W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test):
		n = len(W_test)
		ra = n/batch_size
		samples = []
		for i in range(ra):
			samples.append(range(batch_size*i, batch_size*(i+1)))
		samples.append(range(batch_size*(i+1), n))

		acc = []
		pred = []
		for i in samples:
			p,a = rnn.test_step(W_test[i], test_sent_lengths[i], d1_test[i], d2_test[i], T_test[i], Y_test[i])
			pred.extend(p)
		return pred, acc

	train_len = len(W_train)
	y_true_list = []
	y_pred_list = []
	num_batches_per_epoch = int(train_len/batch_size) + 1

	for epoch in range(num_epochs):
		shuffle_indices = np.random.permutation(np.arange(train_len))
		W_tr =  W_train[shuffle_indices]
		d1_tr = d1_train[shuffle_indices]
		d2_tr = d2_train[shuffle_indices]
		T_tr = T_train[shuffle_indices]
		Y_tr = Y_train[shuffle_indices]
		S_tr = train_sent_lengths[shuffle_indices]
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num*batch_size
			end_index = min((batch_num + 1) * batch_size, train_len)
			loss = rnn.train_step(W_tr[start_index:end_index], S_tr[start_index:end_index], d1_tr[start_index:end_index],
					d2_tr[start_index:end_index], T_tr[start_index:end_index], Y_tr[start_index:end_index],drop_out)

		if (epoch%N) == 0:


			pred, acc = test_step(W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test)
			y_true = np.argmax(Y_test, 1)
			y_pred = pred
			y_true_list.append(y_true)
			y_pred_list.append(y_pred)

	fp.write('lemada and droput\t'+ str(reg_para)+"\t"+str(drop_out)+'\n' )
	for y_true, y_pred in zip(y_true_list, y_pred_list):
		fp.write(str(precision_score(y_true, y_pred,[1,2,3,4,5,6,7], average='micro' )))
		fp.write('\t')
		fp.write(str(recall_score(y_true, y_pred, [1,2,3,4,5,6,7], average='micro' )))
		fp.write('\t')
		fp.write(str(f1_score(y_true, y_pred, [1,2,3,4,5,6,7], average='micro' )))
		fp.write('\t')
		fp.write('\n')

	fp.write('\n\n\n')
	rnn.sess.close()

