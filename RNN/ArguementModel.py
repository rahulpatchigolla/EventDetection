from Utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()
#This file contains the actual arguement model architechure
class Dropout(object):
    def __init__(self,X, p=0.):
        self.retain_prob=1 - p
        X *= srng.binomial(X.shape, p=self.retain_prob, dtype=theano.config.floatX)
        X /= self.retain_prob
        self.output=X
class HiddenLayer(object):
    def __init__(self,rng,in_dim,op_dim,input):
        W_ar=numpy.asarray(
            rng.uniform(
            low=-numpy.sqrt(6. / (in_dim+op_dim)),
            high=numpy.sqrt(6. / (in_dim+op_dim)),
            size=(in_dim,op_dim)
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_ar, name='W', borrow=True)
        b_ar = numpy.zeros((op_dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_ar, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.input = input
        self.output=T.tanh(T.dot(input,self.W)+self.b)
class InputLayer(object):
    def __init__(self,wordVocabSize,wordVocab,entityVocabSize,entityVocab,distanceVocabSize,distanceVocab,embSizeWord,embSizeEntity,embSizeDistance,inputWord,input1,input2,input3,input4,rnnFeature):
        self.WordEmbLayer=EmbedLayer(VocabSize=wordVocabSize,embSize=embSizeWord,wordids=input1,wordVocab=wordVocab,embPresent=True)
        self.EntityEmdLayer=EmbedLayer(VocabSize=entityVocabSize,embSize=embSizeEntity,wordids=input2,wordVocab=entityVocab,embPresent=False)
        self.DistanceEmdLayer=EmbedLayer(VocabSize=distanceVocabSize,embSize=embSizeDistance,wordids=input3,wordVocab=distanceVocab,embPresent=False)
        self.rnnInput=T.concatenate([self.WordEmbLayer.output,self.EntityEmdLayer.output,self.DistanceEmdLayer.output],axis=1)
        self.rnnOutput=rnnFeature.rnn_bi(self.rnnInput)
        self.triggerRnn=self.rnnOutput[inputWord]
        self.triggerWord=self.WordEmbLayer.output[inputWord]
        self.triggerEntity=self.EntityEmdLayer.output[inputWord]
        self.triggerRnnReplicated=T.tile(self.triggerRnn,(self.rnnOutput.shape[0],1))
        self.triggerWordReplicated=T.tile(self.triggerWord,(self.rnnOutput.shape[0],1))
        self.triggerEntityReplicated=T.tile(self.triggerEntity,(self.rnnOutput.shape[0],1))
        self.triggerRnnTruncated=self.triggerRnnReplicated[input4]
        self.rnnOutputTruncated=self.rnnOutput[input4]
        self.triggerWordTruncated=self.triggerWordReplicated[input4]
        self.WordEmbLayerOutputTruncated=self.WordEmbLayer.output[input4]
        self.triggerEntityTruncated=self.triggerEntityReplicated[input4]
        self.EntityEmdLayerOutputTruncated=self.EntityEmdLayer.output[input4]
        self.distanceEmbTruncated=self.DistanceEmdLayer.output[input4]
        #self.output=T.concatenate([self.rnnOutput,self.WordEmbLayer.output,self.EntityEmdLayer.output,self.triggerRnnReplicated,self.triggerWordReplicated,self.triggerEntityReplicated],axis=1)
        self.output = T.concatenate([self.rnnOutputTruncated, self.WordEmbLayerOutputTruncated, self.EntityEmdLayerOutputTruncated, self.triggerRnnTruncated,self.triggerWordTruncated, self.triggerEntityTruncated], axis=1)
        #self.output=T.concatenate([self.triggerRnnTruncated,self.rnnOutputTruncated],axis=1)
        self.params=[self.WordEmbLayer.params,self.EntityEmdLayer.params,self.DistanceEmdLayer.params]
class OutputLayer(object):
    def __init__(self,rng,in_dim,op_dim,input):
        W_ar = numpy.asarray(rng.uniform(low=-numpy.sqrt(6 / (in_dim + op_dim)),
                                         high=numpy.sqrt(6 / (in_dim + op_dim)),
                                         size=(in_dim , op_dim))
                             , dtype=theano.config.floatX
                             )
        W = theano.shared(value=W_ar, name='W', borrow=True)
        b_ar = numpy.zeros((op_dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_ar, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params=[self.W,self.b]
        self.input = input
        self.prob = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.prob, axis=1)
    def predict(self):
        return self.y_pred
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.prob)[T.arange(y.shape[0]), y])
    def errors(self,y):
        return T.mean(T.neq(self.y_pred,y))


class Structured_OutputLayer(object):
    def __init__(self,rng, input, in_dim, op_dim):
        self.n_out = op_dim
        self.W = theano.shared(randomMatrix(in_dim, op_dim))
        self.W_tags = theano.shared(randomMatrix(op_dim, op_dim))
        self.b = theano.shared(randomArray(op_dim))
        self.t_0 = theano.shared(randomArray(op_dim))

        def rec_scores(x_t):
            s_t = (T.dot(x_t, self.W) + self.b)
            return s_t

        s, _ = theano.scan(fn=rec_scores, \
                           sequences=input)

        self.p_y_given_x = s
        self.params = [self.W ,self.b, self.W_tags]

    def theano_logsumexp(self, x, axis=None):
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def score_tag_path(self, y):
        def rec_obj(s_t, t_n1, t_n):
            acc = self.W_tags[t_n1, t_n] + s_t[t_n]
            return acc

        objs, _ = theano.scan(fn=rec_obj, \
                              sequences=[self.p_y_given_x[1:], dict(input=y, taps=[-1, 0])],
                              )
        return self.p_y_given_x[0, y[0]] + T.sum(objs)

    def rec_forward(self, obs, d_t1, trans):
        d_t1 = d_t1.dimshuffle(0, 'x')
        out = obs + self.theano_logsumexp(d_t1 + trans, axis=0)
        return out

    def rec_viterbi(self, obs, d_t1, trans):
        d_t1 = d_t1.dimshuffle(0, 'x')
        out_1 = obs + (d_t1 + trans).max(axis=0)
        out_2 = (d_t1 + trans).argmax(axis=0)
        return out_1, out_2

    def negative_log_likelihood(self, y):
        logsum, _ = theano.scan(fn=self.rec_forward, \
                                sequences=self.p_y_given_x[1:], outputs_info=self.p_y_given_x[0],
                                non_sequences=self.W_tags \
                                )
        return - self.score_tag_path(y) + self.theano_logsumexp(logsum[-1])

    def decode_forward(self):
        [max, argmax], _ = theano.scan(fn=self.rec_viterbi, \
                                       sequences=self.p_y_given_x[1:], outputs_info=[self.p_y_given_x[0], None],
                                       non_sequences=self.W_tags \
                                       )
        return max, argmax
    def predict(self):
        return self.decode_forward()
    def errors(self,y):
        return self.negative_log_likelihood(y)


class EmbedLayer(object):
    def __init__(self,VocabSize,embSize,wordids,embPresent,wordVocab={}):
        self.VocabSize=VocabSize
        self.embSize=embSize
        if embPresent:
            self.E=theano.shared(loadWordEmbeddings(wordVocab,embSize))
        else:
            self.E=theano.shared(value=np.asarray(np.random.uniform(low = -1,
							high = 1,
							size=(self.VocabSize,self.embSize)), dtype=theano.config.floatX), borrow=True)
        self.params=self.E
        self.output=self.E[wordids]
    def lookUp(self,wordids):
        return self.E[wordids]
class Rnn(object):
    def __init__(self,rng,dim,hidden):
        self.Wxf = theano.shared(randomMatrix(dim, hidden))
        self.Whf = theano.shared(randomMatrix(hidden, hidden))
        self.bhf = theano.shared(np.zeros(hidden, dtype=theano.config.floatX))
        self.hf_0 = theano.shared(randomArray(hidden))
        self.Wxb = theano.shared(randomMatrix(dim, hidden))
        self.Whb = theano.shared(randomMatrix(hidden, hidden))
        self.bhb = theano.shared(np.zeros(hidden, dtype=theano.config.floatX))
        self.hb_0 = theano.shared(randomArray(hidden))
        self.hidden=hidden
        self.dim=dim
        self.params=[self.Wxf,self.Whf,self.bhf,self.Wxb,self.Whb,self.bhb,self.hf_0,self.hb_0]
    def rnn_ff(self,inps):
        # model.container['bi_h0']  = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

        '''print "RNN"
        print Wx.shape.eval()
        print Wh.shape.eval()
        print bh.shape.eval()'''

        # bundle
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wxf) + T.dot(h_tm1, self.Whf) + self.bhf)
            return h_t

        h_f, _ = theano.scan(fn=recurrence, \
                           sequences=inps, outputs_info=[self.hf_0], n_steps=inps.shape[0])

        return h_f
    def rnn_bb(self,inps):
        inps=inps[::-1,:]
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wxb) + T.dot(h_tm1, self.Whb) + self.bhb)
            return h_t

        h_b, _ = theano.scan(fn=recurrence, \
                           sequences=inps, outputs_info=[self.hb_0], n_steps=inps.shape[0])
        h_b=h_b[::-1,:]
        return h_b
    def rnn_bi(self,inps):
        h_f=self.rnn_ff(inps)
        h_b=self.rnn_bb(inps)
        h=T.concatenate([h_f,h_b],axis=1)
        return h
class Lstm(object):
    def __init__(self,rng,dim,hidden):
        self.Wxf_i = theano.shared(randomMatrix(dim, hidden))
        self.Wxf_f = theano.shared(randomMatrix(dim,hidden))
        self.Wxf_o = theano.shared(randomMatrix(dim,hidden))
        self.Wxf_g = theano.shared(randomMatrix(dim,hidden))

        self.Whf_i = theano.shared(randomMatrix(hidden, hidden))
        self.Whf_f = theano.shared(randomMatrix(hidden, hidden))
        self.Whf_o = theano.shared(randomMatrix(hidden, hidden))
        self.Whf_g = theano.shared(randomMatrix(hidden, hidden))

        self.bhf_i = theano.shared(randomArray(hidden))
        self.bhf_f = theano.shared(randomArray(hidden))
        self.bhf_o = theano.shared(randomArray(hidden))
        self.bhf_g = theano.shared(randomArray(hidden))

        self.hf_0 = theano.shared(randomArray(hidden))
        self.cf_0 = theano.shared(randomArray(hidden))


        self.Wxb_i = theano.shared(randomMatrix(dim, hidden))
        self.Wxb_f = theano.shared(randomMatrix(dim, hidden))
        self.Wxb_o = theano.shared(randomMatrix(dim, hidden))
        self.Wxb_g = theano.shared(randomMatrix(dim, hidden))

        self.Whb_i = theano.shared(randomMatrix(hidden, hidden))
        self.Whb_f = theano.shared(randomMatrix(hidden, hidden))
        self.Whb_o = theano.shared(randomMatrix(hidden, hidden))
        self.Whb_g = theano.shared(randomMatrix(hidden, hidden))

        self.bhb_i = theano.shared(randomArray(hidden))
        self.bhb_f = theano.shared(randomArray(hidden))
        self.bhb_o = theano.shared(randomArray(hidden))
        self.bhb_g = theano.shared(randomArray(hidden))

        self.hb_0 = theano.shared(randomArray(hidden))
        self.cb_0 = theano.shared(randomArray(hidden))

        self.params = [self.Wxf_i,self.Wxf_f,self.Wxf_o,self.Wxf_g,self.Whf_i,self.Whf_f,self.Whf_o,self.Whf_g,self.bhf_i,self.bhf_f,self.bhf_o,self.bhf_g,
                       self.Wxb_i, self.Wxb_f, self.Wxb_o, self.Wxb_g, self.Whb_i, self.Whb_f, self.Whb_o, self.Whb_g,
                       self.bhb_i, self.bhb_f, self.bhb_o, self.bhb_g,self.hf_0,self.cf_0,self.hb_0,self.cb_0]
    def rnn_ff(self,inps):
        def recurrence(x_t, h_tm1, c_tm1):
            i = T.nnet.sigmoid( T.dot(x_t, self.Wxf_i) + T.dot(h_tm1,self.Whf_i) + self.bhf_i)
            f = T.nnet.sigmoid( T.dot(x_t, self.Wxf_f) + T.dot(h_tm1,self.Whf_f) + self.bhf_f)
            o = T.nnet.sigmoid( T.dot(x_t, self.Wxf_o) + T.dot(h_tm1,self.Whf_o) + self.bhf_o)
            g = T.tanh( T.dot(x_t, self.Wxf_g) + T.dot(h_tm1,self.Whf_g) + self.bhf_g)
            c_t = c_tm1 * f + g * i
            h_t = T.tanh(c_t) * o
            return [h_t,c_t]
        [h_f,c_f], _= theano.scan(fn=recurrence,\
                              sequences=inps, outputs_info= [self.hf_0,self.cf_0]\
                              )
        return h_f
    def rnn_bb(self,inps):
        inps=inps[::-1,:]
        def recurrence(x_t, h_tm1, c_tm1):
            i = T.nnet.sigmoid( T.dot(x_t, self.Wxb_i) + T.dot(h_tm1,self.Whb_i) + self.bhb_i)
            f = T.nnet.sigmoid( T.dot(x_t, self.Wxb_f) + T.dot(h_tm1,self.Whb_f) + self.bhb_f)
            o = T.nnet.sigmoid( T.dot(x_t, self.Wxb_o) + T.dot(h_tm1,self.Whb_o) + self.bhb_o)
            g = T.tanh( T.dot(x_t, self.Wxb_g) + T.dot(h_tm1,self.Whb_g) + self.bhb_g)
            c_t = c_tm1 * f + g * i
            h_t = T.tanh(c_t) * o
            return [h_t,c_t]
        [h_b,c_b], _= theano.scan(fn=recurrence,\
                              sequences=inps, outputs_info= [self.hb_0,self.cb_0]\
                              )
        h_b = h_b[::-1, :]
        return h_b
    def rnn_bi(self,inps):
        h_f = self.rnn_ff(inps)
        h_b = self.rnn_bb(inps)
        h = T.concatenate([h_f, h_b], axis=1)
        return h
class Gru(object):
    def __init__(self,rng,dim,hidden):
        self.Wxf_r=theano.shared(randomMatrix(dim, hidden))
        self.Wxf_z = theano.shared(randomMatrix(dim, hidden))
        self.Wxf_f = theano.shared(randomMatrix(dim, hidden))

        self.Whf_r = theano.shared(randomMatrix(hidden, hidden))
        self.Whf_z = theano.shared(randomMatrix(hidden, hidden))
        self.Whf = theano.shared(randomMatrix(hidden, hidden))

        self.bf_r = theano.shared(randomArray(hidden))
        self.bf_z = theano.shared(randomArray(hidden))
        self.bf_t = theano.shared(randomArray(hidden))
        self.bf_0 = theano.shared(randomArray(hidden))


        self.Wxb_r = theano.shared(randomMatrix(dim, hidden))
        self.Wxb_z = theano.shared(randomMatrix(dim, hidden))
        self.Wxb_f = theano.shared(randomMatrix(dim, hidden))

        self.Whb_r = theano.shared(randomMatrix(hidden, hidden))
        self.Whb_z = theano.shared(randomMatrix(hidden, hidden))
        self.Whb = theano.shared(randomMatrix(hidden, hidden))

        self.bb_r = theano.shared(randomArray(hidden))
        self.bb_z = theano.shared(randomArray(hidden))
        self.bb_t = theano.shared(randomArray(hidden))
        self.bb_0 = theano.shared(randomArray(hidden))

        self.params =[self.Wxf_r,self.Wxf_z,self.Wxf_f,self.Whf_r,self.Whf_z,self.Whf,self.bf_r,self.bf_z,self.bf_t,self.bf_0,
                      self.Wxb_r,self.Wxb_z,self.Wxb_f,self.Whb_r,self.Whb_z,self.Whb,self.bb_r,self.bb_z,self.bb_t,self.bb_0]
    def rnn_ff(self, inps):
        def recurrence(x_t, h_tm1):
            r_t = T.nnet.sigmoid( T.dot(x_t, self.Wxf_r) + T.dot(h_tm1,self.Whf_r) + self.bf_r)
            z_t = T.nnet.sigmoid( T.dot(x_t,self.Wxf_z) + T.dot(h_tm1,self.Whf_z) + self.bf_z)
            h_tilde = T.tanh( T.dot(x_t,self.Wxf_f) + T.dot(r_t * h_tm1 , self.Whf)+ self.bf_t)
            h_t = (1. - z_t) * h_tm1 + z_t * h_tilde
            return h_t
        h_f, _ =theano.scan(fn=recurrence,\
                          sequences=inps,
                          outputs_info=self.bf_0,\
                          )
        return h_f

    def rnn_bb(self, inps):
        inps = inps[::-1, :]
        def recurrence(x_t, h_tm1):
            r_t = T.nnet.sigmoid( T.dot(x_t, self.Wxb_r) + T.dot(h_tm1,self.Whb_r) + self.bb_r)
            z_t = T.nnet.sigmoid( T.dot(x_t,self.Wxb_z) + T.dot(h_tm1,self.Whb_z) + self.bb_z)
            h_tilde = T.tanh( T.dot(x_t,self.Wxb_f) + T.dot(r_t * h_tm1 , self.Whb)+ self.bb_t)
            h_t = (1. - z_t) * h_tm1 + z_t * h_tilde
            return h_t
        h_b, _ =theano.scan(fn=recurrence,\
                          sequences=inps,
                          outputs_info=self.bb_0,\
                          )
        h_b = h_b[::-1,:]
        return h_b
    def rnn_bi(self, inps):
        h_f = self.rnn_ff(inps)
        h_b = self.rnn_bb(inps)
        h = T.concatenate([h_f, h_b], axis=1)
        return h
class MyArguementModel(object):
    def __init__(self,rng,wordVocab,entityVocab,labelVocab,distanceVocab,embSizeWord,embSizeEntity,embSizeDistance,RnnHiddenDim,FFhiddenLayerDim,inputWord,input1,input2,input3,input4,dropout):
        self.wordVocab=wordVocab
        self.entityVocab=entityVocab
        self.labelVocab=labelVocab
        self.distanceVocab=distanceVocab
        self.wordVocabSize=len(wordVocab)
        self.entityVocabSize = len(entityVocab)
        self.labelVocabSize = len(labelVocab)
        self.distanceVocabSize=len(distanceVocab)
        self.embSizeWord = embSizeWord
        self.embSizeEntity = embSizeEntity
        self.embSizeDistance=embSizeDistance
        self.inpDim = self.embSizeWord + self.embSizeEntity +self.embSizeDistance
        self.RnnhidDim = RnnHiddenDim
        self.FFhiddenLayerDim=FFhiddenLayerDim
        # self.rnnFeature=Rnn(rng=rng,dim=self.inpDim,hidden=self.RnnhidDim)
        # self.rnnFeature = Lstm(rng=rng, dim=self.inpDim, hidden=self.RnnhidDim)
        self.rnnFeature = Gru(rng=rng, dim=self.inpDim, hidden=self.RnnhidDim)
        self.finalDim = 4*self.RnnhidDim + 2*(self.embSizeWord + self.embSizeEntity)
        self.inputLayer = InputLayer(wordVocabSize=self.wordVocabSize, wordVocab=self.wordVocab,entityVocabSize=self.entityVocabSize, entityVocab=self.entityVocab,distanceVocabSize=self.distanceVocabSize,distanceVocab=self.distanceVocab,embSizeWord=self.embSizeWord, embSizeEntity=self.embSizeEntity, embSizeDistance=self.embSizeDistance,inputWord=inputWord,input1=input1,input2=input2,input3=input3, input4=input4,rnnFeature=self.rnnFeature)
        self.hiddenLayer=HiddenLayer(rng=rng,in_dim=self.finalDim,op_dim=self.FFhiddenLayerDim,input=self.inputLayer.output)
        self.hiddenLayerWithDropout=Dropout(X=self.hiddenLayer.output,p=dropout)
        self.hiddenLayer1 = HiddenLayer(rng=rng, in_dim=self.FFhiddenLayerDim, op_dim=200,input=self.hiddenLayerWithDropout.output)
        self.hiddenLayer1WithDropout = Dropout(X=self.hiddenLayer1.output, p=dropout)
        #self.outputLayer = OutputLayer(rng=rng, in_dim=200, op_dim=self.labelVocabSize,input=self.hiddenLayer1WithDropout.output)
        self.outputLayer = Structured_OutputLayer(rng=rng, in_dim=200, op_dim=self.labelVocabSize,
                                       input=self.hiddenLayer1WithDropout.output)
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood
        self.errors = self.outputLayer.errors
        self.predict = self.outputLayer.predict
        self.params = self.inputLayer.params + self.outputLayer.params + self.rnnFeature.params + self.hiddenLayer.params + self.hiddenLayer1.params
        '''self.L1 = (
                    abs(self.hiddenLayer1.W).sum()
                    + abs(self.hiddenLayer2.W).sum()
                    + abs(self.outputLayer.W).sum()
                    + abs(self.rnnFeature.Whf).sum()
                    + abs(self.rnnFeature.Whb).sum()
                    + abs(self.rnnFeature.Wxf).sum()
                    + abs(self.rnnFeature.Wxb).sum()
                    + abs(self.rnnFeature.hf_0).sum()
                    + abs(self.rnnFeature.hb_0).sum()
                )
                self.L2_sqr = (
                    (self.hiddenLayer1.W ** 2).sum()
                    + (self.hiddenLayer2.W ** 2).sum()
                    + (self.outputLayer.W ** 2).sum()
                    + (self.rnnFeature.Whf ** 2).sum()
                    + (self.rnnFeature.Whb ** 2).sum()
                    + (self.rnnFeature.Wxf ** 2).sum()
                    + (self.rnnFeature.Wxb ** 2).sum()
                    + (self.rnnFeature.hf_0 ** 2).sum()
                    + (self.rnnFeature.hb_0 ** 2).sum()
                )'''
        '''self.L1 = (
                    abs(self.hiddenLayer1.W).sum()
                    + abs(self.hiddenLayer2.W).sum()
                    + abs(self.outputLayer.W).sum()
                    + abs(self.rnnFeature.Wxf_i).sum()
                    + abs(self.rnnFeature.Wxf_f).sum()
                    + abs(self.rnnFeature.Wxf_o).sum()
                    + abs(self.rnnFeature.Wxf_g).sum()
                    + abs(self.rnnFeature.Whf_i).sum()
                    + abs(self.rnnFeature.Whf_f).sum()
                    + abs(self.rnnFeature.Whf_o).sum()
                    + abs(self.rnnFeature.Whf_g).sum()
                    + abs(self.rnnFeature.bhf_i).sum()
                    + abs(self.rnnFeature.bhf_f).sum()
                    + abs(self.rnnFeature.bhf_o).sum()
                    + abs(self.rnnFeature.bhf_g).sum()
                    + abs(self.rnnFeature.Wxb_i).sum()
                    + abs(self.rnnFeature.Wxb_f).sum()
                    + abs(self.rnnFeature.Wxb_o).sum()
                    + abs(self.rnnFeature.Wxb_g).sum()
                    + abs(self.rnnFeature.Whb_i).sum()
                    + abs(self.rnnFeature.Whb_f).sum()
                    + abs(self.rnnFeature.Whb_o).sum()
                    + abs(self.rnnFeature.Whb_g).sum()
                    + abs(self.rnnFeature.bhb_i).sum()
                    + abs(self.rnnFeature.bhb_f).sum()
                    + abs(self.rnnFeature.bhb_o).sum()
                    + abs(self.rnnFeature.bhb_g).sum()
                    + abs(self.rnnFeature.hf_0).sum()
                    + abs(self.rnnFeature.cf_0).sum()
                    + abs(self.rnnFeature.hb_0).sum()
                    + abs(self.rnnFeature.cb_0).sum()
                )
        self.L2_sqr = (
                abs(self.hiddenLayer1.W ** 2).sum()
                + abs(self.hiddenLayer2.W ** 2).sum()
                + abs(self.outputLayer.W ** 2).sum()
                + abs(self.rnnFeature.Wxf_i ** 2).sum()
                + abs(self.rnnFeature.Wxf_f ** 2).sum()
                + abs(self.rnnFeature.Wxf_o ** 2).sum()
                + abs(self.rnnFeature.Wxf_g ** 2).sum()
                + abs(self.rnnFeature.Whf_i ** 2).sum()
                + abs(self.rnnFeature.Whf_f ** 2).sum()
                + abs(self.rnnFeature.Whf_o ** 2).sum()
                + abs(self.rnnFeature.Whf_g ** 2).sum()
                + abs(self.rnnFeature.bhf_i ** 2).sum()
                + abs(self.rnnFeature.bhf_f ** 2).sum()
                + abs(self.rnnFeature.bhf_o ** 2).sum()
                + abs(self.rnnFeature.bhf_g ** 2).sum()
                + abs(self.rnnFeature.Wxb_i ** 2).sum()
                + abs(self.rnnFeature.Wxb_f ** 2).sum()
                + abs(self.rnnFeature.Wxb_o ** 2).sum()
                + abs(self.rnnFeature.Wxb_g ** 2).sum()
                + abs(self.rnnFeature.Whb_i ** 2).sum()
                + abs(self.rnnFeature.Whb_f ** 2).sum()
                + abs(self.rnnFeature.Whb_o ** 2).sum()
                + abs(self.rnnFeature.Whb_g ** 2).sum()
                + abs(self.rnnFeature.bhb_i ** 2).sum()
                + abs(self.rnnFeature.bhb_f ** 2).sum()
                + abs(self.rnnFeature.bhb_o ** 2).sum()
                + abs(self.rnnFeature.bhb_g ** 2).sum()
                + abs(self.rnnFeature.hf_0 ** 2).sum()
                + abs(self.rnnFeature.cf_0 ** 2).sum()
                + abs(self.rnnFeature.hb_0 ** 2).sum()
                + abs(self.rnnFeature.cb_0 ** 2).sum()
        )'''
        '''self.L1 = (
            abs(self.rnnFeature.Wxf_r ** 2).sum() +
            abs(self.rnnFeature.Wxf_z ** 2).sum() +
            abs(self.rnnFeature.Wxf_f ** 2).sum() +
            abs(self.rnnFeature.Whf_r ** 2).sum() +
            abs(self.rnnFeature.Whf_z ** 2).sum() +
            abs(self.rnnFeature.Whf ** 2).sum() +
            abs(self.rnnFeature.bf_r ** 2).sum() +
            abs(self.rnnFeature.bf_z ** 2).sum() +
            abs(self.rnnFeature.bf_t ** 2).sum() +
            abs(self.rnnFeature.bf_0 ** 2).sum() +
            abs(self.rnnFeature.Wxb_r ** 2).sum() +
            abs(self.rnnFeature.Wxb_z ** 2).sum() +
            abs(self.rnnFeature.Wxb_f ** 2).sum() +
            abs(self.rnnFeature.Whb_r ** 2).sum() +
            abs(self.rnnFeature.Whb_z ** 2).sum() +
            abs(self.rnnFeature.Whb ** 2).sum() +
            abs(self.rnnFeature.bb_r ** 2).sum() +
            abs(self.rnnFeature.bb_z ** 2).sum() +
            abs(self.rnnFeature.bb_t ** 2).sum() +
            abs(self.rnnFeature.bb_0 ** 2).sum()

        )
        self.L2_sqr = (
            abs(self.rnnFeature.Wxf_r ** 2).sum() +
            abs(self.rnnFeature.Wxf_z ** 2).sum() +
            abs(self.rnnFeature.Wxf_f ** 2).sum() +
            abs(self.rnnFeature.Whf_r ** 2).sum() +
            abs(self.rnnFeature.Whf_z ** 2).sum() +
            abs(self.rnnFeature.Whf ** 2).sum() +
            abs(self.rnnFeature.bf_r ** 2).sum() +
            abs(self.rnnFeature.bf_z ** 2).sum() +
            abs(self.rnnFeature.bf_t ** 2).sum() +
            abs(self.rnnFeature.bf_0 ** 2).sum() +
            abs(self.rnnFeature.Wxb_r ** 2).sum() +
            abs(self.rnnFeature.Wxb_z ** 2).sum() +
            abs(self.rnnFeature.Wxb_f ** 2).sum() +
            abs(self.rnnFeature.Whb_r ** 2).sum() +
            abs(self.rnnFeature.Whb_z ** 2).sum() +
            abs(self.rnnFeature.Whb ** 2).sum() +
            abs(self.rnnFeature.bb_r ** 2).sum() +
            abs(self.rnnFeature.bb_z ** 2).sum() +
            abs(self.rnnFeature.bb_t ** 2).sum() +
            abs(self.rnnFeature.bb_0 ** 2).sum()

        )'''

def compute(sentence,entities,distances,labels,model):
    input1=wordsToIndexes1(sentence,model.wordVocab,True,False)
    #print input1
    input2=wordsToIndexes1(entities,model.entityVocab,False,True)
    #print input2
    input3=wordsToIndexes1(distances,model.distanceVocab,False,False)
    #print input3
    labels=wordsToIndexes1(labels,model.labelVocab,False,False)
    #print labels
    '''print sentence
    print input1
    print entities
    print input2'''
    assert (len(input1)==len(input2)==len(input3))
    return input1,input2,input3,labels
