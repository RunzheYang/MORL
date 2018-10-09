###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

'''
basic.py - Define basic functions and classes for RNNLG
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017
 
************************
'''
import numpy as np
import theano
import math
import theano.tensor as T

# numerical stability
eps = 1e-7

def softmax(w):
    e = np.exp(w)
    dist = e/np.sum(e)
    return dist

def sigmoid(w):
    e = np.exp(-w)
    acti = 1/(1+e)
    return acti

def tanh(w):
    return np.tanh(w)

# gradient clipping 
class GradClip(theano.compile.ViewOp):
    '''
    Theano implementation of gradient clipping
    '''
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]

def clip_gradient(x, bound):
    '''
    Gradient clipping
    '''
    grad_clip = GradClip(-bound, bound)
    try:
        T.opt.register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip_%.1f' % (bound))
    except ValueError:
        pass
    return grad_clip(x)

# obtain sent logprob by summing over word logprob
def collectSentLogp(p,cutoff_t,cutoff_b):
    '''
    Get sentence log prob for a minbatch
    '''
    q = p.dimshuffle(1,0)
    def sump(p_b,stop_t):
        logp = T.sum(T.log10(p_b[:stop_t]))
        return logp
    cutoff_logp, _ = theano.scan(fn=sump,\
        sequences=[q[:cutoff_b],cutoff_t[:cutoff_b]],\
        outputs_info=[None])
    return cutoff_logp

# Node class for performing beam search
class BeamSearchNode(object):
    '''
    Beam search node for RNNLG decoding
    '''
    def __init__(self,h,c,prevNode,wordid,logp,leng):
        self.h      = h
        self.c      = c
        self.logp   = logp
        self.leng   = leng
        self.wordid = wordid
        self.prevNode = prevNode
        self.sv = None
    
    def eval(self):
        '''
        Node scoring function
        '''
        if self.leng>40:
            return self.logp/float(self.leng-1+eps)-40.0
        return self.logp/float(self.leng-1+eps)

# basic class for Recurrent Language Generator
class BaseRLG(object):
    '''
    Basic class for RNNLG models. Input arguments usually define the structure of the NNs. However the actual parameters depends on the model architecture used.
    '''
    def __init__(self, gentype, beamwidth, overgen,
            vocab_size, hidden_size, batch_size, da_sizes):
        # setting hyperparameters
        self.gentype= gentype
        self.di     = vocab_size
        self.dh     = hidden_size
        self.db     = batch_size
        self.dfs    = da_sizes
        self.overgen= overgen
        self.beamwidth = beamwidth

    def _init_params(self):
        '''
        Function for initialising the weight parameters.
        '''
        pass

    def unroll(self):
        '''
        The unroll function in Theano for RNNs while training.
        '''
        pass

    def _recur(self):
        '''
        Per step recurrence function in Theano during training, called by unroll function
        '''
        pass

    def beamSearch(self):
        '''
        The generation function in numpy using beam search decoding
        '''
        pass

    def sample(self):
        '''
        The generation function in numpy using ramdom sampling
        '''
        pass

    def _gen(self):
        '''
        Per step generation function in numpy while testing
        '''
        pass
    
    def loadConverseParams(self):
        '''
        Function which loads numpy parameters
        '''
        pass
   
    def setParams(self,params):
        '''
        Function that sets theano parameters
        '''
        for i in range(len(self.params)):
            self.params[i].set_value(params[i])

    def getParams(self):
        '''
        Function that fetches theano parameters
        '''
        return [p.get_value() for p in self.params]

    def numOfParams(self):
        '''
        Report the number of parameters used
        '''
        return sum([p.get_value().size for p in self.params])
 
