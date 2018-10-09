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
GPLib.py - Gaussian Process SARSA algorithm
=============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

This module encapsulates all classes and functionality which implement the GPSARSA algorithm for dialogue learning.

   
**Relevant Config variables** [Default values].  X is the domain tag::

    [gpsarsa_X]
    saveasprior = False 
    random = False
    learning = False
    gamma = 1.0
    sigma = 5.0
    nu = 0.001
    scale = -1
    numprior = 0

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`policy.PolicyUtils`

************************

'''


__author__ = "cued_dialogue_systems_group"

import numpy as np
import math
import scipy.stats
import pickle as pkl
import os.path
import copy
#from profilehooks import profile

import PolicyUtils
import GPPolicy
from utils import Settings
from utils import ContextLogger
logger = ContextLogger.getLogger('')

class LearnerInterface:
    '''
    This class defines the basic interface for the GPSARSA algorithm.
    
    specifies the policy files
    self._inputDictFile input dictionary file
    self._inputParamFile input parameter file
    self._outputDictFile output dictionary file
    self._outputParamFile output parameter file

    self.initial self.terminal flags are needed for learning to specify initial and terminal states in the episode
    '''
    def __init__(self):

        self._inputDictFile = ""
        self._inputParamFile = ""
        self._outputDictFile = ""
        self._outputParamFile = ""
        self._actionSize = 0
        self.initial = False
        self.terminal = False

    def DictionarySize(self):
        return 0
    def savePolicy(self, index):
        pass

    def readPolicy(self):
        pass

    def getValue(self, state, action):
        return [0, 0]

    def policy(self, state, kernel=None, executable = []):
        pass

    def LearningStep(self, pstate, paction, r, state, action, kernel):
        pass

class GPSARSAPrior(LearnerInterface):
    '''
    Defines the GP prior. Derives from LearnerInterface.
    '''
    def __init__(self,in_policyfile, out_policyfile, numPrior=-1, learning=False, domainString=None,sharedParams=None):
        self._inpolicyfile = in_policyfile
        self._outpolicyfile = out_policyfile
        self.domainString = domainString

        self.sharedParams = False
        if sharedParams is None:
            self.params = {}
        else:
            self.params = sharedParams
            self.sharedParams = True
        self._actionSize = 0
        self.initial = False
        self.terminal = False
        self.params['_dictionary'] = []
        self.params['_alpha_tilda'] = []
        self._inputDictFile =""
        self._inputParamFile = ""
        self._outputDictFile = ""
        self._outputParamFile = ""
        self._prior = None
        self._superior = None
        self.learning = learning
        self._random = False

        
        if numPrior >= 0:
            if numPrior > 0:
                self._prior = GPSARSAPrior(in_policyfile,out_policyfile ,numPrior-1)
                self._prior._superior = self

            else:
                self._prior = None

        self._inputDictFile = in_policyfile+"."+str(numPrior)+".prior.dct"
        self._inputParamFile = in_policyfile+"."+str(numPrior)+".prior.prm"
        self._outputDictFile = out_policyfile+"."+str(numPrior)+".prior.dct"
        self._outputParamFile = out_policyfile+"."+str(numPrior)+".prior.prm"

#         print self._inputDictFile
#         print self._inputParamFile
#         print self._outputDictFile
#         print self._outputParamFile

        if self._prior:
            self.readPolicy()


    def readParameters(self):
        """
        Reads input policy parameters
        """
        with open(self._inputParamFile, 'rb') as pkl_file:
            self.params['_alpha_tilda'] = pkl.load(pkl_file)

    def readDictionary(self):
        """
        Reads input policy dictionary file
        """
        with open(self._inputDictFile, 'rb') as pkl_file:
            self.params['_dictionary'] = pkl.load(pkl_file)
        logger.info("In Prior class: Read dictionary of size " + str(len(self.params['_dictionary'])))

    def DictionarySize(self):
        """
        :returns: number of dictionary points
        """
        return len(self.params['_dictionary'])

    def SavePolicy(self, index):
        """
        This function does not exist for prior only for full GP
        """
        pass

    def readPolicy(self):
        """
        Reads dictionary file and parameter file
        """
        if not os.path.isfile(self._inputDictFile) or not os.path.isfile(self._inputParamFile):
            if self.learning or self._random:
                logger.warning(self._inpolicyfile+" does not exist")
                #return
            else:
                #logger.error(self._inpolicyfile+" does not exist")
                logger.warning(self._inpolicyfile+" does not exist")
        else:
            self.readDictionary()
            self.readParameters()

    def QvalueMean(self, state, action, kernel):
        """
        :returns: Q value mean for a given state, action and the kernel function\
        Recursively calculates the mean given depending on the number of recursive priors
        """
        if self._prior != None :
            qprior = self._prior.QvalueMean()
        else:
            qprior = 0

        if len(self.params['_dictionary'])>0:
            k_tilda_t =self.k_tilda(state, action, kernel)
            qval = np.dot(self.params['_alpha_tilda'], k_tilda_t)
            mean = qprior+qval
        else:
            mean = qprior

        #if math.fabs(mean) > 100 :
        #    logger.error("Mean very large")

        return mean


    def k_tilda(self,state, action, kernel):
        """
        Based on product between action and belief kernels. 
        O(N) in dictionary size N.
        :returns: vector of kernel values of given state, action and kernel with all state-action pairs in the dictionary
        """
        res = []
        for [dstate, daction] in self.params['_dictionary']:
            actKer = kernel.ActionKernel(action, daction)
            if actKer > 0:
                if self._prior != None:
                    # TODO - is this correct? seems to not use prior and be same as else: statement
                    stateKer = kernel.PriorKernel(state, dstate)
                else:
                    stateKer = kernel.beliefKernel(state, dstate)


                res.append(actKer*stateKer)
            else:
                res.append(actKer)
        return np.array(res)


class GPSARSA(GPSARSAPrior):
    """
        Derives from GPSARSAPrior
    
       Implements GPSarsa algorithm where mean can have a predefined value
       self._num_prior specifies number of means
       self._prior specifies the prior
       If not specified a zero-mean is assumed

       Parameters needed to estimate the GP posterior
       self._K_tida_inv inverse of the Gram matrix of dictionary state-action pairs
       self.sharedParams['_C_tilda'] covariance function needed to estimate the final variance of the posterior
       self.sharedParams['_c_tilda'] vector needed to calculate self.sharedParams['_C_tilda']
       self.sharedParams['_alpha_tilda'] vector needed to estimate the mean of the posterior
       self.sharedParams['_d'] and self.sharedParams['_s'] sufficient statistics needed for the iterative estimation of the posterior

       Parameters needed for the policy selection
       self._random random policy choice
       self._scale scaling of the standard deviation when sampling Q-value, if -1 than taking the mean
       self.learning if true in learning mode
    """
    def __init__(self, in_policyfile, out_policyfile, domainString=None, learning=False, sharedParams=None):
        """
        Initialise the prior given policy files
        """
        GPSARSAPrior.__init__(self,in_policyfile,out_policyfile,-1,learning,domainString,sharedParams)

        self.save_as_prior = False
        self.initial = False
        self.terminal = False
        self.numpyFileFormat = False
        self.domainString = domainString
        self.sharedParams = False
        if sharedParams is None:
            self.params = {}
        else:
            self.params = sharedParams
            self.sharedParams = True
        
        self._gamma = 1.0
        self._sigma = 5.0
        self._nu = 0.001

        self.params['_K_tilda_inv'] = np.zeros((1, 1))
        self.params['_C_tilda'] = np.zeros((1, 1))
        self.params['_c_tilda'] = np.zeros(1)
        self.params['_a'] = np.ones(1)
        self.params['_alpha_tilda'] = np.zeros(1)
        self.params['_dictionary'] = []
        self.params['_d'] = 0.
        self.params['_s'] = float('inf')


        self._random = False
        self._num_prior = 0
        self._scale = -1
        self._prior = None
        self.beliefparametrisation = 'GP'
        # self.learning = learning => set in GPSARSAPrior __init__

        # parameter settings
        if Settings.config.has_option('gpsarsa', "saveasprior"):
            self.save_as_prior = Settings.config.getboolean('gpsarsa', "saveasprior")
        if Settings.config.has_option('gpsarsa', "numprior"):
            self._num_prior = Settings.config.getint('gpsarsa',"numprior")
        if Settings.config.has_option("gpsarsa", "random"):
            self._random = Settings.config.getboolean('gpsarsa', "random")
        if Settings.config.has_option('gpsarsa', "gamma"):
            self._gamma = Settings.config.getfloat('gpsarsa',"gamma")
        if Settings.config.has_option('gpsarsa', "sigma"):
            self._sigma = Settings.config.getfloat('gpsarsa',"sigma")
        if Settings.config.has_option('gpsarsa', "nu"):
            self._nu = Settings.config.getfloat('gpsarsa',"nu")
        if Settings.config.has_option('gpsarsa', "scale"):
            self._scale = Settings.config.getint('gpsarsa',"scale")
        if Settings.config.has_option("gpsarsa", "saveasnpy"):
            self.numpyFileFormat = Settings.config.getboolean('gpsarsa', "saveasnpy")

        # domain specific parameter settings (overrides general policy parameter settings)
        if Settings.config.has_option("gpsarsa_"+domainString, "saveasprior"):
            self.save_as_prior = Settings.config.getboolean("gpsarsa_"+domainString, "saveasprior")
        if Settings.config.has_option("gpsarsa_"+domainString, "numprior"):
            self._num_prior = Settings.config.getint("gpsarsa_"+domainString,"numprior")
        if Settings.config.has_option("gpsarsa_"+domainString, "random"):
            self._random = Settings.config.getboolean("gpsarsa_"+domainString, "random")
        if Settings.config.has_option("gpsarsa_"+domainString, "gamma"):
            self._gamma = Settings.config.getfloat("gpsarsa_"+domainString,"gamma")
        if Settings.config.has_option("gpsarsa_"+domainString, "sigma"):
            self._sigma = Settings.config.getfloat("gpsarsa_"+domainString,"sigma")
        if Settings.config.has_option("gpsarsa_"+domainString, "nu"):
            self._nu = Settings.config.getfloat("gpsarsa_"+domainString,"nu")
        if Settings.config.has_option("gpsarsa_"+domainString, "scale"):
            self._scale = Settings.config.getint("gpsarsa_"+domainString,"scale")
        if Settings.config.has_option("gpsarsa_"+domainString, "saveasnpy"):
            self.numpyFileFormat = Settings.config.getboolean("gpsarsa_"+domainString, "saveasnpy")


        if self._num_prior == 0:
            self._inputDictFile = in_policyfile+".dct"
            self._inputParamFile = in_policyfile + ".prm"
            self._outputDictFile = out_policyfile + ".dct"
            self._outputParamFile = out_policyfile + ".prm"
        else:
            # TODO -- FIX THIS to deal with prior. Likely need to check how prior is dealt with everywhere.
            self._inputDictFile = in_policyfile+".dct"
            self._inputParamFile = in_policyfile + ".prm"
            self._outputDictFile = out_policyfile + ".dct"
            self._outputParamFile = out_policyfile + ".prm"

        # load the parameters using self._input<X>
        self.readPolicy()

        if self._num_prior > 0:
            # the below usage of in and out files for policy could be a little confusing. Passing below the out_policyfile
            # as the input to the prior... on assumption that you just change the one config by first mentioning
            # saveasprior = True, and then alter it by saying numprior = 1, saveasprior = False, and not touching
            # inpolicyfile or outpolicyfile. This gives flexibility, but may be easier to just go back to having a single
            # policyfile (for in and out). 
            #self._prior = GPSARSAPrior(out_policyfile,'' ,self._num_prior-1)
            self._prior = GPSARSAPrior(in_policyfile,'' ,self._num_prior-1)
            self._prior._superior = self

    def ReInitialise(self):
        """
        :param: None
        :returns None:
        """
        if len(self._ditionary) > 0:
            self.params['_K_tilda_inv'] = np.array([])
            self.params['_C_tilda'] = np.array([])
            self.params['_c_tilda'] = np.array([])
            self.params['_a'] = np.array([])
            self.params['_alpha_tilda'] = np.array([])
            self.params['_dictionary'] = []
            self.params['_d'] = 0
            self.params['_s'] = float('inf')

        self.initial = True
        self.terminal = False

    def Reset(self):
        """
        Resets to new dialogue
        """
        self.initial = True
        self.terminal = False

    def QvalueMeanVar(self, state, action, kernel):
        """
        Gets mean and variance of Q-value at (S,A) pair for given kernel
        :param: state
        :param: action
        :param: kernel
        :returns: mean and variance of GP for given state, action and kernel
        """
        qprior = 0
        qval = 0
        qvar = 0

        #TODO come back to this
        if self._prior != None:
            qprior = self._prior.QvalueMean(state, action, kernel)

        if len(self.params['_dictionary']) > 0 :
            k_tilda_t = self.k_tilda(state,action,kernel)
            qval = np.dot(k_tilda_t, self.params['_alpha_tilda'])
            qvar = np.dot(k_tilda_t, np.dot(self.params['_C_tilda'], k_tilda_t))

        mean = qprior + qval        
        if self._prior != None:
            qorg = kernel.PriorKernel(state, state)*kernel.ActionKernel(action,action)
        else:
            qorg = kernel.beliefKernel(state,state)*kernel.ActionKernel(action,action)
        var = qorg - qvar

        if var < 0:
            #print '-\n'*5   # TODO - deal with this better - put this print statement so that when it occurs you DO SEE IT!
            #logger.warning("Variance negative "+str(var))
            var =0
        #if math.fabs(mean) > 100:
        #    logger.error("Mean very large "+str(mean))

        return [mean, var]


    def getPriorVar(self, state, action, kernel):
        """
        :returns: prior variance for given state, action and kernel
        """
        if self._prior != None:
            priorVar = kernel.PriorKernel(state,state)*kernel.ActionKernel(action,action)
        else:
            priorVar = kernel.beliefKernel(state,state)*kernel.ActionKernel(action,action)
        return (self._scale**2)*priorVar

    def getLiklihood_givenValue(self, state, action, kernel, sampledQvalue):
        [mean, variance] = self.QvalueMeanVar(state, action, kernel)
        gaussvar = self._scale * math.sqrt(variance)
        likelihood = scipy.stats.norm(mean,gaussvar).pdf(sampledQvalue)
        return likelihood

    def policy(self, state, kernel, executable=[]):
        """
        :returns: best action according to sampled Q values
        """        
        
        if len(executable)==0:
            logger.error("No executable actions")
        
        if self._random:
            #Settings.random.shuffle(executable)        -- can be random.choose()  
            #print "Choosing randomly ", executable[0].act
            #return executable TODO - needs to be in new format: [best_action, best_actions_sampled_Q_value, actions_likelihood]
            return [Settings.random.choice(executable).act, 0,0]

        Q =[]
        for action in executable:
            if self._scale <= 0:
                [mean, var] = self.QvalueMeanVar(state, action, kernel)
                logger.debug('action: ' +str(action.act) + ' mean then var:\t\t\t ' + str(mean) + '  ' + str(math.sqrt(var)))
                value = mean
                gaussvar = 0
            else:
                [mean, var ] = self.QvalueMeanVar(state, action, kernel)                
                gaussvar = self._scale * math.sqrt(var)                        
                value = gaussvar * Settings.random.randn() + mean     # Sample a Q value for this action
                logger.debug('action: ' +str(action.act) + ' mean then var:\t\t\t ' + str(mean) + '  ' + str(gaussvar))
            Q.append((action, value, mean, gaussvar))

        Q=sorted(Q,key=lambda Qvalue : Qvalue[1], reverse=True)
        
        # DEBUG STATEMENTS:
        #print "Choosing best ",Q[0][0].act," from ", len(Q), " actions"
        #for item in Q:
        #    print item[0].act, " ", str(item[1])
        #return [ordered_actions[0] for ordered_actions in Q]
        
        best_action, best_actions_sampled_Q_value = Q[0][0], Q[0][1]
        actions_likelihood = 0
        if Q[0][3] != 0:
            actions_likelihood = scipy.stats.norm(Q[0][2], Q[0][3]).pdf(best_actions_sampled_Q_value)
        return [best_action, best_actions_sampled_Q_value, actions_likelihood]


    def Extend(self,delta_prev, pstate, paction):
        """
        Add points pstate and paction in the dictionary and extend sufficient statistics matrices and vectors for one dimension
        Only used for the first state action pair in the episode
        """
        _a_prev = np.zeros(len(self.params['_dictionary']) + 1)
        _a_prev[-1] =1.0
        _c_tilda_prev = np.zeros(len(self.params['_dictionary']) + 1)
        _K_tilda_inv_prev = self.extendKtildainv(self.params['_K_tilda_inv'], self.params['_a'], delta_prev)
        _alpha_tilda_prev = self.extendVector(self.params['_alpha_tilda'])
        _C_tilda_prev = self.extendMatrix(self.params['_C_tilda'])

        self.params['_a'] = _a_prev
        self.params['_alpha_tilda'] = _alpha_tilda_prev
        self.params['_c_tilda'] = _c_tilda_prev
        self.params['_C_tilda'] = _C_tilda_prev
        self.params['_K_tilda_inv'] = _K_tilda_inv_prev

        self.params['_dictionary'].append([pstate, paction])
        #self.checkKtildainv(kernel)

    def extendMatrix(self, M):
        """
        Extend the dimentionality of matrix by one row and column -- new elements are zeros.
        """
        lenM = len(M[0])
        M_new = np.zeros((lenM+1, lenM+1))
        M_new[:lenM,:lenM] = M
        return M_new


    def extendVector(self, v):
        """
        Extend the dimensionality of vector by one element
        """
        lenV = len(v)
        v_new = np.zeros(lenV+1)
        v_new[:lenV] = v
        return v_new
    
    def extendKtildainv(self, K_tilda_inv, a, delta_new):
        """
        # grows nxn -> n+1xn+1 where n is dict size
        :returns: inverse of the Gram matrix using the previous Gram matrix and partition inverse theorem
        """
        lenD = len(self.params['_dictionary'])
        K_tilda_inv_new = np.zeros((lenD+1,lenD+1))   
        K_tilda_inv_new[:lenD,:lenD] = K_tilda_inv + np.outer(a,a)/delta_new
        K_tilda_inv_new[:lenD,lenD] = -a/delta_new      # new col
        K_tilda_inv_new[lenD,:lenD] = -a/delta_new      # new row
        K_tilda_inv_new[-1][-1] = 1/delta_new           # new corner
        return K_tilda_inv_new


    def ExtendNew(self, delta_new, state, action, kernel, _a_new, k_tilda_prev, k_tilda_new, delta_k_tilda_new):
        """
        Add new state and action to the dictionary and extend sufficient statistics matrices and vectors for one dimension
        and reestimates all parameters apart form the ones involving the reward
        """
        _K_tilda_inv_new = self.extendKtildainv(self.params['_K_tilda_inv'], _a_new, delta_new)
        _a_new = np.zeros(len(self.params['_dictionary']) + 1)
        _a_new[-1] =1.0
        _h_tilda_new = self.extendVector(self.params['_a'])
        _h_tilda_new[-1] = - self._gamma

        if self._prior != None:
            kernelValue = kernel.PriorKernel(state,state)
        else:
            kernelValue = kernel.beliefKernel(state,state)

        delta_k_new = np.dot(self.params['_a'], (k_tilda_prev - 2.0 * self._gamma * k_tilda_new)) \
                + (self._gamma**2)*kernelValue*kernel.ActionKernel(action,action)


        part1 = np.dot(self.params['_C_tilda'], delta_k_tilda_new)
        part2 = np.zeros(len(self.params['_dictionary']))\
                    if self.initial else (((self._gamma * (self._sigma ** 2)) * self.params['_c_tilda']) / self.params['_s'])

        _c_tilda_new = self.extendVector(_h_tilda_new[:-1] - part1  + part2)
        _c_tilda_new[-1] = _h_tilda_new[-1]

        spart1 = (1.0 + (self._gamma ** 2))* (self._sigma **2)
        spart2 = np.dot(delta_k_tilda_new, np.dot(self.params['_C_tilda'], delta_k_tilda_new))
        spart3 = 0.0  if self.initial else  ((2*np.dot(self.params['_c_tilda'], delta_k_tilda_new)\
                        - self._gamma*(self._sigma**2)) * (self._gamma * (self._sigma ** 2 )) / self.params['_s'])

        _s_new = spart1 + delta_k_new - spart2 + spart3
        _alpha_tilda_new=self.extendVector(self.params['_alpha_tilda'])
        _C_tilda_new = self.extendMatrix(self.params['_C_tilda'])

        self.params['_s'] = _s_new
        self.params['_alpha_tilda'] = _alpha_tilda_new
        self.params['_c_tilda'] = _c_tilda_new
        self.params['_C_tilda'] = _C_tilda_new
        self.params['_K_tilda_inv'] = _K_tilda_inv_new
        self.params['_a'] = _a_new
        self.params['_dictionary'].append([state, action])
        #self.checkKtildainv(kernel)

    def NoExtend(self, _a_new, delta_k_tilda_new):
        """
        Resestimates sufficient statistics without extending the dictionary
        """
        _h_tilda_new = self.params['_a'] - self._gamma * _a_new

        part1 = np.zeros(len(self.params['_dictionary'])) \
                    if self.initial else (self.params['_c_tilda'] * (self._gamma * (self._sigma ** 2)) / self.params['_s'])
        part2 = np.dot(self.params['_C_tilda'], delta_k_tilda_new)
        _c_tilda_new = part1  + _h_tilda_new - part2

        spart1 = (1.0 + ( 0.0  if self.terminal else (self._gamma ** 2)))* (self._sigma **2)
        spart2 = np.dot(delta_k_tilda_new, (_c_tilda_new + (np.zeros(len(self.params['_dictionary'])) \
                        if self.initial else (self.params['_c_tilda'] * (self._gamma) * (self._sigma ** 2) / self.params['_s']))))
        spart3 = (0 if self.initial else ((self._gamma**2) * (self._sigma ** 4) / self.params['_s']))

        _s_new = spart1  + spart2 - spart3
        self.params['_c_tilda'] = _c_tilda_new
        self.params['_s'] = _s_new
        self.params['_a'] = _a_new

    #@profile
    def LearningStep(self, pstate, paction, reward, state, action, kernel):
        """
        The main function of the GPSarsa algorithm
        :parameter:
        pstate previous state
        paction previous action
        reward current reward
        state next state
        action next action
        kernel the kernel function

        Computes sufficient statistics needed to estimate the posterior of the mean and the covariance of the Gaussian process

        If the estimate of mean can take into account prior if specified
        """
        
        if self._prior != None:
            if not self.terminal:
                offset = self._prior.QvalueMean(pstate,paction,kernel) \
                        - self._gamma*self._prior.QvalueMean(state,action,kernel)
            else:
                offset = self._prior.QvalueMean(pstate,paction,kernel)

            reward = reward - offset
        # INIT:
        if len(self.params['_dictionary']) == 0:
            self.params['_K_tilda_inv'] = np.zeros((1, 1))
            if self._prior != None:
                self.params['_K_tilda_inv'][0][0] = 1.0 / (kernel.PriorKernel(pstate, pstate) * kernel.ActionKernel(paction, paction))
            else:
                self.params['_K_tilda_inv'][0][0] = 1.0 / (kernel.beliefKernel(pstate, pstate) * kernel.ActionKernel(paction, paction))

            self.params['_dictionary'].append([pstate, paction])

        elif self.initial :
            k_tilda_prev = self.k_tilda(pstate,paction,kernel)
            self.params['_a'] = np.dot(self.params['_K_tilda_inv'], k_tilda_prev)
            self.params['_c_tilda'] = np.zeros(len(self.params['_dictionary']))
            if self._prior != None:
                delta_prev = kernel.PriorKernel(pstate,pstate)*kernel.ActionKernel(paction,paction) - np.dot(k_tilda_prev, self.params['_a'])
            else:
                delta_prev = kernel.beliefKernel(pstate,pstate)*kernel.ActionKernel(paction,paction) - np.dot(k_tilda_prev, self.params['_a'])

            self.params['_d'] = 0.0
            self.params['_s'] = float('inf')

            if delta_prev > self._nu :
                self.Extend(delta_prev, pstate, paction)

        k_tilda_prev = self.k_tilda(pstate,paction,kernel)        


        if self.terminal:
            _a_new = np.zeros(len(self.params['_dictionary']))
            delta_new = 0.0
            delta_k_tilda_new = k_tilda_prev                        
        else:   
            k_tilda_new = self.k_tilda(state, action, kernel)      
            _a_new = np.dot(self.params['_K_tilda_inv'], k_tilda_new)

            if self._prior != None:
                curr_ker = kernel.PriorKernel(state,state)*kernel.ActionKernel(action,action)
            else:
                curr_ker = kernel.beliefKernel(state,state)*kernel.ActionKernel(action,action)

            ker_est = np.dot(k_tilda_new, _a_new)
            delta_new = curr_ker - ker_est
            delta_k_tilda_new = k_tilda_prev - self._gamma*k_tilda_new

        _d_new = reward + (0.0 if self.initial else (self._gamma * (self._sigma ** 2) * self.params['_d']) / self.params['_s']) \
                    - np.dot(delta_k_tilda_new, self.params['_alpha_tilda'])

        self.params['_d'] =_d_new

        #logger.warning("Delta new is "+str(delta_new))
        if delta_new<0 and math.fabs(delta_new)>0.0001:
            #logger.error("Negative sparcification "+str(delta_new))
            # dont crash everything - just dont add this point to dictionary
            logger.warning("Negative sparcification "+str(delta_new))
         
        if delta_new > self._nu :
            self.ExtendNew(delta_new, state, action, kernel, _a_new, k_tilda_prev, k_tilda_new, delta_k_tilda_new)
        else:
            self.NoExtend(_a_new, delta_k_tilda_new)
        
        # If optimising HYPERPARAMETERS -- do it when self.terminal:
        #
        #if self.terminal:
        #    self._optimise_hyperparameters()

        self.params['_alpha_tilda'] += self.params['_c_tilda'] * (self.params['_d'] / self.params['_s'])
        self.params['_C_tilda'] += np.outer(self.params['_c_tilda'], self.params['_c_tilda']) / self.params['_s']

                
    def checkKtildainv(self, kernel):
        """
        Checks positive definiteness
        :param: (instance) 
        """
        if not np.all(np.linalg.eigvals(self.params['_K_tilda_inv']) > 0):
            logger.error("Matrix not positive definite")

        K_tilda = []

        for [state, action] in self.params['_dictionary']:
            K_tilda.append(self.k_tilda(state,action,kernel))

        I = np.dot(np.array(K_tilda), self.params['_K_tilda_inv'])

        print np.array(K_tilda)
        for i in range(len(self.params['_dictionary'])):
            if math.fabs(I[i][i]-1.0)>0.0001:
                print I[i][i]
                logger.error("Inverse not calculated properly")
            for j in range(len(self.params['_dictionary'])):
                if i!=j and math.fabs(I[i][j])>0.0001:
                    print I[i][j]
                    logger.error("Inverse not calculated properly")

    def readPolicy(self):
        """
        Reads dictionary and parameter file
        """
        inputDictFile = self._inputDictFile
        inputParamFile = self._inputParamFile
        if self.sharedParams:
            inputDictFile = self._inputDictFile.replace(self.domainString, 'singlemodel')
            inputParamFile = self._inputParamFile.replace(self.domainString, 'singlemodel')
        if not os.path.isfile(inputDictFile) or not os.path.isfile(inputParamFile):
            if self.learning or self._random:
                logger.warning('inpolicyfile:'+self._inpolicyfile+" does not exist")
                #return
            else:
                #logger.error(self._inpolicyfile+" does not exist and policy is not learning nor set to random")
                logger.warning(self._inpolicyfile+" does not exist and policy is not learning nor set to random")
        else:
            self.readDictionary()
            self.readParameters()

    def readDictionary(self):
        """
        Reads dictionary
        """
        inputDictFile = self._inputDictFile
        if self.sharedParams:
            inputDictFile = self._inputDictFile.replace(self.domainString, 'singlemodel')
        if inputDictFile not in  ["",".dct"]:
            logger.info("Loading dictionary file " + inputDictFile)
            with open(inputDictFile,'rb') as pkl_file:
                self.params['_dictionary'] = pkl.load(pkl_file)
                #logger.info("Read dictionary of size "+str(len(self.sharedParams['_dictionary'])))
                logger.info("in SARSA class: Read dictionary of size " + str(len(self.params['_dictionary'])))
        else:
            logger.warning("Dictionary file not given")

    def readParameters(self):
        """
        Reads parameter file
        """
        inputParamFile = self._inputParamFile
        if self.sharedParams:
            inputParamFile = self._inputParamFile.replace(self.domainString, 'singlemodel')
        with open(inputParamFile,'rb') as pkl_file:
            if self.numpyFileFormat:
                npzfile = np.load(pkl_file)
                try:
                    self.params['_K_tilda_inv'] = npzfile['_K_tilda_inv']
                    self.params['_C_tilda'] = npzfile['_C_tilda']
                    self.params['_c_tilda'] = npzfile['_c_tilda']
                    self.params['_a'] = npzfile['_a']
                    self.params['_alpha_tilda'] = npzfile['_alpha_tilda']
                    self.params['_d'] = npzfile['_d']
                    self.params['_s'] = npzfile['_s']
                except Exception as e:
                    print npzfile.files
                    raise e

            else:
                # ORDER MUST BE THE SAME HERE AS WRITTEN IN saveParameters() below.
                self.params['_K_tilda_inv'] = pkl.load(pkl_file)
                self.params['_C_tilda'] = pkl.load(pkl_file)
                self.params['_c_tilda'] = pkl.load(pkl_file)
                self.params['_a'] = pkl.load(pkl_file)
                self.params['_alpha_tilda'] = pkl.load(pkl_file)
                self.params['_d'] = pkl.load(pkl_file)
                self.params['_s'] = pkl.load(pkl_file)
                #-------------------------------

    def saveDictionary(self):
        """
        Saves dictionary
        :param None:
        :returns None:
        """
        outputDictFile = self._outputDictFile
        if self.sharedParams:
            outputDictFile = self._outputDictFile.replace(self.domainString, 'singlemodel')
        PolicyUtils.checkDirExistsAndMake(outputDictFile)
        with open(outputDictFile,'wb') as pkl_file:
            pkl.dump(self.params['_dictionary'], pkl_file)


    def saveParameters(self):
        """
        Save parameter file
        """
        outputParamFile = self._outputParamFile
        if self.sharedParams:
            outputParamFile = self._outputParamFile.replace(self.domainString, 'singlemodel')
        PolicyUtils.checkDirExistsAndMake(outputParamFile)
        with open(outputParamFile,'wb') as pkl_file:
            if self.numpyFileFormat:
                np.savez(pkl_file, _K_tilda_inv=self.params['_K_tilda_inv'], _C_tilda=self.params['_C_tilda'], _c_tilda=self.params['_c_tilda'], _a=self.params['_a'], _alpha_tilda=self.params['_alpha_tilda'], _d=self.params['_d'], _s=self.params['_s'])
            else:
                # ORDER MUST BE THE SAME HERE AS IN readParameters() above.
                pkl.dump(self.params['_K_tilda_inv'], pkl_file)
                pkl.dump(self.params['_C_tilda'], pkl_file)
                pkl.dump(self.params['_c_tilda'], pkl_file)
                pkl.dump(self.params['_a'], pkl_file)
                pkl.dump(self.params['_alpha_tilda'], pkl_file)
                pkl.dump(self.params['_d'], pkl_file)
                pkl.dump(self.params['_s'], pkl_file)
                #-------------------------------

    def savePrior(self, priordictfile, priorparamfile):
        """
        Saves the current GP as a prior (these are only the parameters needed to estimate the mean)
        """
        PolicyUtils.checkDirExistsAndMake(priordictfile)
        with open(priordictfile, 'wb') as pkl_file:
            pkl.dump(self.params['_dictionary'], pkl_file)
        PolicyUtils.checkDirExistsAndMake(priorparamfile)
        with open(priorparamfile, 'wb') as pkl_file:
            pkl.dump(self.params['_alpha_tilda'], pkl_file)

    def savePolicy(self):
        """Saves the GP dictionary (.dct) and parameters (.prm). Saves as a prior if self.save_as_prior is True.

        :param None:
        :returns: None
        """
        outpolicyfile = self._outpolicyfile
        if self.sharedParams:
            outpolicyfile = self._outpolicyfile.replace(self.domainString,'singlemodel')
        if self.save_as_prior:
            logger.info("saving GP policy: "+outpolicyfile + " as a prior")
            priordictfile = outpolicyfile +"."+ str(self._num_prior)+".prior.dct"
            priorparamfile = outpolicyfile +"."+ str(self._num_prior)+".prior.prm"
            self.savePrior(priordictfile,priorparamfile)
        else:
            logger.info("saving GP policy: "+outpolicyfile)
            self.saveDictionary()
            self.saveParameters()



#END OF FILE
