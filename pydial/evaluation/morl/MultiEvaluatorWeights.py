###############################################################################
# CUED PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015-16  Cambridge University Engineering Department 
# Dialogue Systems Group
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
MulitEvaluatorWeights.py - wrapper module for calling multiple evaluators and weighting them
============================================================================================

Copyright CUED Dialogue Systems Group 2016

.. seealso:: CUED PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.Ontology` |.|
    import :class:`evaluation.EvaluationManager.Evaluator`

************************

'''
__author__ = "cued_dialogue_systems_group"

from EvaluationManager import Evaluator
from ObjectiveEvaluators import ObjectiveSuccessEvaluator,ObjectiveTurnEvaluator
from utils import Settings, ContextLogger
import numpy as np
logger = ContextLogger.getLogger('')

class MultiEvaluator(Evaluator):
    
    def __init__(self, domainString):
        super(MultiEvaluator, self).__init__(domainString)
        
        self.evaluators = [ObjectiveSuccessEvaluator(domainString),ObjectiveTurnEvaluator(domainString)]
        
        self.evaluator_label = "multi evaluator ("
        for e in self.evaluators:
            self.evaluator_label += e.evaluator_label + " "
        self.evaluator_label += ")"
        
        self.mainEvaluator = 0
        if Settings.config.has_option('eval_'+domainString, 'mainEvaluator'):
            self.mainEvaluator = Settings.config.getint('eval_'+domainString, 'mainEvaluator')
            
        self.compareEvaluator = None
        if Settings.config.has_option('eval_'+domainString, 'compareEvaluator'):
            self.compareEvaluator = Settings.config.getint('eval_'+domainString, 'compareEvaluator')
            
        self.weights = None
        self._loadWeights()
        
        
    def turnReward(self, turnInfo):
        '''
        Computes the turn reward using turnInfo by calling :func:`_getTurnReward`. Updates total reward and number of turns
        
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''
        reward = self._getTurnReward(turnInfo)
        
        for i in range(0,len(reward)):
            self.total_reward[i] += reward[i]
        self.num_turns += 1
        
        if self.weights is None:
            return reward[self.mainEvaluator]
        else:
            rew = 0.0
            for i in range(len(self.weights)):
                rew += len(self.weights) * self.weights[i] * reward[i]
            return rew
        
    def finalReward(self, finalInfo):
        '''
        Computes the final reward using finalInfo by calling :func:`_getFinalReward`. Updates total reward and dialogue outcome
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward.
        '''
        if self.num_turns > 0:
            final_reward = self._getFinalReward(finalInfo)
            
            for i in range(0,len(final_reward)):
                self.total_reward[i] += final_reward[i]
                
            self.rewards.append(self.total_reward)
            self.outcomes.append(self.outcome)
            self.turns.append(self.num_turns)
        else:
            final_reward = [0] * len(self.evaluators)
        
        
        self.fRew = self._getWeightedReward(final_reward)
        
         
        self.finalrewards.append(self.fRew)
        return self.fRew

    
    def _getTurnReward(self,turnInfo):
        
        rewardVec = [evaluator._getTurnReward(turnInfo) for evaluator in self.evaluators]
                
        return rewardVec
           
    def _getFinalReward(self, finalInfo):
        
        rewardVec = [evaluator._getFinalReward(finalInfo) for evaluator in self.evaluators]
        self.outcome = [evaluator.outcome for evaluator in self.evaluators]

        return rewardVec
    
    def _getWeightedReward(self, final_reward):
        if self.weights is None:
            return final_reward[self.mainEvaluator]
        else:
            rew = 0.0
            for i in range(len(self.weights)):
                rew += len(self.weights) * self.weights[i] * final_reward[i]
            return rew
        
    
    def restart(self):
        """
        Initialise variables and resets internal state.
    
        :param: None
        :returns: None

        """
        super(MultiEvaluator, self).restart()
        
        self.total_reward = [0 for _ in self.evaluators] 
        
        for evaluator in self.evaluators:
            evaluator.restart()
            
        self.fRew = None
        self.weights = None
        self._loadWeights()
        
    def print_dialog_summary(self):
        """
        Prints a summary of the current dialogue. Assumes dialogue outcome represents success. For other types, override methods in sub-class.
        """
        if self.num_turns > 1:
            self._prstr(2, "Evaluation of domain: {} --evaluated by: {}".format(self.domainString, self.evaluator_label))
            l = []
            for i in range(0,len(self.evaluators)):
                l.append(self.evaluators[i].evaluator_short_label + '{} = {}'.format('*' if self.mainEvaluator == i else '', int(self.outcome[i])))
            s = ', '.join(l)
            s += ', rew = {}, turn = {}'.format(self.fRew, self.num_turns)
            self._prstr(2, s)
        return

    def print_summary(self):
        """
        Prints the summary of a run - ie multiple dialogs. Assumes dialogue outcome represents success. For other types, override methods in sub-class.
        """
        num_dialogs = len(self.rewards)
        assert(len(self.outcomes) == num_dialogs)
        assert(len(self.turns) == num_dialogs)
        if self.traceDialog > 1: print '-' * 20
        self._prstr(1, "Results for domain: " + self.domainString + ' --evaluated by: ' + self.evaluator_label)
        
        # computing t-value for confidence interval of 95%
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)

        self._prstr(1, '# of dialogues  = %d' % num_dialogs)
        if num_dialogs:
            for i in range(0,len(self.evaluators)):
                o = [outcome[0] for outcome in self.outcomes if outcome[0] is not None]
                if len(o):
                    self._prstr(1, self.evaluators[i]._getResultString().format(np.mean(o) * 100,
                                                            tinv * 100 * np.std(o) / np.sqrt(num_dialogs)))
            self._prstr(1, 'Average reward  = %.2f +- %.2f' % (np.mean([reward for reward in self.finalrewards]), \
                                                            tinv * np.std([reward for reward in self.finalrewards]) / np.sqrt(num_dialogs)))    
            self._prstr(1, 'Average turns   = %.2f +- %.2f' % (np.mean(self.turns), \
                                                            tinv * np.std(self.turns) / np.sqrt(num_dialogs)))
        return
    
    def doTraining(self):
        if self.compareEvaluator is None:
            return True
        
        return self.outcome[self.mainEvaluator] == self.outcome[self.compareEvaluator]
    
    def _loadWeights(self):
        if self.weights is None:
            if Settings.config.has_option("mogp_"+self.domainString,"weights"):
                self.weights = Settings.config.get("mogp_"+self.domainString,"weights").split()
                self.weights = map(float, self.weights)
                logger.info("MOGP weights read from config: {}".format(self.weights))
        return self.weights
        
        