###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
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
MulitEvaluator.py - wrapper module for calling multiple evaluators
======================================================================================

Copyright CUED Dialogue Systems Group 2016 - 2017

.. seealso:: PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`SuccessEvaluator.ObjectiveSuccessEvaluator` |.|
    import :mod:`SuccessEvaluator.SubjectiveSuccessEvaluator` 
    import :class:`evaluation.EvaluationManager.Evaluator` |.|

************************

'''
__author__ = "cued_dialogue_systems_group"

from EvaluationManager import Evaluator
from SuccessEvaluator import ObjectiveSuccessEvaluator,SubjectiveSuccessEvaluator
from utils import Settings, ContextLogger
import numpy as np
logger = ContextLogger.getLogger('')

class MultiEvaluator(Evaluator):
    '''
    Wrapper class for combining multiple evaluators, e.g. if objective and subjective success needs to be tracked.
    
    Need to set the mainEvaluator in the config file which refers to the index in the list of evaluators which is used to compute the reward.
    
    Optionally may set the compareEvaluator config setting to specify an evaluator to determine if the current dialogue is used for training, e.g., if a setting like subj == obj is used.
    '''
    
    def __init__(self, domainString):
        super(MultiEvaluator, self).__init__(domainString)
        
        self.evaluators = [ObjectiveSuccessEvaluator(domainString), SubjectiveSuccessEvaluator(domainString)]
        
        self.evaluatorList = None
        if Settings.config.has_option('eval', 'multievaluatorlist'):
            self.evaluatorList = Settings.config.get('eval', 'multievaluatorlist').split(',')
        if Settings.config.has_option('eval_'+domainString, 'multievaluatorlist'):
            self.evaluatorList = Settings.config.get('eval_'+domainString, 'multievaluatorlist').split(',')
        if self.evaluatorList is not None:
            self.evaluators = []
            for e in self.evaluatorList:
                e = e.strip()
                self.evaluators.append(self._load_evaluator(e, domainString))
        
        self.evaluator_label = "multi evaluator (" + ", ".join([e.evaluator_label for e in self.evaluators]) + ")"
#         for e in self.evaluators:
#             self.evaluator_label += e.evaluator_label + " "
        
        self.mainEvaluator = 0
        if Settings.config.has_option('eval', 'mainevaluator'):
            self.mainEvaluator = Settings.config.getint('eval', 'mainevaluator')
        if Settings.config.has_option('eval_'+domainString, 'mainevaluator'):
            self.mainEvaluator = Settings.config.getint('eval_'+domainString, 'mainevaluator')
            
        self.compareEvaluator = None
        if Settings.config.has_option('eval', 'compareevaluator'):
            self.compareEvaluator = Settings.config.getint('eval', 'compareevaluator')
        if Settings.config.has_option('eval_'+domainString, 'compareevaluator'):
            self.compareEvaluator = Settings.config.getint('eval_'+domainString, 'compareevaluator')
        
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
        
        return reward[self.mainEvaluator]
        
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
        
        return final_reward[self.mainEvaluator]
    
    def _getTurnReward(self,turnInfo):
        
        rewardVec = [evaluator._getTurnReward(turnInfo) for evaluator in self.evaluators]
                
        return rewardVec
           
    def _getFinalReward(self, finalInfo):
        
        rewardVec = [evaluator._getFinalReward(finalInfo) for evaluator in self.evaluators]
        self.outcome = [evaluator.outcome for evaluator in self.evaluators]

        return rewardVec
    
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
            s += ', rew = {}, turn = {}'.format(self.total_reward[self.mainEvaluator], self.num_turns)
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
                o = [outcome[i] for outcome in self.outcomes if outcome[i] is not None]
                if len(o):
                    self._prstr(1, self.evaluators[i]._getResultString(o))
            self._prstr(1, 'Average reward  = %.2f +- %.2f' % (np.mean([reward[self.mainEvaluator] for reward in self.rewards]), \
                                                            tinv * np.std([reward[self.mainEvaluator] for reward in self.rewards]) / np.sqrt(num_dialogs)))    
            self._prstr(1, 'Average turns   = %.2f +- %.2f' % (np.mean(self.turns), \
                                                            tinv * np.std(self.turns) / np.sqrt(num_dialogs)))
        return
    
    def doTraining(self):
        if self.compareEvaluator is None:
            return True
        
        return self.outcome[self.mainEvaluator] == self.outcome[self.compareEvaluator]
        
    def _load_evaluator(self, configString, domainString):
        '''
        Loads and instantiates the respective evaluator as configured by configString. The new object is returned.
                
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        
        :param configString: the config string pointing to the evaluator class to be instantiated
        :type configString: str
        :param domainString: the domain the evaluator will work on. Default is None.
        :type domainString: str
        
        :returns: The evaluator object
        '''
        

        try:
            # try to view the config string as a complete module path to the class to be instantiated
            components = configString.split('.')
            packageString = '.'.join(components[:-1]) 
            classString = components[-1]
            mod = __import__(packageString, fromlist=[classString])
            klass = getattr(mod, classString)
            return klass(domainString)
        except ImportError as e:
            print e
            logger.error('Unknown domain evaluator "{}" for domain "{}"'.format(configString, domainString))
            
        
    
        