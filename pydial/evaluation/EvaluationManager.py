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
EvaluationManager.py - module for determining the reward
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` 

************************

'''

__author__ = "cued_dialogue_systems_group"
import numpy as np
from ontology import OntologyUtils
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')

class Evaluator(object):
    '''
    Interface class for a single domain evaluation module. Responsible for recording/calculating turns, dialogue outcome, reward for a single 
    dialog. To create your own reward model, derive from this class and depending on your requirements override the methods :func:`_getTurnReward` and :func:`_getFinalReward`.
    '''
    
    def __init__(self, domainString):
        """
        :param domainString: domain tag
        :type domainString: str
        """
        self.domainString = domainString
        
        self.traceDialog = 2
        if Settings.config.has_option("GENERAL", "tracedialog"):
            self.traceDialog = Settings.config.getint("GENERAL", "tracedialog")
            
        # Recording rew,outcome,turn - over all dialogues run during a session - used for printing on hub exit.
        self.rewards = []
        self.preferences = []
        self.outcomes = []
        self.turns = []
        self.finalrewards = []
        
        self.outcome = False
        self.num_turns = 0
        self.total_reward = np.array([0.0, 0.0])  # [turn-penalty, success-reward]
        
        self.evaluator_label = "{}".format(type(self).__name__)
        self.evaluator_short_label = self.evaluator_label[0:3]
        
    def turnReward(self, turnInfo):
        '''
        Computes the turn reward using turnInfo by calling :func:`_getTurnReward`. Updates total reward and number of turns
        
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: np.array -- the turn reward. [turn-penalty, success-reward]
        '''
        reward = self._getTurnReward(turnInfo)
        
        self.total_reward += reward
        self.num_turns += 1
        
        return reward  
    
    def finalReward(self, finalInfo):
        '''
        Computes the final reward using finalInfo by calling :func:`_getFinalReward`. Updates total reward and dialogue outcome
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward.
        '''
        if self.num_turns > 0:
            final_reward = self._getFinalReward(finalInfo)
            self.total_reward += final_reward
                
            self.rewards.append(self.total_reward)
            self.outcomes.append(self.outcome)
            self.turns.append(self.num_turns)
        else:
            final_reward = np.array([0.0, 0.0])

        return final_reward

    def record_preference(self, preference=None):
        self.preferences.append(preference)
    
    def restart(self):
        """
        Reset the domain evaluators internal variables. 
        :param: None
        :returns None:
        """
        self.outcome = False
        self.num_turns = 0
        self.total_reward = np.array([0.0, 0.0])
       
######################################################################
# interface methods
######################################################################

    def _getTurnReward(self, turnInfo):
        '''
        Computes the turn reward using turnInfo. 
        
        Should be overridden by sub-class if values others than 0 should be returned.  
        
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: np.array -- the turn reward, default (0,0).
        '''
        return np.array([0, 0])
        
    def _getFinalReward(self, finalInfo):
        '''
        Computes the final reward using finalInfo and sets the dialogue outcome. 
        
        Should be overridden by sub-class if values others than 0 should be returned.
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: np.array -- the turn reward, default (0,0).
        '''
        return np.array([0, 0])
    
    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes), \
                                                            100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))
    
    def doTraining(self):
        '''
        Defines whether the currently evaluated dialogue should be used for training.
        
        Should be overridden by sub-class if values others than True should be returned.
        
        :return: bool -- whether the dialogue should be used for training
        '''
        return True
    
######################################################################
# print methods
######################################################################    

    def _prstr(self, tracelevel, s, lvl='dial'):
        if self.traceDialog >= tracelevel: print s
        if lvl == 'dial':
            logger.dial(s)
        if lvl == 'results':
            logger.results(s)
        return

    def print_dialog_summary(self):
        """
        Prints a summary of the current dialogue. Assumes dialogue outcome represents success. For other types, override methods in sub-class.
        """
        if self.num_turns > 1:
            self._prstr(2, "Evaluation of domain: {} --evaluated by: {}".format(self.domainString, self.evaluator_label))
            self._prstr(2, 'rew = [%d, %d], suc = %d, turn = %d' % (self.total_reward[0], self.total_reward[1], self.outcome, self.num_turns))
        return

    def print_summary(self):
        """
        Prints the summary of a run - ie multiple dialogs. Assumes dialogue outcome represents success. For other types, override methods in sub-class.
        """
        num_dialogs = len(self.rewards)
        assert(len(self.outcomes) == num_dialogs)
        assert(len(self.turns) == num_dialogs)
        if self.traceDialog > 1: print '-' * 20
        self._prstr(1, "Results for domain: " + self.domainString + ' --evaluated by: ' + self.evaluator_label, 'results')
        
        # computing t-value for confidence interval of 95%
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)

        self._prstr(1, '# of dialogues  = %d' % num_dialogs, 'results')
        if num_dialogs:
            self._prstr(1, 'Average reward  = %.2f +- %.2f' % (np.mean(self.rewards), \
                                                            tinv * np.std(self.rewards) / np.sqrt(num_dialogs)), 'results')
            utilities = np.array(self.rewards) * np.array(self.preferences)
            self._prstr(1, 'Average utility  = %.2f +- %.2f' % (np.mean(utilities), \
                                                               tinv * np.std(utilities) / np.sqrt(num_dialogs)),
                        'results')
            self._prstr(1, self._getResultString(self.outcomes), 'results')
            self._prstr(1, 'Average turns   = %.2f +- %.2f' % (np.mean(self.turns), \
                                                            tinv * np.std(self.turns) / np.sqrt(num_dialogs)), 'results')
        return
    
    

    
class EvaluationManager(object):
    '''
    The evaluation manager manages the evaluators for all domains. It supports two types of reward: a turn-level reward and a dialogue-level reward. 
    The former is accessed using :func:`turnReward` and the latter using :func:`finalReward`.
    You can either use one or both methods for reward computing.
    
    An example where both are used in the traditional reward computation where each turn is penalised with a small negative reward (which is realised with :func:`turnReward`)
    and in the end, the dialogue is rewarded with a big positive reward given the overall dialogue (which is realised with :func:`finalReward`).
    '''
    
    def __init__(self):
        self.domainEvaluators = dict.fromkeys(OntologyUtils.available_domains, None)
        self.final_reward = dict.fromkeys(OntologyUtils.available_domains, None)
        self.SPECIAL_DOMAINS = ['topicmanager', 'wikipedia', 'ood']

    def restart(self):
        '''
        Restarts all domain evaluators.
        '''
        for domain in self.domainEvaluators:
            if self.domainEvaluators[domain] is not None:
                self.domainEvaluators[domain].restart()
            self.final_reward[domain] = None
                
    def turnReward(self, domainString, turnInfo):
        '''
        Computes the turn reward for the given domain using turnInfo by delegating to the domain evaluator.
        
        :param domainString: the domain string unique identifier.
        :type domainString: str
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward for the given domain.
        '''
        # Replaces: for sim: reward_and_success, for texthub/dialoguserver: per_turn_reward, add_DM_history
        # turnInfo: {simulatedUserModel, sys_act} 
        
        if domainString in self.SPECIAL_DOMAINS:
            return 0
        
        if self.domainEvaluators[domainString] is None:
            self._bootup_domain(domainString)
            
        turnInfo['belief'] = turnInfo['state'].getDomainState(domainString)
        
        return self.domainEvaluators[domainString].turnReward(turnInfo)
    
    def finalReward(self, domainString, finalInfo):
        '''
        Computes the final reward for the given domain using finalInfo by delegating to the domain evaluator.
        
        :param domainString: the domain string unique identifier.
        :type domainString: str
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward for the given domain.
        '''
        # Replaces: for all: finalise_dialogue, for dialogue_server/texthub: objective_success_by_task
        # finalInfo: {task, subjectiveFeedback}
        
        if domainString in self.SPECIAL_DOMAINS:
            return 0
        
        self.final_reward[domainString] = self.domainEvaluators[domainString].finalReward(finalInfo)
        return self.final_reward[domainString]
    
    def finalRewards(self, finalInfo=None):
        '''
        Computes the :func:`finalReward` method for all domains where it has not been computed yet.
        
        :param finalInfo: parameters necessary for computing the final rewards, eg., task description or subjective feedback. Default is None
        :type finalInfo: dict
        :returns: dict -- mapping of domain to final rewards 
        '''
        for domain in self.final_reward:
            if self.final_reward[domain] is None and self.domainEvaluators[domain] is not None:
                self.finalReward(domain, finalInfo)
                
        return self.final_reward

    def record_preference(self, preference=None):
        for domain in self.final_reward:
            if self.domainEvaluators[domain] is not None:
                self.domainEvaluators[domain].record_preference(preference)
    
    def doTraining(self):
        training_vec = {}
        for domain in self.domainEvaluators:
            if self.domainEvaluators[domain] is not None:
                training_vec[domain] = self.domainEvaluators[domain].doTraining()
            else:
                training_vec[domain] = True # by default all dialogues are potentially used for training
        return training_vec
    
    def _load_domains_evaluator(self, domainString=None):
        '''
        Loads and instantiates the respective evaluator as configured in config file. The new object is added to the internal
        dictionary. 
        
        Default is 'objective'.
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        
        :param domainString: the domain the evaluator will work on. Default is None.
        :type domainString: str
        :returns: None
        '''
        
        evaluator = 'objective'
        if Settings.config.has_option('eval', 'successmeasure'):
            evaluator = Settings.config.get('eval', 'successmeasure')
        if Settings.config.has_option('eval_' + domainString, 'successmeasure'):
            evaluator = Settings.config.get('eval_' + domainString, 'successmeasure')
        if Settings.config.has_option("usermodel", "simlevel"):
            simlevel = Settings.config.get("usermodel", "simlevel")
            if simlevel == 'sys2text':
                evaluator = 'sys2text'
        
#         if evaluator == "rnn":
#             import RNN_evaluator
#             self.domainEvaluators[domainString] = RNN_evaluator.RNN_evaluator(dstring=domainString)
#         elif
        if evaluator == "objective": 
            import SuccessEvaluator
            self.domainEvaluators[domainString] = SuccessEvaluator.ObjectiveSuccessEvaluator(domainString)
        elif evaluator == "sys2text":
            import SuccessEvaluator
            self.domainEvaluators[domainString] = SuccessEvaluator.Sys2TextSuccessEvaluator()
        elif evaluator == "subjective": 
            import SuccessEvaluator
            self.domainEvaluators[domainString] = SuccessEvaluator.SubjectiveSuccessEvaluator(domainString)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = evaluator.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.domainEvaluators[domainString] = klass(domainString)
            except ImportError:
                logger.error('Unknown domain evaluator "{}" for domain "{}"'.format(evaluator, domainString))
            
    def _bootup_domain(self, dstring):
        '''
        Ensures that the respective domain's evaluator is booted up correctly and resets it.
        
        :param dstring: the domain of which the evaulator should be booted.
        :type dstring: str
        :return: None
        '''
        if dstring in self.SPECIAL_DOMAINS:
            logger.warning("No eval manager a present for {}".format(dstring))
            return
        self._load_domains_evaluator(dstring)
        self.domainEvaluators[dstring].restart()

    def print_dialog_summary(self):
        """
        Prints the history of the just completed dialog.
        """
        for dstring in self.domainEvaluators:
            if self.domainEvaluators[dstring] is not None:
                self.domainEvaluators[dstring].print_dialog_summary()

    def print_summary(self):
        """
        Prints the history over all dialogs run thru simulate.
        """
        for dstring in self.domainEvaluators:
            if self.domainEvaluators[dstring] is not None:
                self.domainEvaluators[dstring].print_summary()

# END OF FILE
