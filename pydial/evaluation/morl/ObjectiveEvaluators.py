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
ObjectiveEvaluators.py - module for determining objective success and turn length
=================================================================================

Copyright CUED Dialogue Systems Group 2016

.. seealso:: CUED PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.Ontology` |.|
    import :class:`evaluation.EvaluationManager.Evaluator`

************************

'''
__author__ = "cued_dialogue_systems_group"

from evaluation.EvaluationManager import Evaluator
from utils import Settings, ContextLogger, DiaAct
from ontology import Ontology
logger = ContextLogger.getLogger('')

class ObjectiveSuccessEvaluator(Evaluator):
    '''
    This class provides a reward model based on objective success. For simulated dialogues, the goal of the user simulator is compared with the the information the system has provided. 
    For dialogues with a task file, the task is compared to the information the system has provided. 
    '''
    
    def __init__(self, domainString):
        super(ObjectiveSuccessEvaluator, self).__init__(domainString)
        
        # only for nice prints
        self.evaluator_label = "objective success evaluator"
        self.evaluator_short_label = "suc"
               
        # DEFAULTS:
        self.reward_venue_recommended = 0  # we dont use this. 100
        self.penalise_all_turns = True   # We give -1 each turn. Note that this is done thru this boolean
        self.wrong_venue_penalty = 0   # we dont use this. 4
        self.not_mentioned_value_penalty = 0  # we dont use this. 4
        self.successReward = 20
        self.using_tasks = False
        
        # CONFIG:
        if Settings.config.has_option('eval', 'rewardvenuerecommended'):
            self.reward_venue_recommended = Settings.config.getint('eval', 'rewardvenuerecommended')
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option('eval', 'wrongvenuepenalty'):
            self.wrong_venue_penalty = Settings.config.getint('eval', 'wrongvenuepenalty')
        if Settings.config.has_option('eval', 'notmentionedvaluepenalty'):
            self.not_mentioned_value_penalty = Settings.config.getint('eval', 'notmentionedvaluepenalty')
        if Settings.config.has_option("eval_"+domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_"+domainString, "successreward")
        if Settings.config.has_option("dialogueserver","tasksfile"):
            self.using_tasks = True     # will record DM actions to deduce objective success against a given task:
            
        self.venue_recommended = False
            
        self.DM_history = None
        
    def restart(self):
        """
        Initialise variables (i.e. start dialog with: success=False, venue recommended=False, and 'dontcare' as \
        the only mentioned value in each slot)  
    
        :param: None
        :returns: None

        """
        super(ObjectiveSuccessEvaluator, self).restart()
        self.venue_recommended = False
            
        if self.using_tasks:
            self.DM_history = []
        
    def _getTurnReward(self,turnInfo):
        '''
        Computes the turn reward regarding turnInfo. The default turn reward is -1 unless otherwise computed. 
        
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''
        
        if turnInfo is not None and isinstance(turnInfo, dict):
            if 'usermodel' in turnInfo and 'sys_act' in turnInfo:
                um = turnInfo['usermodel']
                
                # unpack input user model um.
                prev_consts = um.prev_goal.constraints 
                sys_act = DiaAct.DiaAct(turnInfo['sys_act'])
                
                # Check if the most recent venue satisfies constraints.
                name = sys_act.get_value('name', negate=False)
                if name not in ['none', None]:
                    # Venue is recommended.
                    possible_entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints=prev_consts)
                    match = name in [e['name'] for e in possible_entities]
                    if match:
                        # Success except if the next user action is reqalts.
                        logger.debug('Correct venue is recommended.')
                        self.venue_recommended = True   # Correct venue is recommended.
                    else:
                        # Previous venue did not match.
                        logger.debug('Venue is not correct.')
                        self.venue_recommended = False

            if 'sys_act' in turnInfo and self.using_tasks:
                self.DM_history.append(turnInfo['sys_act'])
                
        return 0
        
    def _getFinalReward(self,finalInfo):
        '''
        Computes the final reward using finalInfo. Should be overridden by sub-class if values others than 0 should be returned.
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward, default 0.
        '''
        if finalInfo is not None and isinstance(finalInfo, dict):
            if 'usermodel' in finalInfo: # from user simulator
                um = finalInfo['usermodel']
                if um is None:
                    self.outcome = False
                elif self.domainString not in um:
                    self.outcome = False
                else:
                    requests = um[self.domainString].goal.requests
                    if self.venue_recommended and None not in requests.values():
                        self.outcome = True
            elif 'task' in finalInfo: # dialogue server with tasks
                task = finalInfo['task']
                if self.DM_history is not None:
                    informs = self._get_informs_against_each_entity()
                    if informs is not None:
                        for ent in informs.keys():
                            if task is None:
                                self.outcome = True   # since there are no goals, lets go with this ... 
                            elif self.domainString not in task:
                                logger.warning("This task doesn't contain the domain: %s" % self.domainString)
                                logger.debug("task was: " + str(task))  # note the way tasks currently are, we dont have 
                                # the task_id at this point ...
                                self.outcome = True   # This is arbitary, since there are no goals ... lets say true?
                            elif ent in str(task[self.domainString]["Ents"]):
                                # compare what was informed() against what was required by task:
                                required = str(task[self.domainString]["Reqs"]).split(",")
                                self.outcome = True
                                for req in required:
                                    if req == 'name':
                                        continue
                                    if req not in ','.join(informs[ent]): 
                                        self.outcome = False

        return self.outcome * self.successReward
    
    def _get_informs_against_each_entity(self):
        if len(self.DM_history) == 0:
            return None
        informs = {}
        currentEnt = None
        for act in self.DM_history:
            if 'inform(' in act:
                details = act.split("(")[1].split(",")
                details[-1] = details[-1][0:-1]  # remove the closing )
                if not len(details):
                    continue
                if "name=" in act:
                    for detail in details:
                        if "name=" in detail:
                            currentEnt = detail.split("=")[1].strip('"')
                            details.remove(detail)
                            break  # assumes only 1 name= in act -- seems solid assumption
                    
                    if currentEnt in informs.keys():
                        informs[currentEnt] += details
                    else:
                        informs[currentEnt] = details
                elif currentEnt is None:
                    logger.warning("Shouldn't be possible to first encounter an inform() act without a name in it")
                else:
                    logger.warning('assuming inform() that does not mention a name refers to last entity mentioned')
                    informs[currentEnt] += details
        return informs

    def _getResultString(self):
        return 'Average success = {0:0.2f} +- {1:0.2f}'
                
class ObjectiveTurnEvaluator(Evaluator):
    '''
    This class implements a reward model based on the turn length.
    '''
    
    def __init__(self, domainString):
        super(ObjectiveTurnEvaluator, self).__init__(domainString)
        
        # only for nice prints
        self.evaluator_label = "turn evaluator"
        self.evaluator_short_label = "tur"
               
        # DEFAULTS:
        self.penalise_all_turns = True   # We give -1 each turn. Note that this is done thru this boolean
        
        # CONFIG:
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option("eval_"+domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_"+domainString, "successreward")
            
        self.internalReward = 0

        
    def restart(self):
        """
        Calls restart of parent.
    
        :param: None
        :returns: None
        """
        super(ObjectiveTurnEvaluator, self).restart()
        
        self.internalReward = 0
        
    def _getTurnReward(self,turnInfo):
        '''
        Computes the turn reward which is always -1 if activated. 
        
        :param turnInfo: NOT USED parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: int -- the turn reward.
        '''
        
        # Immediate reward for each turn.
        self.internalReward -= self.penalise_all_turns
        return 0
        
    def _getFinalReward(self,finalInfo):
        '''
        Computes the final reward using finalInfo's field "subjectiveSuccess".
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: int -- the final reward, default 0.
        '''
        
        self.outcome = self.internalReward * -1
        return self.internalReward
    
    def _getResultString(self):
        return 'Average turns = {0:0.2f} +- {1:0.2f}'
    
#END OF FILE
