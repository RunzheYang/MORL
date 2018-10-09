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
from utils.dact import DactItem

'''
SuccessEvaluator.py - module for determining objective and subjective dialogue success 
======================================================================================

Copyright CUED Dialogue Systems Group 2016

.. seealso:: PyDial Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`ontology.Ontology` |.|
    import :class:`evaluation.EvaluationManager.Evaluator` |.|

************************

'''
__author__ = "cued_dialogue_systems_group"

from EvaluationManager import Evaluator
from utils import Settings, ContextLogger, DiaAct, dact
from ontology import Ontology
import numpy as np
import copy
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
        self.failPenalty = 0
        self.user_goal = None
        
        # CONFIG:
        if Settings.config.has_option('eval', 'rewardvenuerecommended'):
            self.reward_venue_recommended = Settings.config.getint('eval', 'rewardvenuerecommended')
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option('eval', 'wrongvenuepenalty'):
            self.wrong_venue_penalty = Settings.config.getint('eval', 'wrongvenuepenalty')
        if Settings.config.has_option('eval', 'notmentionedvaluepenalty'):
            self.not_mentioned_value_penalty = Settings.config.getint('eval', 'notmentionedvaluepenalty')
        if Settings.config.has_option("eval", "failpenalty"):
            self.failPenalty = Settings.config.getint("eval", "failpenalty")
        if Settings.config.has_option("eval", "successreward"):
            self.successReward = Settings.config.getint("eval", "successreward")
        if Settings.config.has_option("eval_"+domainString, "failpenalty"):
            self.failPenalty = Settings.config.getint("eval_"+domainString, "failpenalty")
        if Settings.config.has_option("eval_"+domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_"+domainString, "successreward")
            
            
        if Settings.config.has_option("dialogueserver","tasksfile"):
            self.using_tasks = True     # will record DM actions to deduce objective success against a given task:
            
        self.venue_recommended = False
        self.mentioned_values = {}      # {slot: set(values), ...}
        sys_reqestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for slot in sys_reqestable_slots:
            self.mentioned_values[slot] = set(['dontcare'])
            
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
        self.last_venue_recomended = None
        self.mentioned_values = {}      # {slot: set(values), ...}
        sys_reqestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for slot in sys_reqestable_slots:
            self.mentioned_values[slot] = set(['dontcare'])  
            
        if self.using_tasks:
            self.DM_history = []
        
    def _getTurnReward(self,turnInfo):
        '''
        Computes the turn reward regarding turnInfo. The default turn reward is -1 unless otherwise computed. 
        
        :param turnInfo: parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: np.array -- the turn reward.
        '''
        
        # Immediate reward for each turn.
        reward = np.array([-self.penalise_all_turns, 0.0])

        if turnInfo is not None and isinstance(turnInfo, dict):
            if 'usermodel' in turnInfo and 'sys_act' in turnInfo:
                um = turnInfo['usermodel']
                self.user_goal = um.goal.constraints
                
                # unpack input user model um.
                #prev_consts = um.prev_goal.constraints
                prev_consts = copy.deepcopy(um.goal.constraints)
                for item in prev_consts:
                    if item.slot == 'name' and item.op == '=':
                        item.val = 'dontcare'
                requests = um.goal.requests
                sys_act = DiaAct.DiaAct(turnInfo['sys_act'])
                user_act = um.lastUserAct
                
                # Check if the most recent venue satisfies constraints.
                name = sys_act.get_value('name', negate=False)
                lvr = self.last_venue_recomended if hasattr(self, 'last_venue_recomended') else 'not existing'
                if name not in ['none', None]:
                    # Venue is recommended.
                    #possible_entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints=prev_consts)
                    #is_valid_venue = name in [e['name'] for e in possible_entities]
                    self.last_venue_recomended = name
                    is_valid_venue = self._isValidVenue(name, prev_consts)
                    if is_valid_venue:
                        # Success except if the next user action is reqalts.
                        if user_act.act != 'reqalts':
                            logger.debug('Correct venue is recommended.')
                            self.venue_recommended = True   # Correct venue is recommended.
                        else:
                            logger.debug('Correct venue is recommended but the user has changed his mind.')
                    else:
                        # Previous venue did not match.
                        logger.debug('Venue is not correct.')
                        self.venue_recommended = False
                        logger.debug('Goal constraints: {}'.format(prev_consts))
                        reward -= self.wrong_venue_penalty
        
                # If system inform(name=none) but it was not right decision based on wrong values.
                if name == 'none' and sys_act.has_conflicting_value(prev_consts):
                    reward -= np.array([self.wrong_venue_penalty, 0.0])
        
                # Check if the system used slot values previously not mentioned for 'select' and 'confirm'.
                not_mentioned = False
                if sys_act.act in ['select', 'confirm']:
                    for slot in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                        values = set(sys_act.get_values(slot))
                        if len(values - self.mentioned_values[slot]) > 0:
                            # System used values which are not previously mentioned.
                            not_mentioned = True
                            break
        
                if not_mentioned:
                    reward -= np.array([self.not_mentioned_value_penalty, 0.0])
                    
                # If the correct venue has been recommended and all requested slots are filled,
                # check if this dialogue is successful.
                if self.venue_recommended and None not in requests.values():
                    reward += np.array([self.reward_venue_recommended, 0.0])
                    
                # Update mentioned values.
                self._update_mentioned_value(sys_act)
                self._update_mentioned_value(user_act)
            if 'sys_act' in turnInfo and self.using_tasks:
                self.DM_history.append(turnInfo['sys_act'])
                
        return reward

    def _isValidVenue(self, name, constraints):    
        constraints2 = None
        if isinstance(constraints, list):
            constraints2 = copy.deepcopy(constraints)
            for const in constraints2:
                if const.slot == 'name':
                    if const.op == '!=':
                        if name == const.val and const.val != 'dontcare':
                            return False
                        else:
                            constraints2.remove(const)
                    elif const.op == '=':
                        if name != const.val and const.val != 'dontcare':
                            return False
            constraints2.append(DactItem('name','=',name))
        elif isinstance(constraints, dict): # should never be the case, um has DActItems as constraints
            constraints2 = copy.deepcopy(constraints)
            for slot in constraints:
                if slot == 'name' and name != constraints[slot]:
                    return False
            constraints2['name'] = name
        entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints2)

#         is_valid_list = []
#         for ent in entities:
#             is_valid = True
#             for const in constraints:
#                 if const.op == '=':
#                     if const.val != ent[const.slot] and const.val != 'dontcare':
#                         is_valid = False
#                 elif const.op == '!=':
#                     if const.val == ent[const.slot]:
#                         is_valid = False
#             is_valid_list.append(is_valid)
        
        return any(entities)

    def _getFinalReward(self,finalInfo):
        '''
        Computes the final reward using finalInfo. Should be overridden by sub-class if values others than 0 should be returned.
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: np.array -- the final reward, default (0, 0).
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
                    '''if self.last_venue_recomended is None:
                        logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                    else:
                        if self.venue_recommended and None not in requests.values():
                            self.outcome = True
                            logger.dial('Success! User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))
                        else:
                            logger.dial('Fail :( User requests: {}, Venue recomended: {}'.format(requests, self.venue_recommended))'''
                    if None not in requests.values():
                        valid_venue = self._isValidVenue(requests['name'], self.user_goal)
                        if valid_venue:
                            self.outcome = True
                            logger.dial(
                                'Success! User requests: {}'.format(requests))
                        else:
                            logger.dial(
                                'Fail :( User requests: {}'.format(requests))
                    else:
                        logger.dial(
                            'Fail :( User requests: {}'.format(requests))
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
                                    if req.strip(" ") == 'name':
                                        continue
                                    if req.strip(" ") not in ','.join(informs[ent]): 
                                        self.outcome = False

        return np.array([0.0, self.outcome * self.successReward - (not self.outcome) * self.failPenalty])
    
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

    
    
    def _update_mentioned_value(self, act):
        # internal, called by :func:`RewardComputer.get_reward` for both sys and user acts to update values mentioned in dialog
        #
        # :param act: sys or user dialog act
        # :type act: :class:`DiaAct.DiaAct`
        # :return: None
        
        sys_requestable_slots = Ontology.global_ontology.get_system_requestable_slots(self.domainString)
        for item in act.items:
            if item.slot in sys_requestable_slots and item.val not in [None, '**NONE**', 'none']:
                self.mentioned_values[item.slot].add(item.val)
                
                
    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes), \
                                                            100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))
                
class Sys2TextSuccessEvaluator(Evaluator):
    def __init__(self):
        self.goal = {}
        self.rewards = []
        self.traceDialog = 0
        domainString = 'CamRestaurants'
        self.evaluator_label = 0
        self.total_reward = 0
        self.outcome = False
        self.num_turns = 0
        super(Sys2TextSuccessEvaluator, self).__init__(domainString)

    def restart(self):
        self.outcome = False
        self.num_turns = 0
        self.total_reward = 0

    def _getTurnReward(self, turnInfo):
        '''
        Computes the turn reward which is always -1 if activated.

        :param turnInfo: NOT USED parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: np.array -- the turn reward.
        '''

        # Immediate reward for each turn.
        reward = np.array([-1, 0]) # should be - penalise all turns
        return reward

    def _getFinalReward(self, finalInfo):
        requests_fullfilled = finalInfo['usermodel'][self.domainString].reqFullfilled()
        accepted_venue = finalInfo['usermodel']['CamRestaurants'].getCurrentVenue()
        constraints = finalInfo['usermodel']['CamRestaurants'].getConstraintDict()
        constraints['name'] = accepted_venue
        entities = Ontology.global_ontology.entity_by_features(self.domainString, constraints)
        '''try:
            venue_constrains = Ontology.global_ontology.entity_by_features(self.domainString, {'name': accepted_venue})[0]
            print len(entities), requests_fullfilled, accepted_venue, constraints, venue_constrains['food'], venue_constrains['area'], venue_constrains['pricerange']
        except:
            print 'nothing', requests_fullfilled, accepted_venue, constraints
            pass'''
        if len(entities) > 0 and requests_fullfilled:
            self.outcome = True
        if self.outcome:
            return np.array([0.0, 20.0])
        else:
            return np.array([0.0, 0.0])


class SubjectiveSuccessEvaluator(Evaluator):
    '''
    This class implements a reward model based on subjective success which is only possible during voice interaction through the :mod:`DialogueServer`. The subjective feedback is collected and
    passed on to this class.
    '''
    
    def __init__(self, domainString):
        super(SubjectiveSuccessEvaluator, self).__init__(domainString)
        
        # only for nice prints
        self.evaluator_label = "subjective success evaluator"
               
        # DEFAULTS:
        self.penalise_all_turns = True   # We give -1 each turn. Note that this is done thru this boolean
        self.successReward = 20
        
        # CONFIG:
        if Settings.config.has_option('eval', 'penaliseallturns'):
            self.penalise_all_turns = Settings.config.getboolean('eval', 'penaliseallturns')
        if Settings.config.has_option("eval", "successreward"):
            self.successReward = Settings.config.getint("eval", "successreward")
        if Settings.config.has_option("eval_" + domainString, "successreward"):
            self.successReward = Settings.config.getint("eval_" + domainString, "successreward")

        
    def restart(self):
        """
        Calls restart of parent.
    
        :param: None
        :returns: None
        """
        super(SubjectiveSuccessEvaluator, self).restart()
        
    def _getTurnReward(self,turnInfo):
        '''
        Computes the turn reward which is always -1 if activated. 
        
        :param turnInfo: NOT USED parameters necessary for computing the turn reward, eg., system act or model of the simulated user.
        :type turnInfo: dict
        :return: np.array -- the turn reward.
        '''
        
        # Immediate reward for each turn.
        return np.array([-self.penalise_all_turns, 0.0])
        
    def _getFinalReward(self,finalInfo):
        '''
        Computes the final reward using finalInfo's field "subjectiveSuccess".
        
        :param finalInfo: parameters necessary for computing the final reward, eg., task description or subjective feedback.
        :type finalInfo: dict
        :return: np.array -- the final reward, default (0,0).
        '''
        if finalInfo is not None and isinstance(finalInfo, dict):
            if 'subjectiveSuccess' in finalInfo:
                self.outcome = finalInfo['subjectiveSuccess']
                
        if self.outcome is None:
            self.outcome = 0

        return np.array([0.0, self.outcome * self.successReward])
    
    def _getResultString(self, outcomes):
        num_dialogs = len(outcomes)
        from scipy import stats
        if num_dialogs < 2:
            tinv = 1
        else:
            tinv = stats.t.ppf(1 - 0.025, num_dialogs - 1)
        return 'Average subj success = {0:0.2f} +- {1:0.2f}'.format(100 * np.mean(outcomes), \
                                                            100 * tinv * np.std(outcomes) / np.sqrt(num_dialogs))
    
#END OF FILE
