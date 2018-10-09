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
PolicyCommittee.py - implementation of the Bayesian committee machine for dialogue management
=============================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.DiaAct` |.|
     

************************

'''

__author__ = "cued_dialogue_systems_group"
import numpy as np
from utils import Settings, ContextLogger,DiaAct
logger = ContextLogger.getLogger('')

class CommitteeMember(object):
    '''
    Base class defining the interface methods which are needed in addition to the basic functionality provided by :class:`~policy.Policy.Policy`
    
    Committee members should derive from this class.
    '''
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Converts the unabstracted domain state into an abstracted belief state to be used with :func:`~getMeanVar_for_executable_actions`.
        
        :param beliefstate: the unabstracted belief state 
        :type beliefstate: dict
        '''
        pass
    
    def get_Action(self, action):     
        '''
        Converts the unabstracted domain action into an abstracted action to be used for multiagent learning.
        
        :param action: the last system action
        :type action: str
        '''   
        pass
    
    def getMeanVar_for_executable_actions(self, belief, abstracted_currentstate, nonExecutableActions):
        '''
        Computes the mean and variance of the Q value based on the abstracted belief state for each executable action.
        
        :param belief: the unabstracted current domain belief
        :type belief: dict
        :param abstracted_currentstate: the abstracted current belief
        :type abstracted_currentstate: :class:`~policy.Policy.State` or subclass
        :param nonExecutableActions: actions which are not selected for execution based on heuristic
        :type nonExecutableActions: list
        '''
        pass
    
    def getPriorVar(self, belief, act):
        '''
        Returns prior variance for a given belief and action
        
        :param belief: the unabstracted current domain belief state
        :type belief: dict
        :param act: the unabstracted action
        :type act: str
        '''
        pass
    
    def abstract_actions(self, actions):
        '''
        Converts a list of domain acts to their abstract form
        
        :param actions: the actions to be abstracted
        :type actions: list of actions
        '''
        pass
    
    def unabstract_action(self, actions):
        '''
        Converts a list of abstract acts to their domain form
        
        :param actions: the actions to be unabstracted
        :type actions: list of actions
        '''
        pass

class PolicyCommittee(object):
    '''
    Manages everything related to policy committee. All policy members must inherit from :class:`~policy.Policy.Policy` and :class:`~CommitteeMember`.
    '''
    
#     @staticmethod
#     def createCommittee(policyManager):
#         committee = dict.fromkeys(OntologyUtils.available_domains, None)
#         useBCM = False
#         
#         if Settings.config.has_option("policycommittee","bcm"):
#             useBCM = Settings.config.getboolean("policycommittee","bcm")
#         
#         if not useBCM:    
#             return committee # return an empty committee dict to inidcate that committees are not used
#          
#         if Settings.config.has_option("policycommittee","pctype"):
#             pcCreator_type =  Settings.config.get("policycommittee","pctype")
#         if pcCreator_type == 'hdc':
#             self.pcCreator = HDC_PolicyCommittee()
#         elif pcCreator_type == 'configset':
#             self.pcCreator = ConfigSetCommittee()
#         else:
#             logger.error("No such policy committee creator as %s" % pcCreator_type)            
#         
#         if Settings.config.has_option("policycommittee","learningmethod"):
#             self.learning_method = Settings.config.get("policycommittee","learningmethod")
#     
    
    
    def __init__(self, policyManager, committeeMembers, learningmethod):
        self.manager = policyManager
        self.members = committeeMembers
        self.learning_method = learningmethod
    
    # hyps are not used as they are currently only needed by HDC policy which is not compatible with committee
    def act_on(self, domainInControl, state):
        '''
        Provides the next system action based on the domain in control and the belief state.
        
        The belief state is mapped to an abstract representation which is used for all committee members.
        
        :param domainInControl: the domain unique identifier string of the domain in control
        :type domainInControl: str
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :returns: the next system action
        '''
        # 1. check if domains are booted
        
        # 2. calculate domain Qs etc from each committee member
        
        # 3. calculate committee Q and select act
        
        belief = state.getDomainState(domainInControl)
        
        domainPolicies = self.manager.domainPolicies
        
        if not isinstance(domainPolicies[domainInControl], CommitteeMember):
            logger.error("Committee member is not of type CommitteeMember: {}".format(domainInControl))
        
        # 1. Abstract the belief state (blief) based on domainInControl:          
        abstracted_state = domainPolicies[domainInControl].get_State(belief)         
        
        # 2. Determine the domains non-executable actions and then abstract them:
        nonExecutableActions = domainPolicies[domainInControl].actions.getNonExecutable(belief,
                                                        domainPolicies[domainInControl].lastSystemAction)
        nonExecutableActions = domainPolicies[domainInControl].abstract_actions(nonExecutableActions)
        
        
        # 3. Get the statistics needed for BCM decision:
        # 3.1 - Q(b,*) for all abstract executable actions for each committee member
        domainQs = {}
        just_return_bye = False
        for dstring in self.members:                    
            if domainPolicies[dstring] is None:
                self.manager.bootup(dstring)
            if isinstance(domainPolicies[dstring], CommitteeMember): 
                #method exists as type must be GPPolicy or derived
                padded_state = domainPolicies[dstring].get_State(abstracted_state)
                domain_QmeanVar = domainPolicies[dstring].getMeanVar_for_executable_actions(belief, 
                                                                                            padded_state,
                                                                                            nonExecutableActions)
                if not isinstance(domain_QmeanVar,dict):
                    # have returned the bye action -- 
                    just_return_bye = True
                    summaryAct = domain_QmeanVar 
                    break
                else:
                    domainQs[dstring] = domain_QmeanVar
            else:
                logger.warning('Skipping policy committee member %s as policy is not a GP' % dstring)
        
        if not just_return_bye:
            # 3.2 - get domain priors over acts:
            priors = {}
            for abs_act in domainQs[domainInControl]:
                priors[abs_act] = domainPolicies[domainInControl].getPriorVar(belief, abs_act)
                            
            # 4. form BCM decision
            # ----- get BCM abstract summary action
            abstractAct = self._bayes_committee_calculator(domainQs, 
                                                                         priors, 
                                                                         domainInControl,
                                                                         domainPolicies[domainInControl].learner._scale)
            logger.info('BCM: abstract action: %s' % abstractAct)
            # 5. Convert abstract action back to real action:
            summaryAct = domainPolicies[domainInControl].unabstract_action(abstractAct)
        logger.info('BCM: summary action: %s' % summaryAct)
        
        
        # 6. Finally convert to master action:
        systemAct = domainPolicies[domainInControl].actions.Convert(belief,summaryAct,
                                                                       domainPolicies[domainInControl].lastSystemAction)        
        
        
        # 7. Bookkeeping:
        domainPolicies[domainInControl].lastSystemAction = systemAct
        domainPolicies[domainInControl].summaryAct = summaryAct  # TODO -check- not sure this is correct
        domainPolicies[domainInControl].prevbelief = belief
        domainPolicies[domainInControl].actToBeRecorded = abstractAct
        
        # 8. Multiagent learning
        if self.learning_method == "multiagent":
            abstract_GPAction = domainPolicies[domainInControl].get_Action(summaryAct)
            self.domainInControl = domainInControl
            self.multiagent_abstract_state = abstracted_state
            self.multiagent_abstract_action = abstract_GPAction  
            
        _systemAct = DiaAct.DiaAct(systemAct)     
        
        return _systemAct

    
    def record(self, reward, domainInControl):
        '''
        record for committee members. in case of multiagent learning, use information held in committee 
        along with the reward to record (b,a) + r
        
        :param reward: the turn reward to be recorded
        :type reward: np.array
        :param reward: the domain the reward was achieved in
        :type reward: str
        :returns: None
        '''
        if self.learning_method == "multiagent":
                for dstring in self.multiagent_weights:
                    try:
                        padded_state = self.manager.domainPolicies[dstring].get_State(self.multiagent_abstract_state)
                        self.manager.domainPolicies[dstring].record(reward=reward,
                                                                        domainInControl=domainInControl,
                                                                        state=padded_state,
                                                                        action=self.multiagent_abstract_action)
                    except KeyError:
                        logger.error("Mismatch between ontology and slot abstractions \
                        for domain {}\nself.manager.domainPolicies {}\nmembers {}".format(dstring,self.manager.domainPolicies,self.members))
                    
        elif self.learning_method == "singleagent":
            self.manager.domainPolicies[domainInControl].record(reward)
        return
            
    def finalizeRecord(self, reward, domainInControl):
        '''
        Records for each committee member the reward and the domain the dialogue has been on 
        
        :param reward: the final reward to be recorded
        :type reward: int
        :param domainInControl: the domain the reward was achieved in
        :type domainInControl: str
        '''
        # TODO how is final scaling computed?
        if self.learning_method == "multiagent":
            for dstring in self.members:
                self.manager.domainPolicies[dstring].finalizeRecord(reward, domainInControl)
        elif self.learning_method == "singleagent":
            self.manager.domainPolicies[domainInControl].finalizeRecord(reward,
                                                                 domainInControl)
        return
            
    def _bayes_committee_calculator(self, domainQs, priors, domainInControl, scale):
        '''
        Given means and variances of committee members - forms the Bayesian committee distribution for each action, draws
        sample from each, returns act with highest sample. 
         
        .. note::
            this implementation is probably **slow** -- can reformat domainQs - and redo this via matricies and slicing
            
        :param domainQs: the means and variances of all Q-value estimates of all domains
        :type domainQs: dict of domains and dict of actions and dict of variance/mu and values
        :param priors: the prior of the Q-value
        :type priors: dict of actions and values
        :param domainInControl: domain the dialoge is in
        :type domainInControl: str
        :param scale: a scaling factor used to control exploration during learning
        :type scale: float
        :returns: the next abstract system action
        '''
        bcm_samples = {}
        
        for act_i in domainQs[domainInControl]:          
            com_members_means_act_i = [domainQs[dstring][act_i]['mu'] for dstring in domainQs.keys() if act_i in domainQs[dstring]]
            com_members_sigmas_act_i = [domainQs[dstring][act_i]['variance'] for dstring in domainQs.keys() if act_i in domainQs[dstring]]
            
            M = len(com_members_means_act_i) 
            
            # Get BCM mean and sigma for this act:
            for i in range(len(com_members_sigmas_act_i)):
                if com_members_sigmas_act_i[i] < .0000001:
                    #print 'WARNING!!!! com_members_sigmas_act_i is zero', com_members_sigmas_act_i
                    com_members_sigmas_act_i[i] = .0000001

            com_members_sigmas_act = np.reciprocal(com_members_sigmas_act_i)
            '''for i in range(len(com_members_sigmas_act)):
                if np.isnan(com_members_sigmas_act[i]):
                    com_members_sigmas_act[i] = 0.000001'''

            bcm_sigma_act = float(1-M)/priors[act_i] + np.sum(com_members_sigmas_act)

            if bcm_sigma_act < .0000001:
                #print 'WARNING!!!! bcm_sigma_act_i is zero', bcm_sigma_act
                bcm_sigma_act = .0000001

            bcm_sigma_act_i = np.reciprocal(bcm_sigma_act)
            bcm_mean_act_i = bcm_sigma_act_i * np.sum(np.divide(com_members_means_act_i,com_members_sigmas_act_i))            
            
            # Sample a Q value for this act from BCM:        
            bcm_samples[act_i] = scale*np.sqrt(bcm_sigma_act_i) * Settings.random.randn() + bcm_mean_act_i  # sample from gaussian            
            
        # Take action with the largest sampled Q: 
        act = bcm_samples.keys()[np.argmax(bcm_samples.values())]

        if self.learning_method == "multiagent":
            self._set_multi_agent_learning_weights(comm_meansVars=domainQs, chosen_act=act)
        return act
    
    def _set_multi_agent_learning_weights(self, comm_meansVars, chosen_act):
        '''
        Set reward scalings for each committee member. Implements NAIVE approach from
        "Multi-agent learning in multi-domain spoken dialogue systems", Milica Gasic et al. 2015.
        
        :param comm_meansVars: the means and variances of all committee members
        :type comm_meansVars: dict of domains and dict of actions and dict of variance/mu and values
        :param chosen_act: the abstract system action to be executed 
        :type chosen_act: str
        :returns: None
        '''
        # set dictionary indexed by domain_names in committee - with weight for each
        self.multiagent_weights = {}
                
        for dstring in comm_meansVars:  # dstrings in committee
            if chosen_act in comm_meansVars[dstring]:
                ''' skip the domains which do not have chosen_act
                '''
                self.multiagent_weights[dstring] = 1.
        
        return 
    
#END OF FILE
