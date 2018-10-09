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
GPPolicy.py - Gaussian Process policy 
============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

   
**Relevant Config variables** [Default values]::

    [gppolicy]
    kernel = polysort
    thetafile = ''    

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.GPLib` |.|
    import :mod:`policy.Policy` |.|
    import :mod:`policy.PolicyCommittee` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

import math
import copy
import numpy as np
import json
import os
import sys
import time
from itertools import starmap, izip, combinations, product
from operator import mul    #,sub
from scipy.stats import entropy


from Policy import Policy, Action, State, TerminalAction, TerminalState
from policy import PolicyCommittee, SummaryUtils
from GPLib import GPSARSA
from ontology import Ontology
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')

class GPPolicy(Policy,PolicyCommittee.CommitteeMember):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning, sharedParams=None):
        super(GPPolicy, self).__init__(domainString,learning)
        
        inpolicyfile = ''
        outpolicyfile = ''
        
        if Settings.config.has_option('policy', 'inpolicyfile'):
            inpolicyfile = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy', 'outpolicyfile'):
            outpolicyfile = Settings.config.get('policy', 'outpolicyfile')
        if Settings.config.has_option('policy_'+domainString, 'inpolicyfile'):
            inpolicyfile = Settings.config.get('policy_'+domainString, 'inpolicyfile')
        if Settings.config.has_option('policy_'+domainString, 'outpolicyfile'):
            outpolicyfile = Settings.config.get('policy_'+domainString, 'outpolicyfile')
        
        # DEFAULTS:
        self.kerneltype = "polysort"
        self.thetafile = ""
        self.theta = [1.0, 1.0]
        self.action_kernel_type = 'delta'
        self.replace = {}    
        self.slot_abstraction_file = os.path.join(Settings.root, 'policy/slot_abstractions/'+domainString + '.json')       # default mappings
        self.abstract_slots = False
        self.unabstract_slots = False
        self.doForceSave = False
        self.beliefParametrisation = None
         
        # CONFIG:
        if Settings.config.has_option('gppolicy',"abstractslots"):
            self.abstract_slots = Settings.config.getboolean('gppolicy',"abstractslots")
        if Settings.config.has_option("gppolicy_"+domainString,"abstractslots"):
            self.abstract_slots = Settings.config.getboolean("gppolicy_"+domainString,"abstractslots")
        if not self.abstract_slots:
            # Check - in case you are using BCM but forgot to set abstractslots True in config
            if Settings.config.has_option('policycommittee','bcm'):
                if Settings.config.getboolean('policycommittee','bcm'):
                    if Settings.config.has_option("policycommittee","pctype"):
                        pcType =  Settings.config.get("policycommittee","pctype")
                        if pcType == 'configset':
                            try:
                                committeeMembers = Settings.config.get('policycommittee', 'configsetcommittee')
                                committeeMembers = committeeMembers.split(',')
                                if domainString in committeeMembers:
                                    logger.warning('You should set abstractslots to True for each domain involved in BCM - overriding here:')
                                    self.abstract_slots = True
                            except Exception: #ConfigParser.NoOptionError:  # can import ConfigParser if you wish 
                                pass
        if not self.abstract_slots:
            if Settings.config.has_option('gppolicy',"unabstractslots"):
                self.unabstract_slots = Settings.config.getboolean('gppolicy',"unabstractslots")
            if Settings.config.has_option("gppolicy_"+domainString,"unabstractslots"):
                self.unabstract_slots = Settings.config.getboolean("gppolicy_"+domainString,"unabstractslots")
          
        if Settings.config.has_option('gppolicy', "kernel"):
            self.kerneltype = Settings.config.get('gppolicy', "kernel")
        if Settings.config.has_option('gppolicy', "thetafile"):
            self.thetafile = Settings.config.get('gppolicy', "thetafile")
        if Settings.config.has_option('gppolicy', "slotabstractionfile"):
            self.slot_abstraction_file = Settings.config.get('gppolicy', "slotabstractionfile")
        if Settings.config.has_option('gppolicy', "actionkerneltype"):
            self.action_kernel_type = Settings.config.get('gppolicy', "actionkerneltype")
        if Settings.config.has_option('gppolicy', "doforcesave"):
            self.beliefParametrisation = Settings.config.getboolean('gppolicy', "doforcesave")

        if Settings.config.has_option("gppolicy_" + domainString, "kernel"):
            self.kerneltype = Settings.config.get("gppolicy_" + domainString, "kernel")
        if Settings.config.has_option("gppolicy_" + domainString, "thetafile"):
            self.thetafile = Settings.config.get("gppolicy_" + domainString, "thetafile")
        if Settings.config.has_option("gppolicy_" + domainString, "slotabstractionfile"):
            self.slot_abstraction_file = Settings.config.get("gppolicy_" + domainString, "slotabstractionfile")
        if Settings.config.has_option("gppolicy_" + domainString, "actionkerneltype"):
            self.action_kernel_type = Settings.config.get("gppolicy_" + domainString, "actionkerneltype")
        if Settings.config.has_option("gppolicy_" + domainString, "doforcesave"):
            self.beliefParametrisation = Settings.config.getboolean("gppolicy_" + domainString, "doforcesave")


        # Learning algorithm:
        self.learner = GPSARSA(inpolicyfile,outpolicyfile,domainString=domainString, learning=self.learning, sharedParams=sharedParams)
        
        # Load slot abstraction mapping - for everything BCM related         
        if self.abstract_slots and self.unabstract_slots:   # enforce some logic on your config settings:
            logger.error('Cant both be true - if abstracting, we keep dictionary abstract. If unabstracting dictionary, do so and\
            keep everything in real format. Adjust your config.')
        if self.abstract_slots or self.unabstract_slots:
            self._load_slot_abstraction_mapping()   # loads self.abstraction_mapping
            if self.abstract_slots:  
                # if using BCM & we have a GP policy previously with nonempty dictionary not used with BCM before
                self.replace = self.abstraction_mapping['real2abstract']    # will be used throughout 
                assert(len(self.replace))
                self._abstract_dictionary()                        
                # TODO - should be able to optionally write out the dictionary -- since it can take a while to abstract 
            if self.unabstract_slots:
                # Un abstract the dictionary if it was used in BCM before and not now:        
                self._unabstract_dictionary()
        
        # KERNEL:
        self._readTheta()        # kernel hyperparameters                
        self.kernel = Kernel(self.kerneltype, self.theta, None, self.action_kernel_type, self.actions.action_names, domainString)
        
        self._byeAction = None
        # TODO - sort out bye action
        #if 'bye' in self.actions.action_names:  # this is always the case. -- make this config controlled - for now just removing bye 
        #    self._byeAction = 'bye'  
        
        # Collect data for hyperparameter play:
        #self.collect_data = True
        #self.data_for_hp = CollectSuccessfulData()      
        
#########################################################
# overridden methods from Policy
######################################################### 
    
    def nextAction(self, belief):
        '''
        Selects next action to take based on the current belief and a list of non executable actions
        NOT Called by BCM
        
        :param belief:
        :type belief:
        :param hyps:
        :type hyps:
        :returns:
        '''
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        
        goalMethod = belief["beliefs"]["method"]
        if "finished" in goalMethod:
            if goalMethod["finished"] > 0.85 and self._byeAction is not None:
                return self._byeAction
            
        if self._byeAction is not None:
            nonExecutableActions.append(self._byeAction)
        currentstate = self.get_State(belief)
        executable = self._createExecutable(nonExecutableActions)
        
#         print "-------------- non-executable actions: {}".format(nonExecutableActions)
#         print "--------------     executable actions: "
#         for act in executable:
#             print act.toString()
#         
#         print "################## GPState"
#         print currentstate._bstate

        if len(executable) < 1:
            logger.error("No executable actions")

         
        """
        ordered_actions_with_Qsamples = self.learner.policy(state=currentstate, kernel=self.kernel, executable=executable)
        best_action = ordered_actions_with_Qsamples[0][0].act  # [0][1] is sampled Q value
        self.episode.latest_Q_sample_from_choosen_action = ordered_actions_with_Qsamples[0][1]
        """
        best_action, actions_sampledQ, actions_likelihood = self.learner.policy(
                                                                state=currentstate, kernel=self.kernel, executable=executable)
        
        summaryAct = self._actionString(best_action.act)
        
        if self.learning:                    
            best_action.actions_sampled_Qvalue = actions_sampledQ
            best_action.likelihood_choosen_action = actions_likelihood            
            
        self.actToBeRecorded = best_action
        # Finally convert action to MASTER ACTION
        masterAct = self.actions.Convert(belief, summaryAct, self.lastSystemAction)
        return masterAct
    
    def savePolicy(self, FORCE_SAVE=False):
        '''
        Saves the GP policy.
        
        :param belief:
        :type belief:
        '''
        if self.learning or (FORCE_SAVE and self.doForceSave):
            self.learner.savePolicy()
        
    def train(self):
        '''
        At the end of learning episode calls LearningStep for accumulated states and actions and rewards
        '''
        
        # SOMEWHAT TEMPORARY THING FOR HYPERPARAM PLAY
#         if self.collect_data:
#             if self.episode.rtrace[-1] == 20:  # check success
#                 self.data_for_hp.add_data(blist=self.episode.strace, alist=self.episode.atrace)
#             if self.data_for_hp.met_length():
#                 self.data_for_hp.write_data()
#                 raw_input('ENOUGH DATA COLLECTED')
#             return
#                 
        if self.USE_STACK: 
            self.episode_stack.add_episode(copy.deepcopy(self.episodes))    
            if self.episode_stack.get_stack_size() == self.PROCESS_EPISODE_STACK:
                self.process_episode_stack()
            return
        # process single episode
        else:
            for dstring in self.episodes:
                if self.episodes[dstring] is not None:
                    if len(self.episodes[dstring].atrace):   # domain may have just been part of committee but
                        # never in control - and whenever policy is booted an Episode() is created for its own domain ... 
                        episode = self.episodes[dstring]   
                        self._process_single_episode(episode)   
        return
    
    def convertStateAction(self, state, action):
        '''
        
        :param belief:
        :type belief:
        :param belief:
        :type belief:
        '''
        cState = state
        cAction = action
        
        if not isinstance(state,GPState):
            if isinstance(state,TerminalState):
                cState = TerminalGPState()
            else:
                cState = self.get_State(state)
                
        if not isinstance(action,GPAction):
            if isinstance(action,TerminalAction):
                cAction = TerminalGPAction()
            else:
                cAction = self.get_Action(action)

        return cState, cAction

#########################################################
# overridden methods from CommitteeMember
######################################################### 
    
    def get_State(self, beliefstate, keep_none=False):     
        '''
        Called by BCM
        
        :param beliefstate:
        :type beliefstate:
        :param keep_none:
        :type keep_none:
        '''
        return GPState(beliefstate, keep_none=keep_none, replace=self.replace, domainString=self.domainString)

    def get_Action(self, action):     
        '''
        Called by BCM
        
        :param action:
        :type action:
        '''   
        return GPAction(action, self.numActions, replace=self.replace)


    def getMeanVar_for_executable_actions(self, belief, abstracted_currentstate, nonExecutableActions):
        '''
        abstracted_currentstate - is abstracted by actual domain belief is coming from ...
        
        :param belief:
        :type belief:
        :param abstracted_currentstate:
        :type abstracted_currentstate:
        :param nonExecutableActions:
        :type nonExecutableActions:
        '''
        goalMethod = belief['beliefs']['method']
        if "finished" in goalMethod:            
            if goalMethod['finished'] > 0.85 and self._byeAction is not None:
                return self._byeAction
        if self._byeAction is not None:
            nonExecutableActions.append(self._byeAction)
        
        executable = self._createExecutable(nonExecutableActions) # own domains abstracted actions
        
        values = {}
        for executable_i in executable:           
            [mean, variance] = self.learner.QvalueMeanVar(state=abstracted_currentstate, action=executable_i, kernel=self.kernel)                      
            values[executable_i.act] ={'mu':mean, 'variance':variance} 
        return values        


    def getPriorVar(self, belief, act):
        '''
        Returns prior variance for a given belief and action
        
        :param belief:
        :type belief:
        :param act:
        :type act:
        '''
        action = GPAction(act, self.numActions, self.replace)
        state = self.get_State(belief)
        return self.learner.getPriorVar(state, action, self.kernel)

    
    
    def abstract_actions(self, actions):
        '''
        convert a list of domain acts to their abstract form based on self.replace
        
        :param actions:
        :type actions:
        '''
        if len(self.replace)>0:
            abstract = []
            for act in actions:
                if '_' in act:
                    [prefix,slot] = act.split('_')
                    if slot in self.replace:
                        abstract.append(prefix+'_'+self.replace[slot])
                    else:
                        abstract.append(act)
                else:
                    abstract.append(act)
            return abstract
        else:
            logger.error('No slot abstraction mapping has been given - check config')
              
    def unabstract_action(self, actions):
        '''
        action is a string
        
        :param actions:
        :type actions:
        '''
        if len(actions.split("_")) != 2:        # handle not abstracted acts like 'inform' or 'repeat' 
            return actions        
        [prefix, slot] = actions.split("_")
        if prefix == 'inform':              # handle not abstracted acts like 'inform_byname' or 'inform_requested'
            return actions
        else:                               # handle abstracted acts like 'request_slot00' or 'confirm_slot03'
            matching_actions = []
            for abs_slot in self.abstraction_mapping['abstract2real'].keys():
                if abs_slot == slot:                
                    match = prefix +'_'+ self.abstraction_mapping['abstract2real'][abs_slot]      
                    matching_actions.append(match)
            Settings.random.shuffle(matching_actions)                            
            return Settings.random.choice(matching_actions)
        
        logger.error('{} - no real slot found for this abstract slot'.format(actions)) 
        
#########################################################
# public methods
#########################################################  
            
    def getPolicyFileName(self):
        '''
        Returns the policy file name
        '''
        return self.policy_file
        
#########################################################
# private methods
######################################################### 
    
    def _readTheta(self):
        '''
        Kernel parameters
        '''
        if self.thetafile != "":
            f = open(self.thetafile, 'r')
            self.theta = []
            for line in f:
                line = line.strip()
                elems =line.split(" ")
                for elem in elems:
                    self.theta.append(float(elem))
                break
            f.close()

    def _createExecutable(self,nonExecutableActions):
        '''
        Produce a list of executable actions from non executable actions
        
        :param nonExecutableActions:
        :type nonExecutableActions:
        '''                
        executable_actions = []
        for act_i in self.actions.action_names:
            if act_i in nonExecutableActions:
                continue
            elif len(self.replace) > 0:                            # with abstraction  (ie BCM)                
                # check if possibly abstract act act_i is in nonExecutableActions
                if '_' in act_i:
                    [prefix,slot] = act_i.split('_')
                    if slot in self.replace.keys():
                        if prefix+'_'+self.replace[slot] not in nonExecutableActions:       # assumes nonExecutable is abstract 
                            executable_actions.append(GPAction(act_i, self.numActions, replace=self.replace))
                        else:
                            pass # dont add in this case
                    else:       # some actions like 'inform_byname' have '_' in name but are not abstracted
                        executable_actions.append(GPAction(act_i, self.numActions, replace=self.replace))
                else:           # only abstract actions with '_' in them like request_area --> request_slot1 etc
                    executable_actions.append(GPAction(act_i, self.numActions, replace=self.replace))                
            else:                   # no abstraction
                executable_actions.append(GPAction(act_i,self.numActions))    #replace not needed here - no abstraction
        return executable_actions



    def _actionString(self, act):
        '''
        Produce a string representation from an action - checking as well that the act coming in is valid
        Should only be called with non abstract action. Use _unabstract_action() otherwise
        
        :param act:
        :type act:
        '''        
        if act in self.actions.action_names:
            return act           
        logger.error('Failed to find action %s' % act)
        
    def _process_episode_stack(self, episode_stack):
        '''With BCM - items on the stack are now dictionaries (keys=domain names, values=Episode() instances)
        '''
        
        # copy original policy to observe how far we deviate from it as we sequentially move through our batch of episodes, updating
        self.orig_learner = copy.deepcopy(self.learner)  # nb: deepcopy is slow
        
        # process episodes - since adding BCM - now have domain_episodes -- 
        for episode_key in episode_stack.episode_keys():                    
            domain_episodes = episode_stack.retrieve_episode(episode_key)
            for dstring in domain_episodes:
                if domain_episodes[dstring] is not None:
                    if len(domain_episodes[dstring].atrace):   # domain may have just been part of committee but
                        # never in control - and whenever policy is booted an Episode() is created for its own domain ... 
                        self._process_single_episode(domain_episodes[dstring],USE_STACK=True)
        return 
    
    def _process_single_episode(self, episode, USE_STACK = False):
        if len(episode.strace) == 0:
            logger.warning("Empty episode")
            return
        if not self.learner.learning:
            logger.warning("Policy not learning")
            return

        episode.check()  # just checks that traces match up. 
        
        i=1
        r=0
        is_ratios = []
        while i< len(episode.strace) and self.learner.learning:
            
            # FIXME how are state/action-pairs recorded? generic or specific objects, ie, State or GPState?
            
            # pGPState = self.get_State(episode.strace[i-1])
            # pGPAction = self.get_Action(episode.atrace[i-1])
            # cGPState = self.get_State(episode.strace[i])
            # cGPAction = self.get_Action(episode.atrace[i])
            
            pGPState = episode.strace[i-1]
            pGPAction = episode.atrace[i-1]
            cGPState = episode.strace[i]
            cGPAction = episode.atrace[i]

            self.learner.initial = False
            self.learner.terminal = False
              
            if i == 1:
                self.learner.initial = True
            
            if i+1 == len(episode.strace) or isinstance(episode.strace[i],TerminalGPState):
                self.learner.terminal = True
                r = episode.getWeightedReward()
                
            
            #----- calculate ~ Importance Sampling Ratio - for sampled Q value from behaviour policy compared to learning policy
            # This is just experimental --
            if USE_STACK:                
                if i == 1:
                    pi_i_likelihood = self.learner.getLiklihood_givenValue(pGPState, 
                                                                                  pGPAction, 
                                                                                  self.kernel, 
                                                                                  pGPAction.actions_sampled_Qvalue)
                    is_ratios.append(pi_i_likelihood/pGPAction.likelihood_choosen_action)
                    
                # and current:
                pi_i_likelihood = self.learner.getLiklihood_givenValue(cGPState, 
                                                                                  cGPAction, 
                                                                                  self.kernel, 
                                                                                  cGPAction.actions_sampled_Qvalue)
                
                is_ratios.append(pi_i_likelihood/cGPAction.likelihood_choosen_action)
            #---------

            self.learner.LearningStep( pGPState, pGPAction, r , 
                                       cGPState, cGPAction, self.kernel)
            i+=1
            
            if (self.learner.terminal and i < len(episode.strace)):
                logger.warning("There are {} entries in episode after terminal state for domain {} with episode of domain {}".format(len(episode.strace)-i,self.domainString,episode.learning_from_domain))
                break
        logger.info("Number of dictionary points in domain {}: ".format(self.domainString) + str(len(self.learner.params['_dictionary'])))
        return
    
    def _unabstract_dictionary(self):
        for i in range(len(self.learner.params['_dictionary'])):
            # for back compatibility with earlier trained policies
            try:
                _ = self.learner.params['_dictionary'][i][0].is_abstract
            except AttributeError:
                self.learner.params['_dictionary'][i][0].is_abstract = False
                self.learner.params['_dictionary'][i][1].is_abstract = False
                                      
            # 0 index is state  
            if self.learner.params['_dictionary'][i][0].is_abstract:
                for islot in self.learner.params['_dictionary'][i][0]._bstate:
                    if 'slot' in islot: # covers 'goal_slot01' and 'goal_infoslot00' -- i.e all things abstracted:
                        try:
                            [prefix,slot] = islot.split('_')
                            real_name = prefix + '_' + self.abstraction_mapping['abstract2real'][slot]
                            self.learner.params['_dictionary'][i][0]._bstate[real_name] = self.learner.params['_dictionary'][i][0]._bstate.pop(islot)
                        except:
                            logger.warning('{} - slot not in mapping'.format(islot))
                self.learner.params['_dictionary'][i][0].is_abstract = False
            
            # 1 index is action. should always be the same as state ...
            if self.learner.params['_dictionary'][i][1].is_abstract:
                self.learner.params['_dictionary'][i][1].act = self._unabstract_action(self.learner.params['_dictionary'][i][1].act)
                self.learner.params['_dictionary'][i][1].is_abstract = False
        logger.info('Un-abstracted dictionary in domain {}'.format(self.domainString))
        return               
    
    def _abstract_dictionary(self):
        for i in range(len(self.learner.params['_dictionary'])):
            # for back compatibility with earlier trained policies
            try:
                _ = self.learner.params['_dictionary'][i][0].is_abstract
            except AttributeError:
                self.learner.params['_dictionary'][i][0].is_abstract = False
                self.learner.params['_dictionary'][i][1].is_abstract = False
            
            # 0 index is state    
            if not self.learner.params['_dictionary'][i][0].is_abstract:
                for islot in self.learner.params['_dictionary'][i][0]._bstate:
                    
                    #islot is real name --> must be abstracted                     
                    if '_' in islot:    # if not --> not abstract
                        if len(islot.split('_')) == 2:           # if not --> not abstract                 
                            [prefix,slot] = islot.split('_')
                            try:                            
                                abstract_name = prefix + '_' + self.replace[slot]
                                self.learner.params['_dictionary'][i][0]._bstate[abstract_name] = self.learner.params['_dictionary'][i][0]._bstate.pop(islot)
                            except:
                                logger.debug('{} - slot not in mapping'.format(islot))
                self.learner.params['_dictionary'][i][0].is_abstract = True
            
            # 1 index is action. abstraction status should always be the same as state ...
            if not self.learner.params['_dictionary'][i][1].is_abstract:
                act = self.learner.params['_dictionary'][i][1].act
                # Use GPAction instance method replaceAction() to perform the action abstraction:
                self.learner.params['_dictionary'][i][1].act = self.learner.params['_dictionary'][i][1].replaceAction(act, self.replace)
                self.learner.params['_dictionary'][i][1].is_abstract = True
        logger.info('Abstracted dictionary in domain {}'.format(self.domainString))
        return  
    
    def _load_slot_abstraction_mapping(self):
        '''Loads the slot mappings. self.replace does abstraction: request_area --> request_slot0 etc        
        '''        
        with open(self.slot_abstraction_file,'r') as f:
            self.abstraction_mapping = json.load(f)                    
        return
    
class Kernel(object):
    '''
    The Kernel class defining the kernel for the GPSARSA algorithm. 
    
    The kernel is usually divided into a belief part where a dot product or an RBF-kernel is used. The action kernel is either the delta function or a handcrafted or distributed kernel.
    '''    
    def __init__(self, kernel_type, theta, der=None, action_kernel_type='delta', action_names=None, domainString=None):
        '''
        self.kernel_type specifies the type of kernel
        self.theta are kernel parameters
        self.der is the derivative
        '''

        self.pad_warning_issued = False
        self.kernel_type = kernel_type
        self.theta = theta
        self.der = der    
        self.action_kernel_type = action_kernel_type  # 'delta' or 'hdc' or 'distributed'
        if self.action_kernel_type == 'distributed':
            if action_names == None:
                exit("You need to pass action names when using the distributed action kernel")
            ssqrIn = 20
            if Settings.config.has_option('gppolicy', "distlength"):
                ssqrIn = float(Settings.config.get('gppolicy', "distlength"))
            if Settings.config.has_option("gppolicy_"+domainString, "distlength"):
                ssqrIn = float(Settings.config.get("gppolicy_"+domainString, "distlength"))
            self._set_distibuted_action_vectors(action_names, ssqrIn)
            # DEBUG STATEMENT
            if self.distributed_action_kernel._debug:
                self.distributed_action_kernel._DEBUG_pretty_print()
                raw_input('Hold to inspect heat plot.')
        elif self.action_kernel_type == 'parameterised':
            self.parameterised_action_kernel = ParameterisedActionKernel(action_names)
        
        if self.kernel_type == "gausssort":
            if len(self.theta)!=2:
                logger.error("Parameters not given")
            

    def _set_distibuted_action_vectors(self, action_names, ssqrIn):
        self.distributed_action_kernel = DistributedActionVecKernel(vecFile=Settings.root+'./resources/vectors-50.txt', 
                                                                     action_names=action_names,
                                                                     ssqrIn=ssqrIn)
        return        

    def sigma(self):
        '''
        Noise parameter
        '''
        return self.theta[-1]

    @staticmethod
    def _polynomial_kernel(a, b): 
        '''
        Compute kernel of two vectors a and b, optionally they can be sorted
        NOTE: assume kernel always linear. No provisions for quadratic or higher powers etc...
        '''
        # su259: already sorted what should be sorted in GPState conversion
#         if sorting:
#             a = sorted(a, reverse=True)
#             b = sorted(b, reverse=True)
        return sum(starmap(mul,izip(a,b))) # this looks odd ... but it is simply computing dot product efficiently.
            
    @staticmethod
    def _gauss_kernel(a, b): 
        '''
        Compute kernel of two vectors a and b, optionally they can be sorted
        '''
        ker = 0.0
        # su259: already sorted what should be sorted in GPState conversion
#         if sorting:
#             a = sorted(a, reverse = True)       # think this sorting is safe regarding not modifying the object passed in
#             b = sorted(b, reverse=True)
            
        
        for i in xrange(len(a)):                # handles variable length when BCM is comparing beliefs from diff domains
            ker +=  (a[i]-b[i])*(a[i]-b[i])
           
        return ker
 

    #@profile
    def subBeliefKernel(self, aslot, a, b): 
        '''
        Kernel value for a particular slot
        NB: p = self.theta[0] -- sigmak = self.theta[1]  for gausssort kernel
        
        .. note:: 
            There was a convention to increase the importance of matching the parts of the belief that are sorted
            These were multiplied by 2, and unsorted was divided by /2 (multiply by 4 and leave alone would be better no?)
            Point is: this is now not present.
        '''        
        # su259: already sorted what should be sorted
#         if "goal" in aslot and aslot != "goal_discourseAct" and aslot != "goal_method":
#             sorting = True
#         else:
#             sorting = False
        
        if self.kernel_type == 'gausssort':
            return self._gauss_kernel(a, b) 
        elif self.kernel_type == 'polysort':
            return self._polynomial_kernel(a, b) 
        else:
            logger.error('not a valid kernel type {}'.format(self.kernel_type))
    

    def beliefKernel(self, ns, s):
        '''
        Kernel value between two GP states ns and s
        '''
        if ns.is_abstract != s.is_abstract:
            logger.error('Cant compare abstracted and real beliefs - check your config settings')
        
        ker = 0.0;
        
        # 1. Calculate actual kernel
        if self.kernel_type == 'polysort':
            # new fast linear kernel
            if len(ns.beliefStateVec) == len(s.beliefStateVec):
                ker = np.dot(s.beliefStateVec, ns.beliefStateVec)
            else:
                # old linear kernel for committee calculations
                if not self.pad_warning_issued:
                    #logger.warning('incorrect beliefStateVec padding/truncating, returning 0')
                    #logger.warning('ns: {} {}'.format(ns._bstate, len(ns.beliefStateVec)))
                    #logger.warning('s: {} {}'.format(s._bstate, len(s.beliefStateVec)))
                    logger.warning('ns:  {}'.format(len(ns.beliefStateVec)))
                    logger.warning('s:  {}'.format( len(s.beliefStateVec)))
                #sys.exit()
                ker = 0
                        
        # 2. DEAL WITH DERIVATIVES (depending on self.der)
        if self.kernel_type == 'gausssort':
            if self.der == None:
                ker = self.theta[0]*self.theta[0] * math.exp(-ker / self.theta[1]*self.theta[1])
            elif self.der == 0:
                ker = 2 * self.theta[0] * math.exp(-ker / self.theta[1]*self.theta[1])
            elif self.der == 1:
                ker = self.theta[0]*self.theta[0] * math.exp(-ker / self.theta[1]*self.theta[1]) * \
                        ker / self.theta[1]*self.theta[1]*self.theta[1]
            else:
                ker = 0     # if this happened frequently (or at all) --> would avoid all above calculation and check this first ...

        return ker
    
    
    def hdc_action_kernel(self,na,a):
        ''' Rough handcrafted generalisation away from just identity matrix. 
        '''
        small_val = 0.5  # attempt at hdc control on positive definiteness. 
        if '_' in na.act and '_' in a.act:
            # compare 2 parts for partial score
            na_intent, na_slot = na.act.split('_')
            a_intent, a_slot = a.act.split('_')
            match = 0.0
            if na_intent == a_intent:
                match += 0.5
            if na_slot == a_slot:
                match += 0.5
            return match if match in [0.0,1.0] else small_val
        else:
            return 1.0 if na.act==a.act else 0.0
    
    def ActionKernel(self, na, a):
        '''
        Kroneker delta on actions
        '''             
            
        if self.action_kernel_type == 'delta':
            return 1.0 if na.act == a.act else 0.0
        elif self.action_kernel_type == 'hdc':
            return self.hdc_action_kernel(na,a)
        elif self.action_kernel_type == 'distributed':
            return self.distributed_action_kernel.kernel_func(na.act,a.act)
        elif self.action_kernel_type == 'parameterised':
            return self.parameterised_action_kernel.kernel_func(na.act,a.act)
        elif self.action_kernel_type == '_externally_defined_':
            return self._externally_defined_(na.act,a.act)
        else:
            logger.error('Unknown action kernel type %s' % self.action_kernel_type)
    
    #def _externally_defined_(self, na_string,a_string):
    #    pass # action kernel to be overridden somewhere else

    def PriorKernel(self, ns, s):
        '''
        Prior Kernel is normalised
        '''
        core =  self.beliefKernel(ns, s)
        nskernel =  self.beliefKernel(ns, ns)
        skernel = self.beliefKernel(s, s)

        return core/math.sqrt(nskernel*skernel)

class ParameterisedActionKernel(object):   
    def __init__(self, action_names):
        '''action names is a list of strings
        '''
        self.action_names = action_names        # NB: list order will be referred to and order must stay the same
        self.lenA = len(action_names)
        self._init_params()
    
    def _init_params(self, paramIn=None):
        # TODO optionally load based on config
        if paramIn is None:  
            self._aVec = Settings.random.uniform(low=-1.0, high=1.0, size=self.lenA)
        return
    
    def _actionStr2index(self, act_str_i, act_str_j):
        return self.action_names.index(act_str_i), self.action_names.index(act_str_j)
    
    def kernel_func(self, act_str_i, act_str_j):
        i,j = self._actionStr2index(act_str_i, act_str_j)
        # quetion : add a idenitity matrix in here?
        
        # can precompute ...
        return self._aVec[i]*self._aVec[j]

class DistributedActionVecKernel(object):
    def __init__(self, vecFile, action_names, ssqrIn=20):
        
        # Values, defaults etc:
        self.normalise_vecs = False
        self._debug = False         # just a hardcoded switch - turn on to produce heat map
        self.precomputed_kernel_values_SET = False
        self.file = vecFile
        self.ssqr = ssqrIn  # length scale in sqr exponential
        
        # Init:
        self.action2vec = dict.fromkeys(action_names)    # action_name --> vec
        self._set_action_vecs(vecFile, action_names)
        self._precompute_kernel_pairs(action_names)
        
    def _handle_double_words(self, w2):
        if w2 == 'byname':
            return ['name']         # byname is not a word --> just take the essence (name) and ignore'by'
        elif w2 == 'reqmore':   # similar hack, keep both words though
            return ['request', 'more']
        #elif w2 == 'dogs'        # just for CamRestaurants for now
        else:
            return [w2]

    def _handle_single_words(self, w):
        if w == 'reqmore':
            return ['request', 'more']
        else:
            return [w]
                    
    def _get_required_words(self, action_names):
        req = {}
        for a in action_names:
            if '_' in a:
                w1,w2 = a.split('_')
                req[w1] = None
                w2 = self._handle_double_words(w2)  # returns a list of words. eg reqmore -> [request, more]
                for w in w2:
                    req[w] = None
            else:
                a = self._handle_single_words(a)
                for w in a:
                    req[w] = None
        return req
    
    def _set_action_vecs(self, vecFile, action_names):
        # get all required words
        required_words = self._get_required_words(action_names)
        # and their vecs
        with open(vecFile, "r") as f:
            for line in f:
                bits = line.split()
                word = bits[0]
                if word in required_words.keys():
                    required_words[word] = np.array(bits[1:], dtype=float)
        
        # set action vecs
        #TODO  AVERAGE BAG OF WORDS -- or just add?
        for a in self.action2vec.keys():
            if '_' in a:
                w1,w2 = a.split('_')
                self.action2vec[a] = np.copy(required_words[w1])   # first word when _ is present is always fine
                w2 = self._handle_double_words(w2)
                for w in w2:
                    self.action2vec[a] += np.copy(required_words[w])   # check w is a vec
            else:
                w1 = self._handle_single_words(a)   # deal with reqmore etc
                self.action2vec[a] = np.copy(required_words[w1[0]])    # always 1 word in list
                if len(w1) > 1:     # are there more words?
                    for w in w1[1:]:
                        self.action2vec[a] += np.copy(required_words[w])       # check w is a 50 dim vec 
        return
                
    def _sort_pair(self,na,a):
        return (na,a) if na < a else (a,na)      
                
    def _precompute_kernel_pairs(self, action_names):
        self.precomputed_kernel_values = {}
        for act_i in action_names:
            for act_j in action_names:
                key = self._sort_pair(act_i, act_j)
                if key not in self.precomputed_kernel_values:  # dont need .keys()
                    self.precomputed_kernel_values[key] = self.kernel_func(act_i, act_j)
                else:
                    pass # kernel is symmetric - so we wont calculate this pair
        
        # Just use these precomputed values from now on then:
        self.precomputed_kernel_values_SET = True
        if not self._debug:
            self.action2vec = None  # dont need this memory footprint
         
        
    def _retrieve_precomputed_pair(self,na,a):
        """ string acts na and a (eg 'request_area')
        """
        key = self._sort_pair(na, a)
        K = self.precomputed_kernel_values[key]
        return K
    
    def _squared_exponential(self, x,y):
        """
        """
        #NOTE -- if you change this value -- then need to remove return 1.0 directly for exact match case in kernel_func()
        #p = 1.0    #scaling - 
        diff = x-y
        K = np.exp(-np.dot(diff,diff)/self.ssqr)
        
        #from scipy.stats import logistic
        #K = logistic.cdf(K)
        from scipy.special import expit #much faster
        K = expit(K)
        
        """
        if 1:    # DEBUG
            import time
            print K
            time.sleep(0.25)
        """
        return K
        
    
    def kernel_func(self, na, a):
        """Use pre trained word vecs and distance btw bag of words
        NB: the act in is just a string (ie Action.act  is passed in NOT Action)
        """
        try:
            if na==a:
                return 1.0  # saves calculating, looking up
    
            if self.precomputed_kernel_values_SET:
                #-- With precomputed values:
                K = self._retrieve_precomputed_pair(na,a)
            else:
                # --- Without precomputed values:
                na_vec = self.action2vec[na]
                a_vec = self.action2vec[a]
                K = self._squared_exponential(na_vec, a_vec)
            return K
        except KeyError as e:
            logger.warning("Key error - not a valid action: " + str(e))
            return 0
        except Exception as e:
            print '*\n'*5   # want to see this if it happens...
            logger.warning("Something else went wrong " + str(e))
            return 0
    
    
#     def _DEBUG_pretty_print(self):
#         """ prints the actions by actions matrix of kernel values K
#         """
#         actions = self.action2vec.keys()
#         numA = len(actions)
#         K = np.zeros(shape=(numA,numA),dtype=float)  # can look at pairwise values  (IN a totally inefficient way)
#         for ai in range(numA):
#             for aj in range(numA):
#                 K[ai][aj] = self.kernel_func(actions[ai], actions[aj])
#         
#         # Pos def?
#         #is_pos_def
#         _ = np.all(np.linalg.eigvals(K) > 0)
#         
#         # PRETTY PRINT:
#         try:
#             import plotly.plotly as py
#             import plotly.graph_objs as go
#             data = [
#                     go.Heatmap( z=[list(r) for r in list(K)],
#                     x=actions,
#                     y=actions)
#                 ]
#             plot_url = py.plot(data, filename='Summary-action-distributed-kernel-heatmap')
#             #plotly.offline.plot(data, filename='Summary-action-distributed-kernel-heatmap')
#         except Exception as e:
#             print e
#         return
    
class GPAction(Action):
    '''
    Definition of summary action used for GP-SARSA.
    '''
    def __init__(self, action, numActions, replace={}):    
        self.numActions = numActions
        self.act=action     
        self.is_abstract = True if len(replace) else False           # record whether this state has been abstracted -   
        
                # append to the action the sampled Q value from when we chose it --> for access in batch calculations later
        self.actions_sampled_Qvalue = 0
        self.likelihood_choosen_action = 0
        
                
        if len(replace) > 0:
            self.act = self.replaceAction(action, replace)
        
                
    def replaceAction(self, action, replace):
        '''
        Used for making abstraction of an action
        '''
        if "_" in action:
            slot = action.split("_")[1]
            if slot in replace:
                replacement = replace[slot]
                return action.replace(slot, replacement)        # .replace() is a str operation
        return action


    def __eq__(self, a):
        """
        Action are the same if their strings match
        :rtype : bool
        """
        if self.numActions != a.numActions:
            return False
        if self.act != a.act:
                return False
        return True

    def __ne__(self, a):
        return not self.__eq__(a)

    def show(self):
        '''
        Prints out action and total number of actions
        '''
        print str(self.act), " ", str(self.numActions)


    def toString(self):
        '''
        Prints action
        '''
        return self.act
    
    def __repr__(self):
        return self.toString()
    
class GPState(State):
    '''
    Definition of state representation needed for GP-SARSA algorithm
    Main requirement for the ability to compute kernel function over two states
    '''    
    def __init__(self, belief, keep_none=False, replace={}, domainString=None):
        self.domainString = domainString
        # get constants
        self._bstate = {}
        self.keep_none = keep_none
        
        self.is_abstract = True if len(replace) else False           # record whether this state has been abstracted -
        #self.is_abstract = False 
        
        self.beliefStateVec = None
        
        #self.extractBelief(b, replace)
        if belief is not None:
            if isinstance(belief,GPState):
                self._convertState(belief, replace)
            else:
                self.extractSimpleBelief(belief, replace)


    def extractBeliefWithOther(self, belief, sort = True):
        '''
        Copies a belief vector, computes the remaining belief, appends it and returnes its sorted value
        
        :return: the sorted belief state value vector
        '''

        bel = copy.deepcopy(belief)
        res = []
        
        if '**NONE**' not in belief:
            res.append(1.0 - sum(belief.values()))  # append the none probability
        else:
            res.append(bel['**NONE**'])
            del bel['**NONE**']
            
        # ensure that all goal slots have dontcare entry for GP belief representation
        if 'dontcare' not in belief:
            bel['dontcare'] = 0.0
        
        if sort:
            # sorting all possible slot values including dontcare
            res.extend(sorted(bel.values(),reverse=True))
        else:
            res.extend(bel.values())
        return res

    def extractSingleValue(self, val):
        '''
        for a probability p returns a list  [p,1-p]
        '''
        return [val,1-val]
    

    def extractSimpleBelief(self, b, replace={}):
        '''
        From the belief state b extracts discourseAct, method, requested slots, name, goal for each slot,
        history whether the offer happened, whether last action was inform none, and history features.
        Sets self._bstate
        '''
        with_other = 0
        without_other = 0
        self.isFullBelief = True
        
        for elem in b['beliefs'].keys():
            if elem == 'discourseAct':
                self._bstate["goal_discourseAct"] = b['beliefs'][elem].values()
                without_other +=1
            elif elem == 'method':
                self._bstate["goal_method"] = b['beliefs'][elem].values()
                without_other +=1
            elif elem == 'requested' :
                for slot in b['beliefs'][elem]:
                    cur_slot=slot
                    if len(replace) > 0:
                        cur_slot = replace[cur_slot]
                    self._bstate['hist_'+cur_slot] = self.extractSingleValue(b['beliefs']['requested'][slot])
                    without_other +=1
            else:
                if elem == 'name':
                    self._bstate[elem] = self.extractBeliefWithOther(b['beliefs']['name'])
                    with_other +=1
                else:
                    cur_slot=elem
                    if len(replace) > 0:
                        cur_slot = replace[elem]

                    self._bstate['goal_'+cur_slot] = self.extractBeliefWithOther(b['beliefs'][elem])
                    with_other += 1

                    additionalSlots = 2
                    # if elem not in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
                    #     additionalSlots = 1
                    if len(self._bstate['goal_'+cur_slot]) !=\
                         Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+additionalSlots:
                        print self._bstate['goal_'+cur_slot]
                        logger.error("Different number of values for slot "+cur_slot+" "+str(len(self._bstate['goal_'+cur_slot]))+\
                            " in ontology "+ str(Ontology.global_ontology.get_len_informable_slot(self.domainString, slot=elem)+2)) 
                    

        self._bstate["hist_offerHappened"] = self.extractSingleValue(1.0 if b['features']['offerHappened'] else 0.0)
        without_other +=1
        self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(
                                                                1.0 if len(b['features']['informedVenueSinceNone'])>0 else 0.0)
        without_other +=1
        for i,inform_elem in enumerate(b['features']['inform_info']):
            self._bstate["hist_info_"+str(i)] = self.extractSingleValue(1.0 if inform_elem else 0.0)
            without_other +=1
            
        # Tom's speedup: convert belief dict to numpy vector 
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return
    

    def extractBelief(self, b, replace={}):
        '''NB - untested function since __init__ makes choice to use extractSimpleBelief() instead
        '''
        self.isFullBelief = True

        self._bstate["hist_lastActionInformNone"] = self.extractSingleValue(b["lastActionInformNone"])
        self._bstate["hist_offerHappened"] = self.extractSingleValue(b["offerHappened"])
        self._bstate["goal_name"] = self.extractBeliefWithOther(b["name"])
        self._bstate["goal_discourseAct"] = b["discourseAct"].values()
        self._bstate["goal_method"] = b["method"].values()
        
        for i in xrange(len(b["goal"])):
            curSlotName = b["slotAndName"][i]
            if len(replace) > 0:
                curSlotName = replace[curSlotName]
            self._bstate["goal_" + curSlotName] = self.extractBeliefWithOther(b["goal"][i]) 

        for i in range(min(len(b["slotAndName"]), len(b["goal_grounding"]))):
            histName = b["slotAndName"][i]
            if len(replace) > 0:
                histName = replace[histName]
            self._bstate["hist_" + histName] = b["goal_grounding"][i].values()

        for i in range(min(len(b["infoSlots"]), len(b["info_grounding"]))):
            infoName = b["infoSlots"][i]
            if len(replace) > 0:
                infoName = replace[infoName]
            self._bstate["hist_" + infoName] = b["info_grounding"][i].values()
            
        self.state_size = len(self._bstate)

    def slowToFastBelief(self, bdic) :
        '''Converts dictionary format to numpy vector format
        '''
        values = np.array([])
        for slot in sorted(bdic.keys()) :
            if slot == "hist_location":
                continue
#             if "goal" in slot and slot != "goal_discourseAct" and slot != "goal_method":
#                 toadd = np.array(bdic[slot])
#                 values = np.concatenate((values, np.sort(toadd)[::-1]))
#             else :
#                 values = np.concatenate((values, np.array(bdic[slot])))

            # su259 sorting already done before
            values = np.concatenate((values, np.array(bdic[slot])))
        return values
    
    def _convertState(self,b, replace={}):
        '''
        converts GPState to GPState of shape of current domain by padding/truncating slots/values
        
        assumes that non-slot information is the same for both
        '''
        
        # 1. take care of non-slot information
        self._bstate["goal_discourseAct"] = copy.deepcopy(b._bstate['goal_discourseAct'])
        self._bstate["goal_method"] = copy.deepcopy(b._bstate['goal_method'])
        
        self._bstate['hist_offerHappened'] = copy.deepcopy(b._bstate['hist_offerHappened'])
        self._bstate['hist_lastActionInformNone'] = copy.deepcopy(b._bstate['hist_lastActionInformNone'])
        
        # copy remaining hist information:
        for elem in b._bstate:
            if 'hist_info_' in elem:
                self._bstate[elem] = copy.deepcopy(b._bstate[elem])
                
        # requestable slots
        origRequestSlots = Ontology.global_ontology.get_requestable_slots(self.domainString)
        if len(replace) > 0:
            requestSlots = map(lambda x: replace[x], origRequestSlots)
        else:
            requestSlots = origRequestSlots
        
        
        for slot in requestSlots:
            if 'hist_'+slot in b._bstate:
                self._bstate['hist_'+slot] = copy.deepcopy(b._bstate['hist_'+slot])
            else:
                self._bstate['hist_'+slot] = self.extractSingleValue(0.0)
                
        # informable slots
        
        origInformSlots = Ontology.global_ontology.get_informable_slots(self.domainString)
        informSlots = {}
        for slot in origInformSlots:
            curr_slot = slot
            if len(replace) > 0:
                curr_slot = replace[curr_slot]
            informSlots[curr_slot] = Ontology.global_ontology.get_len_informable_slot(self.domainString, slot)+2 # dontcare + none
        
        slot = 'name'
        self._bstate[slot] = b._bstate[slot]
        if len(self._bstate[slot]) > informSlots[slot]:
            # truncate
            self._bstate[slot] = self._bstate[slot][0:informSlots[slot]]
        elif len(self._bstate[slot]) < informSlots[slot]: # 3 < 5 => 5 - 3
            # pad with 0
            self._bstate[slot].extend([0] * (informSlots[slot] - len(self._bstate[slot])))
        del informSlots[slot]
        
        for curr_slot in informSlots:
            slot = 'goal_'+curr_slot
            if slot in b._bstate:
                self._bstate[slot] = b._bstate[slot]
                if len(self._bstate[slot]) > informSlots[curr_slot]:
                    # truncate
                    self._bstate[slot] = self._bstate[slot][0:informSlots[curr_slot]]
                elif len(self._bstate[slot]) < informSlots[curr_slot]: # 3 < 5 => 5 - 3
                    # pad with 0
                    self._bstate[slot].extend([0] * (informSlots[curr_slot] - len(self._bstate[slot])))
            else:
                # create empty entry
                self._bstate[slot] = [0] * informSlots[curr_slot]
                self._bstate[slot][0] = 1.0 # the none value set to 1.0
            
        # Tom's speedup: convert belief dict to numpy vector 
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

        return
    
    def toString(self):
        '''
        String representation of the belief
        '''
        res = ""

        if len(self._bstate) > 0:
            res += str(len(self._bstate)) + " "
            for slot in self._bstate:
                res += slot + " "
                for elem in self._bstate[slot]:
                    for e in elem:
                        res += str(e) + " "
        return res
    
    def __repr__(self):
        return self.toString()
    
class TerminalGPAction(TerminalAction,GPAction):
    '''
    Class representing the action object recorded in the (b,a) pair along with the final reward. 
    '''
    def __init__(self):
        self.actions_sampled_Qvalue = None
        self.likelihood_choosen_action = None
        self.act = 'TerminalGPAction'
        self.is_abstract = None
        self.numActions = None

class TerminalGPState(GPState,TerminalState):
    '''
    Basic object to explicitly denote the terminal state. Always transition into this state at dialogues completion. 
    '''
    def __init__(self):
        super(TerminalGPState,self).__init__(None)

# END OF FILE
