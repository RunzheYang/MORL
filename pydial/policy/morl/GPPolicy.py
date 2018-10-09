###############################################################################
# CUED PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015, 2016
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

Copyright CUED Dialogue Systems Group 2015, 2016

   
**Relevant Config variables** [Default values]::

    [mogp]
    randomWeightLearning = False
    weights = 0.5 0.5
    
    
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.GPLib` |.|
    import :mod:`policy.Policy` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

import policy.GPPolicy as gp
from policy import Policy
from ontology import Ontology
from utils import ContextLogger, Settings
import numpy as np
logger = ContextLogger.getLogger('')

class GPPolicy(gp.GPPolicy):
    '''
    An implementation of the dialogue policy based on Gaussian process and the GPSarsa algorithm to optimise actions where states are GPState and actions are GPAction.
    
    The class implements the public interfaces from :class:`~Policy.Policy` and :class:`~PolicyCommittee.CommitteeMember`.
    '''
    def __init__(self, domainString, learning):  
        super(GPPolicy, self).__init__(domainString,learning)
        
        
        
#########################################################
# overridden methods from gp.GPPolicy
######################################################### 
    
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
            if isinstance(state,Policy.TerminalState):
                cState = gp.TerminalGPState()
            else:
                cState = self.get_State(state)
                
        if not isinstance(action,gp.GPAction):
            if isinstance(action,Policy.TerminalAction):
                cAction = gp.TerminalGPAction()
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

    
    

    
class GPState(gp.GPState):
    '''
    Definition of state representation needed for GP-SARSA algorithm
    Main requirement for the ability to compute kernel function over two states
    '''    
    def __init__(self, belief, keep_none=False, replace={}, domainString=None): 
        super(GPState,self).__init__(belief, keep_none, replace, domainString)

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
            
        self._bstate['weights'] = self._loadWeights()
            
        # Tom's speedup: convert belief dict to numpy vector 
        self.beliefStateVec = self.slowToFastBelief(self._bstate)

    def _loadWeights(self):
        weights = [0.5,0.5]
        if Settings.config.has_option("mogp_"+self.domainString,"weights"):
            weights = Settings.config.get("mogp_"+self.domainString,"weights").split()
            weights = map(float, weights)
            logger.info("MOGP weights read from config: {}".format(weights))
        return weights
    
    
    def slowToFastBelief(self, bdic) :
        '''Converts dictionary format to numpy vector format
        '''
        values = np.array([])
        for slot in sorted(bdic.keys()) :
            if slot == "hist_location":
                continue

            normVec = np.array(bdic[slot])
            divisor = np.linalg.norm(normVec, 2)
            if divisor:
                normVec = normVec / divisor
#                 np.divide(normVec, divisor, out=normVec)
                
            values = np.concatenate((values, normVec))
        return values
