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
ModularSemanticBeliefTraker.py - separate modelling of semantic decoding and belief tracking
============================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`belieftracking.BeliefTrackingManager` |.|
    import :mod:`semi.SemI` |.|
    import :class:`semanticbelieftracking.SemanticBeliefTrackingManager.SemanticBeliefTracker`

*********************************************************************************************

'''

from utils import ContextLogger
from SemanticBeliefTrackingManager import SemanticBeliefTracker
from semi import SemI
from belieftracking import BeliefTrackingManager

logger = ContextLogger.getLogger('')

    
class ModularSemanticBeliefTracker(SemanticBeliefTracker):
    '''
    This class implements the functionality of the original spoken dialogue systems pipeline where semantic decoding and belief tracking
    are looked at as two separate problems. Refers all requests to :class:`semi.SemI.SemIManager` and :class:`belieftracking.BeliefTrackingManager.BeliefTrackingManager`.
    '''
    
    belief_manager = None
    semi_manager = None
    
    def __init__(self, domainString):
        super(ModularSemanticBeliefTracker, self).__init__(domainString)
        
        self.semi_manager = ModularSemanticBeliefTracker.getSemiManager()
        self.belief_manager = ModularSemanticBeliefTracker.getBeliefTrackingManager()

        self.lastHyps = []
    
    def update_belief_state(self, ASR_obs, sys_act, constraints, turn=None, hub_id = None, sim_lvl='dial_act'):
        
        # SEMI PARSING:
        if ASR_obs is not None:
            if hub_id == 'simulate':
                if sim_lvl=='dial_act':
                    ASR_obs = [(h.to_string(), h.P_Au_O) for h in ASR_obs]
                    self.lastHyps = self.semi_manager.simulate_add_context_to_user_act(sys_act, ASR_obs,
                                                                                       self.domainString)
                elif sim_lvl == 'text' or sim_lvl == 'sys2text':
                    #ASR_obs = self.semi_manager.clean_possible_texthub_switch(ASR_obs) # ic340: not sure if this is necesary
                    self.lastHyps = self.semi_manager.decode(ASR_obs=ASR_obs, sys_act=sys_act,
                                                             domainTag=self.domainString, turn=turn)
                    logger.dial('SemI > ' + str(self.lastHyps))
                else:
                    logger.error('sim level not recognised')
            else:
                if hub_id == 'texthub':
                    ASR_obs = self.semi_manager.clean_possible_texthub_switch(ASR_obs)
                self.lastHyps = self.semi_manager.decode(ASR_obs=ASR_obs, sys_act=sys_act,
                                                         domainTag=self.domainString, turn=turn)
                logger.info('SemI   > '+ str(self.lastHyps))
                #print '>>>>>>', self.lastHyps
        
        
        # 2. SYSTEM response:
        #--------------------------------------------------------------------------------------------------------------
        
        # SYSTEM ACT:
                # 1. Belief state tracking -- (currently just in single domain as directed by topic tracker)
        logger.debug('active domain is: '+self.domainString)
        self.prevbelief = self.belief_manager.update_belief_state(self.domainString, 
                                                             sys_act, self.lastHyps, constraints)
        self.prevbelief['userActs'] = self.lastHyps
        
        return self.prevbelief
    
    def restart(self, previousDomainString = None):
        super(ModularSemanticBeliefTracker,self).restart(previousDomainString)
        self.belief_manager.restart()
        self.lastHyps = []
        return
    
    def hand_control(self, previousDomain):
        return self.belief_manager.conditionally_init_new_domains_belief(self.domainString, previousDomain)
    
    def bootup(self, previousDomainString):
        return self.belief_manager.bootup(self.domainString, previousDomainString)
    
    @staticmethod
    def getBeliefTrackingManager():
        if ModularSemanticBeliefTracker.belief_manager is None:
            ModularSemanticBeliefTracker.belief_manager = BeliefTrackingManager.BeliefTrackingManager()
        return ModularSemanticBeliefTracker.belief_manager
    
    @staticmethod
    def getSemiManager():
        if ModularSemanticBeliefTracker.semi_manager is None:
            ModularSemanticBeliefTracker.semi_manager = SemI.SemIManager()
        return ModularSemanticBeliefTracker.semi_manager
        
