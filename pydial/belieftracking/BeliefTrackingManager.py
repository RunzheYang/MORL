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
BeliefTrackingManager.py - wrapper for belief tracking across domains
======================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"
from utils import Settings
from ontology import OntologyUtils
from utils import ContextLogger
logger = ContextLogger.getLogger('') 


#--------------------------------------------------------------------------------------------------
# BELIEF MANAGER - controls each domains belief tracking abilities
#--------------------------------------------------------------------------------------------------
class BeliefTrackingManager(object):
    '''
    Higher-level belief tracker manager 
    '''
    def __init__(self):
        self.domainBeliefTrackers = dict.fromkeys(OntologyUtils.available_domains, None)
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia','ood']
        self.CONDITIONAL_BELIEF = False
        if Settings.config.has_option("conditional","conditionalbeliefs"):
            self.CONDITIONAL_BELIEF = Settings.config.getboolean("conditional","conditionalbeliefs")
        self.prev_domain_constraints = None
    
    def restart(self):
        '''
        Restart every alive Belief Tracker
        '''
        for dstring in self.domainBeliefTrackers.keys():
            if self.domainBeliefTrackers[dstring] is not None: 
                self.domainBeliefTrackers[dstring].restart()
        if self.CONDITIONAL_BELIEF:
            self.prev_domain_constraints = dict.fromkeys(OntologyUtils.available_domains) # used to facilate cond. behaviour
    
    def update_belief_state(self, dstring, lastSysAct, inputActs, constraints):
        '''
        Update belief state given infos
        '''
        return self.domainBeliefTrackers[dstring].update_belief_state(lastSysAct, inputActs, constraints)

    def bootup(self, domainString, previousDomainString=None):
        '''
        Boot up the belief tracker
        
        :param domainString: domain name
        :type domainString: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        '''
        self._load_domains_belieftracker(domainString)
        return self.conditionally_init_new_domains_belief(domainString, previousDomainString)
        

    def conditionally_init_new_domains_belief(self, domainString, previousDomainString):
        """
        If just starting this domain in this dialog: Get count80 slot=value pairs from previous
        domains in order to initialise the belief state of the new domain (reflecting dialogs history
        and likelihood that similar values will be desired if there are slot overlaps.

        :param domainString: domain name
        :type domainString: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        """
        if previousDomainString in self.SPECIAL_DOMAINS:
            return  # no information from these domains to carry over 
        if previousDomainString is not None and self.CONDITIONAL_BELIEF:
            # 1. get 'count80' slot=values:
            self.prev_domain_constraints[previousDomainString] = self.domainBeliefTrackers[previousDomainString].getBelief80_pairs()
            # 2. initialise belief in (this dialogs) new domain:
            return self.domainBeliefTrackers[domainString].get_conditional_constraints(self.prev_domain_constraints)  
        else:
            return
        
    def _load_domains_belieftracker(self, domainString=None):
        '''
        Load domain's belief tracker

        :param domainString: domain name
        :type domainString: string

        :return: None
        '''
        belief_type = 'focus'
        
        if Settings.config.has_option('policy_'+domainString, 'belieftype'):
            belief_type = Settings.config.get('policy_'+domainString, 'belieftype')

        if belief_type == 'focus':
            from baseline import FocusTracker
            self.domainBeliefTrackers[domainString] = FocusTracker(domainString)
        elif belief_type == 'baseline':
            from baseline import BaselineTracker
            self.domainBeliefTrackers[domainString] = BaselineTracker(domainString)
#         elif belief_type == 'rnn':
#             beliefs = BeliefTrackerRules.RNNTracker()
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = belief_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.domainBeliefTrackers[domainString] = klass(domainString)
            except ImportError:
                logger.error('Invalid semantic belief tracking type "{}" for domain "{}"'.format(belief_type, domainString))

#END OF FILE
