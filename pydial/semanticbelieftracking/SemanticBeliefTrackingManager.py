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
from utils.DialogueState import DialogueState

'''
SemanticBeliefTrakingManager.py - module handling mapping from words to belief state
====================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''

from utils import ContextLogger, Settings
from ontology import OntologyUtils
logger = ContextLogger.getLogger('')


class SemanticBeliefTracker(object):   
    '''
    The class specifying the interface for all per domain semantic belief trackers.  
    ''' 
    def __init__(self,dstring):
        self.domainString = dstring
        self.prevbelief = None
        
    def restart(self, previousDomainString = None):
        '''
        Restarts the semantic belief tracker and resets internal variables. If special handling is required if a domain switch has occurred, this method should be overridden in a sub-class.
        
        :param previousDomainString: the domain identifier string of the previous domain, obtional, default None
        :type previousDomainString: str
        :returns: None
        '''
        self.prevbelief = None

######################################################################
# interface methods
######################################################################

    def update_belief_state(self, ASR_obs, sys_act, constraints, turn=None, hub_id = None):
        '''
        Updates the belief state based on the given input. 
        
        Should be implemented in a sub-class.
        
        :param ASR_obs: the list of ASR hypotheses
        :type ASR_obs: list
        :param sys_act: the last system act (necessary for deriving context information)
        :type sys_act: str ?
        :param  constraints: internal constraints derived from previous domain in case of domain switch
        :type  constraints: list
        :param turn: the current turn number
        :type turn: int
        :param hub_id: the hub id (identifying texthub vs. simulate vs. dialogue server)
        :type hub_id: string
        :returns: the updated belief state and the last semantic hypotheses if present, otherwise empty list
        '''
        pass
    
    def hand_control(self, previousDomain):
        '''
        To hand over control from domainString to previousDomainString, this method is called to handle effects internally, e.g., transferring information from one domain to the other.
        
        Should be implemented in a sub-class.
        
        :param previousDomain: the domain identifier string of the previous domain in case there was a domain switch
        :type previousDomain: str
        :returns: a list of constraints
        '''
        pass
    
    
    def bootup(self, previousDomain):
        '''
        To boot up the semi tracker for the given domain taking into account constraints of previousDomain, this method is called to handle effects internally, e.g., transferring information from one domain to the other.
        
        Should be implemented in a sub-class.
        
        :param previousDomain: the domain identifier string of the previous domain in case there was a domain switch
        :type previousDomain: str
        :returns: a list of constraints
        '''
        pass


class SemanticBeliefTrackingManager(object):
    '''
    The semantic belief tracking manager manages the semantic belief trackers for all domains. A semantic belief tracker is a mapping from words/ASR input to belief state.
    
    Its main interface method is :func:`update_belief_state` which updates and returns the internal belief state on given input.
    
    Internally, a dictionary is maintained which maps each domain to a :class:`SemanticBeliefTracker` object which handles the actual belief tracking.
    '''
    
    def __init__(self):
        self.domainSemiBelieftrackers = dict.fromkeys(OntologyUtils.available_domains, None)
        self.constraints = None
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia','ood']
        
        self.state = DialogueState()
        
        for dstring in self.domainSemiBelieftrackers:
            if Settings.config.has_option("semibelief_" + dstring, "preload"):
                preload = Settings.config.getboolean("semibelief_" + dstring, "preload")
                if preload:
                    self.bootup(dstring)
    
    def update_belief_state(self, dstring, ASR_obs, sys_act, turn=None, hub_id=None, sim_lvl='dial_act'):
        '''
        Updates the belief state of the specified domain based on the given input. 
        
        :param dstring: the domain identifier string
        :type dstring: str
        :param ASR_obs: the list of ASR hypotheses
        :type ASR_obs: list
        :param sys_act: the last system act (necessary for deriving context information)
        :type sys_act: str ?
        :param turn: the current turn number
        :type turn: int
        :param hub_id: the hub id (identifying texthub vs. simulate vs. dialogue server)
        :type hub_id: string
        :returns: the updated belief state
        '''
       
        if self.domainSemiBelieftrackers[dstring] is None:
            self.bootup(dstring)
        
        self.state.domainStates[dstring] = self.domainSemiBelieftrackers[dstring].update_belief_state(ASR_obs, sys_act, self.constraints, turn, hub_id, sim_lvl)
        self.state.currentdomain = dstring
        
        return self.state
        
    def hand_control(self, domainString, previousDomainString=None):
        '''
        To hand over control from domainString to previousDomainString, this method is called to handle effects internally, e.g., transferring information from one domain to the other.
        
        Calls :func:`SemanticBeliefTracker.hand_control` of the domain domainString and sets the internal constraints accordingly.
        
        :param domainString: the domain identifier string
        :type  domainString: str
        :param previousDomainString: the domain identifier string of the previous domain in case there was a domain switch, optional
        :type  previousDomainString: str
        '''
        self.constraints = self.domainSemiBelieftrackers[domainString].hand_control(previousDomainString)
    
    def bootup(self, domainString, previousDomainString=None):
        '''
        Loads a semi belief tracker for a given domain.
        
        :param domainString: the domain identifier string
        :type domainString: str
        :param previousDomainString: the domain identifier string of the previous domain in case there was a domain switch, optional
        :type previousDomainString: str
        :returns: None
        '''
        
        # with BCM if domain was in a committee -- then its policy can have already been loaded. check first:
        if self.domainSemiBelieftrackers[domainString] is not None:
            logger.warning('{} semi belief tracker is already loaded'.format(domainString))
        else:
            self._load_domains_semantic_belief_tracker(domainString)
            self.constraints = self.domainSemiBelieftrackers[domainString].bootup(previousDomainString)
        return
    
    def restart(self):
        '''
        Restarts all semantic belief trackers of all domains and resets internal variables.
        '''
        for dstring in self.domainSemiBelieftrackers.keys():
            if self.domainSemiBelieftrackers[dstring] is not None: 
                self.domainSemiBelieftrackers[dstring].restart()
        self.constraints = None
        self.state = DialogueState()
        return
    
    def getDomainBelief(self,operatingDomain):
        '''
        Returns the belief state of the specified domain.
        
        :param operatingDomain: the domain identifier string
        :type operatingDomain: str
        :returns: the belief state of the specified domain
        '''
        if self.domainSemiBelieftrackers[operatingDomain] is not None:
            return self.domainSemiBelieftrackers[operatingDomain].prevbelief
    
    def _load_domains_semantic_belief_tracker(self, domainString):
        '''
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        '''
        
        semiBeliefType = "modularSemiBelief"
        
        
            
        
        if not Settings.config.has_section('semibelief_' + domainString):
            if domainString not in self.SPECIAL_DOMAINS: # it is ok to load modular semantic belief tracker for special domains
                logger.debug("No semibelief section specified for domain {} - defaulting to modular semibelief".format(domainString))
        else:
            if Settings.config.has_option('semibelief_' + domainString, 'type'):
                semiBeliefType = Settings.config.get('semibelief_' + domainString, 'type')
                
                
        if semiBeliefType == "modularSemiBelief":
            from ModularSemanticBeliefTracker import ModularSemanticBeliefTracker
            self.domainSemiBelieftrackers[domainString] = ModularSemanticBeliefTracker(domainString)
        
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = semiBeliefType.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.domainSemiBelieftrackers[domainString] = klass(domainString)
            except ImportError:
                logger.error('Invalid semantic belief tracking type "{}" for domain "{}"'.format(semiBeliefType, domainString))
