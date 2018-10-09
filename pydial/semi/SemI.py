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
SemI.py - Semantic input parser
===================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies:

    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`semi.SemIContextUtils` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"


from utils import ContextLogger, Settings
from ontology import OntologyUtils
import SemIContextUtils as contextUtils
logger = ContextLogger.getLogger('')



class SemI(object):
    '''
    Interface class SemI, it defines the method decode that all the SemI classes must implement
    '''

    def decode(self, ASR_obs, sys_act=None, turn=None):
        '''
        The main interface method for semantic decoding. It takes the ASR input and returns a list of semantic interpretations.
        
        This method must be implemented by a sub-class.
        
        :param ASR_obs: the ASR observation list
        :param ASR_obs: list
        :param sys_act: the last system action, optional
        :param sys_act: str
        :param turn: the current turn number, optional
        :param turn: int
        :return: list of semantic interpretations of the input
        '''
        pass # Generate semantic representation from the asr output

    
        

class SemIManager(object):
    '''
    The Semantic Input Manager contains a dictionary with all the SemI objects currently running, the key of the dictionary is the
    domain tag

    '''

    def __init__(self):
        '''
        When Initialised it tries to load all the domains, in the case there are configuration problems these SemI objects are not loaded.
        :return:
        '''
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia','ood']
        self.semiManagers = dict.fromkeys(OntologyUtils.available_domains)
        for domainTag in self.semiManagers.keys():
            try:
                self.semiManagers[domainTag] = self._load_domains_semi(dstring=domainTag)
            except AttributeError:
                continue

        

    def _ensure_booted(self, domainTag):
        '''
        Boots up the semi ability for the specified domain if required
        
        :param domainTag: domain description
        :type domainTag: str
        :return: None
        '''
        if self.semiManagers[domainTag] is None:
            self.semiManagers[domainTag] = self._load_domains_semi(dstring=domainTag)
        return

    def _load_domains_semi(self, dstring):
        '''
        Get from the config file the SemI choice of method for this domain and load it.
        If you want to add a new semantic input parser you must add a line in this method calling explicitly
        the new SemI class, that must inherit from SemI
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.

        :param dstring: the name of the domain
        :type dstring: str
        :return: the class with the SemI implementation
        '''


        # 1. get type:
        #semi_type = 'PassthroughSemI'  # domain+resource independent default
        semi_type = 'RegexSemI'
        if Settings.config.has_option('semi_'+dstring, 'semitype'):
            semi_type = Settings.config.get('semi_'+dstring, 'semitype')
            
        parsing_method = None
        
        if dstring in self.SPECIAL_DOMAINS:
            semi_type = 'RegexSemI'

        # 2. And load that method for the domain:
        if semi_type == 'PassthroughSemI':
            from RuleSemIMethods import PassthroughSemI
            parsing_method = PassthroughSemI()
        elif semi_type == 'RegexSemI':
            from RuleSemIMethods import RegexSemI
            parsing_method = RegexSemI(domainTag=dstring)
        # To load DLSemI, looks for 'DLSemI' in the config file, see texthub_dl.cfg
        elif semi_type == 'DLSemI':
            from DeepLSemI import DeepLSemI
            parsing_method = DeepLSemI()
        # To load SVMSemI, see texthub_svm.cfg
        elif semi_type == 'SVMSemI':
            from SVMSemI import SVMSemI
            parsing_method = SVMSemI()
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = semi_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                parsing_method = klass(dstring)
            except ImportError:
                logger.warning('Unknown semantic decoder "{}" for domain "{}". Using PassthroughSemI.'.format(semi_type, dstring))
                from RuleSemIMethods import PassthroughSemI
                parsing_method = PassthroughSemI()

        return parsing_method
            
    # Added turn as an optional argument, only needed by DLSemI
    def decode(self, ASR_obs, sys_act, domainTag, turn=None):
        '''
        The main method for semantic decoding. It takes the ASR input and returns a list of semantic interpretations. To decode, the task is delegated to the respective domain semantic decoder
        
        :param ASR_obs: ASR hypotheses
        :type ASR_obs: list
        :param sys_act: is the system action prior to collecting users response in obs.
        :type sys_act: str
        :param domainTag: is the domain we want to parse the obs in
        :type domainTag: str
        :param turn: the turn id, this parameter is optional
        :type turn: int
        :return: None
        '''

        # use the correct domain:
        self.active_domain = domainTag # used down call chain in adding context
        self._ensure_booted(domainTag)
        
        # --------
        # explore how to clean IBM asr - "i dont care" problem ...    
        for i in range(len(ASR_obs)):            
            was = ASR_obs[i][0]
            fix = [str(c) for c in was if c.isalpha() or c.isspace() or c=="'" or c!="!" or c!="?"]     # slow way - initial fix #lmr46 06/09/16 this filtering filters question marks that are important!!!! syj requirement
            res = ''.join(fix)
            ASR_obs[i] = (res.rstrip(), ASR_obs[i][1])
        #---------------------------------------------------  
        
        # Additional argument turn as described above
        hyps = self.semiManagers[domainTag].decode(ASR_obs, sys_act, turn=turn)
        logger.info(hyps)
        # add context if required
        hyps = contextUtils._add_context_to_user_act(sys_act,hyps,self.active_domain)
        return hyps

    def clean_possible_texthub_switch(self,userActText):
        '''
        NB: only for texthub.py
        
        This removes switch("Domain") - as you may enter in texthub if using the switch topic tracker
        You can add domain information after e.g.: switch("CamRestaurants")i want a cheap restaurant
        
        :param userActText: list of user act hypothesis?
        :return:
        '''

        text_first_hyp = userActText[0][0]    # userActText is [('switch("CamRestaurants")',1.0)]
        if 'switch("' in text_first_hyp:
            tmp = text_first_hyp.split('"') 
            if len(tmp) > 2 and len(tmp[2]): # remove the ) in switch("CamRestaurants")
                assert(tmp[2][0]==')') 
                tmp[2] = tmp[2][1:]   
            cleaned = "".join(tmp[2:])            
            return [(cleaned,1.0)]      # TODO -- will need fixing if simulated errors are introduced into texthub
        return userActText
    
    
    def simulate_add_context_to_user_act(self, sys_act, user_acts, domainTag):
        '''
        NB: only for Simulate.py
        
        While for simulation no semantic decoding is needed, context information needs to be added in some instances. This is done with this method.
        
        :param sys_act: the last system act
        :type sys_act: str
        :param user_acts: user act hypotheses
        :type user_acts: list
        :param domainTag: the domain of the dialogue
        :type domainTag: str
        '''
        # simulate - will only pass user_act, and we will return contextual user act
        self.active_domain = domainTag  # this is required below in checking binary slots of given domain
        hyps = contextUtils._add_context_to_user_act(sys_act,hyps=user_acts,active_domain=self.active_domain)
        return hyps  # just the act
    
#END OF FILE
