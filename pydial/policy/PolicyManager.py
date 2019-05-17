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
PolicyManager.py - container for all policies
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''
__author__ = "cued_dialogue_systems_group"

from utils import Settings, ContextLogger
from ontology import Ontology,OntologyUtils
logger = ContextLogger.getLogger('')

class PolicyManager(object):
    '''
    The policy manager manages the policies for all domains. 
    
    It provides the interface to get the next system action based on the current belief state in :func:`act_on` and to initiate the learning in the policy in :func:`train`.
    '''
    def __init__(self):
        self.domainPolicies = dict.fromkeys(OntologyUtils.available_domains, None)
        self.committees = self._load_committees()
        self.shared_params = None
        
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia']
        
#         self.prevbelief = None
#         self.lastSystemAction = None
        
        for dstring in self.domainPolicies:
            if Settings.config.has_option("policy_"+dstring,"preload"):
                preload = Settings.config.getboolean("policy_"+dstring,"preload")
                if preload:
                    self.bootup(dstring)
      
    def savePolicy(self, FORCE_SAVE=False):
        """
        Initiates the policies of all domains to be saved.
        
        :param FORCE_SAVE: used to force cleaning up of any learning and saving when we are powering off an agent.
        :type FORCE_SAVE: bool
        """
        for dstring in self.domainPolicies.keys():
            if self.domainPolicies[dstring] is not None: 
                self.domainPolicies[dstring].savePolicy(FORCE_SAVE)
        return   
        
    def bootup(self, domainString):
        '''Loads a policy for a given domain. 
        '''
        # with BCM if domain was in a committee -- then its policy can have already been loaded. check first:
        if Settings.config.has_option('policycommittee', 'singlemodel') \
                and Settings.config.getboolean('policycommittee', 'singlemodel'):
            if self.shared_params is None:
                self.shared_params = {}
            self._load_domains_policy(domainString)
            self.domainPolicies[domainString].restart()

        elif self.domainPolicies[domainString] is not None:
            logger.warning('{} policy is already loaded'.format(domainString))
        else:
            self._load_domains_policy(domainString)
            self.domainPolicies[domainString].restart()
        return
    
    def act_on(self, dstring, state, preference=None):
        '''
        Main policy method which maps the provided belief to the next system action. This is called at each turn by :class:`~Agent.DialogueAgent`
        
        :param dstring: the domain string unique identifier.
        :type dstring: str
        :param state: the belief state the policy should act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :returns: the next system action as :class:`~utils.DiaAct.DiaAct`
        '''
        if self.domainPolicies[dstring] is None:
            self.bootup(dstring)
                                
        if self.committees[dstring] is not None:
            systemAct = self.committees[dstring].act_on(state=state, domainInControl=dstring)
        else:
            systemAct = self.domainPolicies[dstring].act_on(state=state, preference=preference)

        return systemAct
    
    def train(self, training_vec = None):
        '''
        Initiates the training for the policies of all domains. This is called at the end of each dialogue by :class:`~Agent.DialogueAgent`
        '''
        for domain in self.domainPolicies:
            if self.domainPolicies[domain] is not None and self.domainPolicies[domain].learning:
                if training_vec is not None:
                    if training_vec[domain]:
                        self.domainPolicies[domain].train()
                    else:
                        logger.info("No training due to evaluator decision.")
                else:
                    self.domainPolicies[domain].train()
        
    def record(self, reward, domainString):
        '''
        Records the current turn reward for the given domain. In case of a committee, the recording is delegated. 
        
        This method is called each turn by the :class:`~Agent.DialogueAgent`.
        
        :param reward: the turn reward to be recorded
        :type reward: np.array
        :param domainString: the domain string unique identifier of the domain the reward originates in
        :type domainString: str
        :returns: None
        '''
        if self.committees[domainString] is not None:
            self.committees[domainString].record(reward, domainString)
        else:
            self.domainPolicies[domainString].record(reward)
            
    def finalizeRecord(self, domainRewards):
        '''
        Records the final rewards of all domains. In case of a committee, the recording is delegated. 
        
        This method is called once at the end of each dialogue by the :class:`~Agent.DialogueAgent`. (One dialogue may contain multiple domains.)
        
        :param domainRewards: a dictionary mapping from domains to final rewards
        :type domainRewards: dict
        :returns: None
        '''
        for dstring in self.domainPolicies:
            if self.domainPolicies[dstring] is not None:
                domains_reward = domainRewards[dstring]
                if domains_reward is not None:
                    if self.committees[dstring] is not None:
                        self.committees[dstring].finalizeRecord(domains_reward,dstring)
                    elif self.domainPolicies[dstring] is not None:
                        self.domainPolicies[dstring].finalizeRecord(domains_reward,dstring)
                else:
                    logger.warning("Final reward in domain: "+dstring+" is None - Should mean domain wasnt used in dialog")
                    
    def getLastSystemAction(self, domainString):
        '''
        Returns the last system action of the specified domain.
        
        :param domainString: the domain string unique identifier.
        :type domainString: str
        :returns: the last system action of the given domain or None
        '''
        if self.domainPolicies[domainString] is not None:
            return self.domainPolicies[domainString].lastSystemAction
        return None
            
    def restart(self):
        '''
        Restarts all policies of all domains and resets internal variables.
        '''
#         self.lastSystemAction = None
#         self.prevbelief = None
        
        for dstring in self.domainPolicies.keys():
            if self.domainPolicies[dstring] is not None: 
                self.domainPolicies[dstring].restart()
        return
    
    def printEpisodes(self):
        '''
        Prints the recorded episode of the current dialogue. 
        '''
        for dString in self.domainPolicies:
            if self.domainPolicies[dString] is not None:
                print "---------- Episodes for domain {}".format(dString)
                for domain in self.domainPolicies[dString].episodes:
                    if self.domainPolicies[dString].episodes[domain] is not None:
                        print domain
                        self.domainPolicies[dString].episodes[domain].tostring()
        
    def _load_domains_policy(self, domainString=None):
        '''
        Loads and instantiates the respective policy as configured in config file. The new object is added to the internal
        dictionary. 
        
        Default is 'hdc'.
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString, learning
        
        :param domainString: the domain the policy will work on. Default is None.
        :type domainString: str
        :returns: the new policy object
        '''
        
        # 1. get type:
        policy_type = 'hdc'  # domain+resource independent default
        in_policy_file = ''
        out_policy_file = ''
        learning = False
        useconfreq = False
        
        
        if not Settings.config.has_section('policy_'+domainString):
            if not Settings.config.has_section('policy'):
                logger.warning("No policy section specified for domain: "+domainString+" - defaulting to HDC")
            else:
                logger.info("No policy section specified for domain: " + domainString + " - using values from 'policy' section")
        if Settings.config.has_option('policy', 'policytype'):
            policy_type = Settings.config.get('policy', 'policytype')
        if Settings.config.has_option('policy', 'learning'):
            learning = Settings.config.getboolean('policy', 'learning')
        if Settings.config.has_option('policy', 'useconfreq'):
            useconfreq = Settings.config.getboolean('policy', 'useconfreq')
        if Settings.config.has_option('policy', 'inpolicyfile'):
            in_policy_file = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy', 'outpolicyfile'):
            out_policy_file = Settings.config.get('policy', 'outpolicyfile')

        if Settings.config.has_option('policy_'+domainString, 'policytype'):
            policy_type = Settings.config.get('policy_'+domainString, 'policytype')
        if Settings.config.has_option('policy_'+domainString, 'learning'):
            learning = Settings.config.getboolean('policy_'+domainString, 'learning')
        if Settings.config.has_option('policy_'+domainString, 'useconfreq'):
            useconfreq = Settings.config.getboolean('policy_'+domainString, 'useconfreq')
        if Settings.config.has_option('policy_'+domainString, 'inpolicyfile'):
            in_policy_file = Settings.config.get('policy_'+domainString, 'inpolicyfile')
        if Settings.config.has_option('policy_'+domainString, 'outpolicyfile'):
            out_policy_file = Settings.config.get('policy_'+domainString, 'outpolicyfile')

        if domainString in self.SPECIAL_DOMAINS:
            if domainString == 'topicmanager':
                policy_type = 'hdc_topicmanager'
                from policy import HDCTopicManager
                self.domainPolicies[domainString] = HDCTopicManager.HDCTopicManagerPolicy()
            elif domainString == "wikipedia":
                policy_type = 'hdc_wikipedia'
                import WikipediaTools 
                self.domainPolicies[domainString] = WikipediaTools.WikipediaDM()         
        else:            
            if policy_type == 'hdc':
                from policy import HDCPolicy
                self.domainPolicies[domainString] = HDCPolicy.HDCPolicy(domainString)
            elif policy_type == 'gp':
                from policy import GPPolicy
                self.domainPolicies[domainString] = GPPolicy.GPPolicy(domainString, learning, self.shared_params)
            elif policy_type == 'dipgp':
                from policy import DIPGPPolicy
                self.domainPolicies[domainString] = DIPGPPolicy.GPPolicy(domainString, learning, self.shared_params)
            elif policy_type == 'dqn':
                from policy import DQNPolicy
                self.domainPolicies[domainString] = DQNPolicy.DQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'a2c':
                from policy import A2CPolicy
                self.domainPolicies[domainString] = A2CPolicy.A2CPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'enac':
                from policy import ENACPolicy
                self.domainPolicies[domainString] = ENACPolicy.ENACPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'bdqn':
                from policy import BDQNPolicy
                self.domainPolicies[domainString] = BDQNPolicy.BDQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'acer':
                from policy import ACERPolicy
                self.domainPolicies[domainString] = ACERPolicy.ACERPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'tracer':
                from policy import TRACERPolicy
                self.domainPolicies[domainString] = TRACERPolicy.TRACERPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'bbqn':
                from policy import BBQNPolicy
                self.domainPolicies[domainString] = BBQNPolicy.BBQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'feudal':
                from policy import FeudalPolicy
                self.domainPolicies[domainString] = FeudalPolicy.FeudalPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'feudalAC':
                from policy import FeudalACPolicy
                self.domainPolicies[domainString] = FeudalACPolicy.FeudalACPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'morl':
                from policy import MORLPolicy
                self.domainPolicies[domainString] = MORLPolicy.MORLPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'roi-morl':
                from policy import RoiMORLPolicy
                self.domainPolicies[domainString] = RoiMORLPolicy.RoiMORLPolicy(in_policy_file, out_policy_file, domainString, learning)
            else:
                try:
                    # try to view the config string as a complete module path to the class to be instantiated
                    components = policy_type.split('.')
                    packageString = '.'.join(components[:-1]) 
                    classString = components[-1]
                    mod = __import__(packageString, fromlist=[classString])
                    klass = getattr(mod, classString)
                    self.domainPolicies[domainString] = klass(domainString,learning)
                except ImportError as e:
                    logger.error('Invalid policy type "{}" for domain "{}" raising error {}'.format(policy_type, domainString, e))
                    
            #------------------------------ 
            # TODO - Not currently implemented as we aren't currently using these policy types 
#             elif True:
#                 exit('NOT IMPLEMENTED... see msg at this point in code')
#             elif policy_type == 'type':
#                 from policy import TypePolicy
#                 policy = TypePolicy.TypePolicy()
#             elif policy_type == 'select':
#                 from policy import SelectPolicy
#                 policy = SelectPolicy.SelectPolicy(use_confreq=useconfreq)
#             elif policy_type == 'nn':
#                 from policy import NNPolicy
#                 # TODO - further change here - train is now implmented in config file. below needs updating 
#                 policy = NNPolicy.NNPolicy(use_confreq=useconfreq, is_training=train)           
            #------------------------------
        return
    
    def _load_committees(self):
        '''
        Loads and instantiates the committee as configured in config file. The new object is added to the internal
        dictionary. 
        '''
        committees = dict.fromkeys(OntologyUtils.available_domains, None)
        useBCM = False
        learningMethod = "singleagent"
        
        if Settings.config.has_option("policycommittee","bcm"):
            useBCM = Settings.config.getboolean("policycommittee","bcm")
        
        if not useBCM:    
            return committees # return an empty committee dict to indicate that committees are not used
        
        from policy import PolicyCommittee
        if Settings.config.has_option("policycommittee","learningmethod"):
            learningMethod = Settings.config.get("policycommittee","learningmethod")
         
        if Settings.config.has_option("policycommittee","pctype"):
            pcType =  Settings.config.get("policycommittee","pctype")
        if pcType == 'hdc':
            # handcrafted committees are a bit strange, I think they should be removed
            predefinedCommittees = {}
            predefinedCommittees['Electronics'] = ['Laptops6','Laptops11','TV']
            predefinedCommittees['SF'] = ['SFHotels','SFRestaurants']
            
            for key in predefinedCommittees:
                committee = PolicyCommittee.PolicyCommittee(self,predefinedCommittees[key],learningMethod)
                self._check_committee(committee)
                
                for domain in predefinedCommittees[key]:
                    committees[domain] = committee
            
            
        elif pcType == 'configset':
            # TODO extend settings to allow multiple committees
            try:
                committeeMembers = Settings.config.get('policycommittee', 'configsetcommittee')
            except Exception as e: #ConfigParser.NoOptionError:  # can import ConfigParser if you wish 
                print e
                logger.error('When using the configset committee - you need to set configsetcommittee in the config file.')
                
            committeeMembers = committeeMembers.split(',')
            committee = PolicyCommittee.PolicyCommittee(self,committeeMembers,learningMethod)
            self._check_committee(committee)
            
            for domain in committeeMembers:
                committees[domain] = committee

        else:
            logger.error("Unknown policy committee type %s" % pcType)            
        
        return committees
    
    def _check_committee(self,committee):
        '''
        Safety tool - should check some logical requirements on the list of domains given by the config
        
        :param committee: the committee be be checked
        :type committee: :class:`~policy.PolicyCommittee.PolicyCommittee`
        '''
        committeeMembers = committee.members
        
        if len(committeeMembers) < 2:
            logger.warning('Only 1 domain given')
        
        # Ensure required objects are available to committee (ontology for domains etc)
        for dstring in committeeMembers:
            Ontology.global_ontology.ensure_domain_ontology_loaded(dstring)
        
        
        # TODO - should check that domain tags given are valid according to OntologyUtils.py
        return
    