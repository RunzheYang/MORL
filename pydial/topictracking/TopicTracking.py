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
TopicTracking.py -  Topic tracking interface  
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils` |.|

************************

'''


__author__ = "cued_dialogue_systems_group"
from utils import Settings, ContextLogger
from ontology import OntologyUtils
logger = ContextLogger.getLogger('')


class TopicTrackingManager(object):
    '''
    Interface for all topic tracking
    '''
    def __init__(self):
        self.topic_tracker_type = None  # TopicTracker may not be needed in simulate, 
        if Settings.config.has_option("topictracker","type"):
            self.topic_tracker_type = Settings.config.get("topictracker","type")
        else:
            self.topic_tracker_type = "keyword"  # set it as a default

        if self.topic_tracker_type == "keyword":
            from RuleTopicTrackers import KeywordSpottingTopicTracker
            self.tt_model = KeywordSpottingTopicTracker()
        elif self.topic_tracker_type == 'switch':
            from RuleTopicTrackers import TextBasedSwitchTopicTracker
            self.tt_model = TextBasedSwitchTopicTracker() 
        elif self.topic_tracker_type != None:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = self.topic_tracker_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.tt_model = klass()
            except ImportError:
                logger.error('Unknown topic tracker "{}"'.format(self.topic_tracker_type))
        else:
            self.tt_model = None  # simulate for example, may just "cheat"
            
        # SINGLE DOMAIN SYSTEM:
        self.USE_SINGLE_DOMAIN = False
        if Settings.config.has_option("GENERAL","singledomain"):
            self.USE_SINGLE_DOMAIN = Settings.config.getboolean("GENERAL","singledomain")
        if self.USE_SINGLE_DOMAIN:
            # will now be restricted to a single doman system:
            # Below Settings.config.get is safe -- all hubs have already checked config has this section
            domain = Settings.config.get("GENERAL","domains")  
            if "," in domain:
                logger.error("It's confusing to specify more than 1 domain if you want to use single domain system")
            if domain not in OntologyUtils.available_domains:
                logger.error("Domain: "+domain+" not an available domain tag")
            else:
                self.singleDomain = domain
        else:
            # Maintain record of systems last acts in each domain, to facilitate returning to domain at later turns
            self.domainsLastAct = dict.fromkeys(OntologyUtils.available_domains, None)  # NB: is for multidomain only


    def restart(self):
        '''
        # Start with control where? Single domain --> with its manager, Multi-Domain --> with topicmanager
        '''
        # restart and set operatingDomain 
        if self.tt_model is not None:
            self.tt_model.restart()
        if self.USE_SINGLE_DOMAIN:
            self.operatingDomain = self.singleDomain
        else:
            self.operatingDomain = 'topicmanager'       # control starts with this in the multi-domain system.
        self.previousDomain = None  # no history in dialogue yet
        self.in_present_dialog = [self.operatingDomain]  # record domains involved in present dialog.
        return

    def track_topic(self, userAct_hyps=None, domainString=None): 
        ''' # CURRENT ASSUMPTION: ONLY ONE DOMAIN CAN BE ACTIVE IN EACH TURN.
        Sets member variables: operatingDomain
        '''
        tt_domain = None
        # 1. Track topic:
        if self.USE_SINGLE_DOMAIN:
            tt_domain = self.operatingDomain   #  i.e nothing to do
        elif domainString is not None and "ood" != domainString:
            tt_domain = domainString
        elif self.tt_model is not None:
            if userAct_hyps is None:
                logger.error("Using topic tracker: "+self.topic_tracker_type+", but no information for inference passed")
            else:
                tt_domain = self.tt_model.infer_domain(userActHyps=userAct_hyps)  
        else:
            logger.error("Must directly pass domain name or information for inference")
            
        # for ood handling identified outside of pydial, eg dialport
        if tt_domain is None and "ood" == domainString:
            tt_domain = domainString
            
        # 2. Update current and previous domains
        self.previousDomain = None  # indicates we did not change domain [default assumption - will be changed below if we did]
        if tt_domain is not None:
            if isinstance(tt_domain, str) or isinstance(tt_domain, unicode):
                if tt_domain != self.operatingDomain:
                    self.previousDomain = self.operatingDomain 
                    self.operatingDomain = tt_domain
                    logger.info('TopicTracker believes we switched domains. From %s to %s' % (self.previousDomain,self.operatingDomain))
            else:
                # NOTE: -- ASSUMED ABOVE THAT self.tt_output IS A SINGLE DOMAIN TAG (str) --> adjust when proper trackers 
                #  (ie ones giving distribution over domain labels for example) are added. 
                logger.error('new topic tracker -- needs integrating: {}; {}; {}'.format(tt_domain, isinstance(tt_domain, str), type(tt_domain)))
        # else failed to recognise anything - refrain to old value, ie, do nothing
        logger.info('After user_act - domain is now: '+self.operatingDomain)
        
        return self.operatingDomain

    


# END OF FILE
