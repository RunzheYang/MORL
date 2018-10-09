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
SemOManager.py - Semantic Output
==================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

from utils import Settings, ContextLogger
from ontology import OntologyUtils
logger = ContextLogger.getLogger('')


class SemO(object):
    """
    Interface class for a single domain langauge generator. Responsible for generating a natural langauge sentence from a dialogue act representation. 
    To create your own SemO methods, derive from this class.
    """
    def generate(self, act):
        """
        Main generation method: mapping from system act to natural language
        :param act: the system act to generate
        :type act: str
        :returns: the natural language realisation of the given system act
        """
        pass
        #return self.semo_method.generate(act)
        



class SemOManager(SemO):
    """SemO manager for each domain. Independently for each domain you can load a generator.
    Implementations of actual generators are in other modules
    """
    def __init__(self): 
        '''
        Constructor for the SemOManager
        '''
        self.semoManagers = dict.fromkeys(OntologyUtils.available_domains)

    def _load_domains_semo(self, dstring):
        """Get from the config file the SemO choice of method for this domain
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        """
        # 1. get semo method for the domain
        semo_type = 'PassthroughSemO'  # default, will work in all domains
        if Settings.config.has_option('semo_'+dstring, 'semotype'):
            semo_type = Settings.config.get('semo_'+dstring, 'semotype')
            if semo_type == 'BasicSemO': # then check there is a template file -- will be read in BasicSemO class
                if not Settings.config.has_option('semo_'+dstring,'templatefile'):
                    logger.warning("You must specify a template file if using BasicSemO - defaulting back to PassthroughSemO")
                    semo_type = 'PassthroughSemO'
            elif semo_type == 'RNNSemO':
                # RNNSemO, check whether config file exist
                if not Settings.config.has_option('semo_'+dstring,'configfile'):
                    logger.warning("You must specify a config file if using RNNSemO - defaulting back to PassthroughSemO")
                    semo_type = 'PassthroughSemO'
        
        # 2. And load that method for the domain:
        if semo_type == 'PassthroughSemO':
            from RuleSemOMethods import PassthroughSemO
            self.semoManagers[dstring] = PassthroughSemO()
        elif semo_type == 'BasicSemO':
            if dstring == 'topicmanager':
                from RuleSemOMethods import TopicManagerBasicSemO
                self.semoManagers[dstring] = TopicManagerBasicSemO(domainTag=dstring)
            else:
                from RuleSemOMethods import BasicSemO
                self.semoManagers[dstring] = BasicSemO(domainTag=dstring)
        elif semo_type == 'RNNSemO':
            ### ADDED By SHAWN Oct 12, 2016
            if dstring == 'topicmanager':
                from RuleSemOMethods import TopicManagerBasicSemO
                self.semoManagers[dstring] = TopicManagerBasicSemO(domainTag=dstring)
            else:
                from RNNSemOMethods import RNNSemO
                self.semoManagers[dstring] = RNNSemO(domainTag=dstring)
            ###############################
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = semo_type.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                self.semoManagers[dstring] = klass(dstring)
            except ImportError:
                logger.warning('Unknown output generator "{}" for domain "{}". Using PassthroughSemO.'.format(semo_type, dstring))
                from RuleSemOMethods import PassthroughSemO
                self.semoManagers[dstring] = PassthroughSemO()

    def _ensure_booted(self, domainTag):
        """
        The function to ensure the given domain generator is properly loaded
         :param domainTag: the domain string unique identifier, the domain you operate on.
        """
        if self.semoManagers[domainTag] is None:
            self._load_domains_semo(dstring=domainTag)
        return 
    
    def generate(self, act, domainTag=None):
        """
        Main generation method which maps the given system act into natural langauge realisation. 
        :param domainTag: the domain string unique identifier, the domain you operate on.
        :act: the system act you want to generate
        """
        
        act_s = act.to_string()
        
        if domainTag is None:
            logger.warning('Not sure which domain to generate in.')
            return act_s
        else:
            self._ensure_booted(domainTag) 
        return self.semoManagers[domainTag].generate(act_s)



#END OF FILE
