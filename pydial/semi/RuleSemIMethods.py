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
RuleSemIMethods.py - Semantic input parsers based on rules
============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :class:`semi.RegexSemI_generic.RegexSemI_generic` |.|

Collection of Semantic Parsers that are rule based. 

************************

'''


__author__ = "cued_dialogue_systems_group"
import os
from RegexSemI_generic import RegexSemI_generic
from utils import ContextLogger
logger = ContextLogger.getLogger('')

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

class PassthroughSemI(object):
    '''**Does nothing** - simply pass observation directly through (on assumption that observation was a 
    semantic act input by a texthub user) -- hence is domain independent and doesn't need a manager
    '''
    # Added some optional arguments that are only needed by DLSemI, here they are just for consistency
    def decode(self, obs, sys_act=None,turn=None):
        '''
        :param obs: the list of observations
        :type obs: str
        :returns: (str) **EXACT** obs as was input
        '''
        return obs


class RegexSemI(object):
    """
    A semantic parser based on regular expressions. One parser for each domain is necessary. To implement a new 
    regex parser, derive from semi.RegexSemI.RegexSemI and create your own module called "RegexSemI_<yourdomainname>"
    containing the class "RegexSemI_<yourdomainname>".
    """
    def __init__(self,domainTag):
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia','ood']
        self.domain_tag = domainTag
        if domainTag not in self.SPECIAL_DOMAINS:
#             sys.path.append(Settings.root+"semi/")
            try:
                parser_module = __import__("semi.RegexSemI_"+self.domain_tag, fromlist=["RegexSemI_"+self.domain_tag]) 
                self.parser = getattr(parser_module, "RegexSemI_"+self.domain_tag)()
            except ImportError:
                logger.warning("No suitable regex SemI module found. Defaulting to generic module.")
                self.parser = RegexSemI_generic(self.domain_tag)
        else:
            from RegexSemI import RegexSemI
            self.parser = RegexSemI()

    # Again, added some optional arguments that are only needed by DLSemI, here they are just for consistency
    def decode(self, obs, sys_act=None, turn=None):
        """
        """
        return self.parser.decode(obs, sys_act)


# END OF FILE