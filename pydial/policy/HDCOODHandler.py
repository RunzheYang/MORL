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
HDCOODTopicManager.py - policy for detecting out of domain inputs
=================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Module specifies behaviour for special domain "ood".

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.Policy` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` 

************************

'''

__author__ = "cued_dialogue_systems_group"
from utils import Settings
from utils import ContextLogger
import Policy
logger = ContextLogger.getLogger('')

#TODO - add ability to say 'bye' when control is with the topic manager .. 
class HDCOODHandlerPolicy(Policy.Policy):
    """
    The dialogue if an out of domain has been found.
    
    At the current stage, this only happens when using a topic tracker outside of pydial in comabination with the dialogue server.
    
    The first time ood is detected, a prompt is generated asking the user to return to the original topic. If that does not help, control is returned to topic manager.
    """
    def __init__(self, originalDomain):
        super(HDCOODHandlerPolicy,self).__init__(domainString="ood",specialDomain=True)
        
        self.times_called = 0
        self.max_attempts = 1
        self.originalDomain = originalDomain
        
        # separate control for ending dialog based on topic tracker failing   
        if Settings.config.has_option("policy_ood","maxoods"):
            self.max_attempts = Settings.config.getint("policy_topicmanager","maxoods")
        
        
    def nextAction(self, beliefstate):
        """
        In case system takes first turn - Topic manager will just greet the user
        Note hyps are ASR  
        """
        self.times_called += 1
        return self._conditional_response(beliefstate)

    def _conditional_response(self, beliefstate):
        '''Note that self.max_attempts is only used if given by config 
        '''
        if self._check_if_user_said_bye(beliefstate):
            return 'bye(oodhandleruserended)'
        if self.times_called > self.max_attempts and self.max_attempts is not None:
            return 'bye(maxoods)'
        return 'ood(domain={})'.format(self.originalDomain)

    def _check_if_user_said_bye(self, beliefstate):
        '''Checks if the user said goodbye based on discourseAct in belief state 
        '''
        if len(beliefstate["beliefs"]):
            if 'bye' in beliefstate["beliefs"]["discourseAct"]:
                if beliefstate["beliefs"]["discourseAct"]['bye'] > 0.8:
                    return True
        return False

    def restart(self):
        self.times_called = 0


#END OF FILE