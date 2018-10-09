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
HDCTopicManager.py - policy for the front end topic manager
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

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
class HDCTopicManagerPolicy(Policy.Policy):
    """
    The dialogue while being in the process of finding the topic/domain of the conversation.
    
    At the current stage, this only happens at the beginning of the dialogue, so this policy has to take care of wecoming the user as well as creating actions which disambiguate/clarify the topic of the interaction.
    
    It allows for the system to hang up if the topic could not be identified after a specified amount of attempts.
    """
    def __init__(self):
        super(HDCTopicManagerPolicy,self).__init__(domainString="topicmanager",specialDomain=True)
        
        self.TIMES_CALLED = 0
        self.QUIT_AFTER_NUM_FAILED_TURNS = None 
        
        self.startwithhello=False
        
        # separate control for ending dialog based on topic tracker failing   
        if Settings.config.has_option("policy_topicmanager","maxattempts"):
            self.QUIT_AFTER_NUM_FAILED_TURNS = Settings.config.getint("policy_topicmanager","maxattempts")
        
        
    def nextAction(self, beliefstate):
        """
        In case system takes first turn - Topic manager will just greet the user
        Note hyps are ASR  
        """
        self.TIMES_CALLED += 1
        return self._conditional_response(beliefstate)

    def _conditional_response(self, beliefstate):
        '''Note that self.QUIT_AFTER_NUM_FAILED_TURNS is only used if given by config 
        '''
        if self._check_if_user_said_bye(beliefstate):
            return 'bye(topictrackeruserended)'
        if self.TIMES_CALLED > self.QUIT_AFTER_NUM_FAILED_TURNS and self.QUIT_AFTER_NUM_FAILED_TURNS is not None:
            return 'bye(toptictrackertimedout)'
        if self.TIMES_CALLED > 1:
            return 'hello(help)'
        return 'hello()'

    def _check_if_user_said_bye(self, beliefstate):
        '''Checks if the user said goodbye based on discourseAct in belief state 
        '''
        if beliefstate is not None:
            if 'bye' in beliefstate["beliefs"]["discourseAct"]:
                if beliefstate["beliefs"]["discourseAct"]['bye'] > 0.8:
                    return True
        return False

    def restart(self):
        self.TIMES_CALLED = 0


#END OF FILE