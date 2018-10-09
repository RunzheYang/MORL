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
UMSimulator.py - Interface for simulated user behaviour 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

'''

__author__ = "cued_dialogue_systems_group"

class UMSimulator(object):
    '''
    Interface class for a single domain user model decision simulator. Responsible for selecting the next user action.
    
    To create your own simulator, derive from this class.
    '''
    def __init__(self, domainString, max_patience = 5):
        '''
        Constructor for domains policy.
        :param domainString: domain tag
        :type domainString: str
        :param max_patience: the max patience of for this user model. If patience runs out, the user hangs up.
        :type max_patience: int
        '''
        self.dstring = domainString
        
        # DEFAULTS:
        self.max_patience = max_patience
        
    def init(self, goal, um_patience):
        '''
        Initializes the simulator. 
        
        This method is automatically invoked by the init method of the user model.
        
        It needs to be implemented in a sub-class.
        
        :param goal: the user goal
        :type goal: UMGoal
        :param um_patience: the max patience for this simulation run.
        :type um_patience: int
        '''
        pass
    
    def receive(self, sys_act, goal):
        '''
        This method processes the new input system act and updates the agenda.
        
        It needs to be implemented in a sub-class.
        
        :param sys_act: the max patience for this simulation run.
        :type sys_act: :class:`DiaAct.DiaAct`
        :param goal: the user goal
        :type goal: :class:`UserModel.UMGoal`
        '''
        pass
    
    def respond(self, goal):
        '''
        This method is called to get the user response.

        :param goal: of :class:`UserModel.UMGoal`
        :type goal: :class:`UserModel.UMGoal`
        :returns: (instance) of :class:`DiaAct.DiaActWithProb`
        '''
        pass