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
Wizard.py - a wizard-of-oz implementation based on summary actions
======================================================================================

Copyright CUED Dialogue Systems Group 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.Policy` |.|

************************

'''

from policy import Policy

class Wizard(Policy.Policy):
    '''
    A wizard-of-oz implementation using the terminal which selects possible responses based on summary actions.
    '''
    
    def __init__(self, domainString, learning):
        super(Wizard, self).__init__(domainString,learning=learning)
        
        self.alwaysShowAllActions = True
 
    def nextAction(self, belief):
        nonExecutableActions = self.actions.getNonExecutable(belief, self.lastSystemAction)
        executable = self._createExecutable(nonExecutableActions)
        
        if self.alwaysShowAllActions:
            actionList = self.actions.action_names
        else:
            actionList = executable
        
        print ""
        for i in range(0,len(actionList)):
#         for i in range(0,len(executable)):
            action = actionList[i]
            if action in executable:
                print "{:2d}: {:<35} --> {}".format(i+1, action, self.actions.Convert(belief, action, self.lastSystemAction))
            else:
                print " -  {:<35}".format(action)
        print ""
        
        actionNum = raw_input("Enter the number of the action: ")
        summaryAct = actionList[int(actionNum)-1]
        masterAct = self.actions.Convert(belief, summaryAct, self.lastSystemAction)
        
        return masterAct
        
        
    def _createExecutable(self,nonExecutableActions):
        '''
        Produce a list of executable actions from non executable actions
        
        :param nonExecutableActions:
        :type nonExecutableActions:
        '''                
        executable_actions = []
        for act_i in self.actions.action_names:
            if act_i in nonExecutableActions:
                continue
            else:
                executable_actions.append(act_i)
        return executable_actions
    
#END OF FILE