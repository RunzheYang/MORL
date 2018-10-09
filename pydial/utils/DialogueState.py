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
DialogueState.py - dialogue state object specification
===========================================================

Copyright CUED Dialogue Systems Group 2017

**Basic Usage**: 
    >>> import DialogueState   
   
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|

************************

'''

from utils.ContextLogger import ContextLogger
import pprint as pp
logger = ContextLogger()


class DialogueState(object):
    '''
    The encapsulation of the dialogue state with a definition of the main interface.
    '''


    def __init__(self):
        self.domainStates = {}
        self.lastSystemAct = {}
        self.currentdomain = None
        
    def getDomainState(self, dstring):
        '''
        Access to the dialogue state (belief state) of the specified domain. Returns None if there is no state yet.
        
        :param dstring: the string identifier of the domain the state should be retreived from
        :type dstring: str
        
        :returns: the state dict or None
        '''
        if dstring in self.domainStates:
            return self.domainStates[dstring]
        return None
    
    def printUserActs(self, dstring):
        '''
        Utility function to print the user acts stored in the belief state of domain dstring.
        
        :param dstring: the string identifier of the domain of which the user act should be printed 
        :type dstring: str
        '''
        if self.domainStates is not None and 'userActs' in self.domainStates[dstring]:
            print '   Usr > {}'.format(self.domainStates[dstring]['userActs'])
            
    def check_user_ending(self):
        '''
        Utility function to check whether the user has said good bye.
        
        '''
        for domain in self.domainStates:
            belief = self.domainStates[domain]
            if len(belief["beliefs"]):
                if 'bye' in belief["beliefs"]["discourseAct"]:
                    if belief["beliefs"]["discourseAct"]['bye'] > 0.8:
                        return True             
        return False
    
    def setLastSystemAct(self, sysAct):
        '''
        Sets the last system act of the current domain. Note that currentdomain needs to be set first, otherwise it does not work.
        
        :param sysAct: string representation of the last system action
        :type sysAct: str
        '''
        if self.currentdomain is not None:
            self.lastSystemAct[self.currentdomain] = sysAct
        else:
            logger.error('Attempt to store last system action for unknown domain.')
            
    def getLastSystemAct(self, dstring):
        '''
        Retreives the last system act of domain dstring.
        
        :param dstring: the string identifier of the domain of which the last system act should be retreived from
        :type dstring: str
        
        :returns: the last system act of domain dstring or None
        '''
        if dstring in self.lastSystemAct:
            return self.lastSystemAct[dstring]
        else:
            return None


    def __str__(self):
        if self.currentdomain is not None:
            pp.pprint(self.getDomainState(self.currentdomain))


            
    
        
    
        
