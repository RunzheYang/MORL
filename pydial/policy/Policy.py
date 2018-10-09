###############################################################################
# CU-PyDial: Multi-domain Statistical Spoken Dialogue System Software
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
Policy.py - abstract class for all policies
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`policy.SummaryAction`

************************
'''

__author__ = "cued_dialogue_systems_group"
from utils import Settings, ContextLogger, DiaAct
from ontology import OntologyUtils
from copy import deepcopy
import SummaryAction
logger = ContextLogger.getLogger('')

class Policy(object):
    '''
    Interface class for a single domain policy. Responsible for selecting the next system action and handling the learning of the policy.
    
    To create your own policy model or to change the state representation, derive from this class.
    '''
    def __init__(self, domainString, learning=False, specialDomain=False): 
        """
        Constructor for domains policy.
        :param domainString: domain tag
        :type domainString: str
        :return:
        """
        
        self.summaryAct = None
        self.actToBeRecorded = None    
        self.lastSystemAction = None # accessed from outside of policy
        self.prevbelief = None
        
        self.learning = learning
        self.domainString = domainString
        
        self.startwithhello = False
        if Settings.config.has_option('policy', 'startwithhello'):
            self.startwithhello = Settings.config.getboolean('policy', 'startwithhello')
        if Settings.config.has_option('policy_'+domainString, 'startwithhello'):
            self.startwithhello = Settings.config.getboolean('policy_'+domainString, 'startwithhello')
        
        self.useconfreq = False
        if Settings.config.has_option('policy', 'useconfreq'):
            self.useconfreq = Settings.config.getboolean('policy', 'useconfreq')
        if Settings.config.has_option('policy_'+domainString, 'useconfreq'):
            self.useconfreq = Settings.config.getboolean('policy_'+domainString, 'useconfreq')
        
        # episode information to be collected for all relevant domains
        # used mostly for training
        self.episode_stack = None
        self.USE_STACK = False
        self.PROCESS_EPISODE_STACK = 0 # and process them whenever stack gets this high. 
        if Settings.config.has_option("policy", "usestack"):
            self.USE_STACK = Settings.config.getboolean("policy", "usestack")
        if Settings.config.has_option("policy_"+domainString, "usestack"):
            self.USE_STACK = Settings.config.getboolean("policy_"+domainString, "usestack")
        if self.USE_STACK:
            # if we store any episodes (and dont strictly stay on policy with SARSA) - we store them here.
            self.episode_stack = EpisodeStack()  
            # and process them in sequential batches of size: 
            self.PROCESS_EPISODE_STACK = 5   
            if Settings.config.has_option("policy", "processstack"):
                self.PROCESS_EPISODE_STACK = Settings.config.getint("policy_"+domainString, "processstack")
            if Settings.config.has_option("policy", "processstack"):
                self.PROCESS_EPISODE_STACK = Settings.config.getint("policy_"+domainString, "processstack")
        
        empty = specialDomain
        # action information are all maintained in a class SummaryAction.SummaryAction
        
        self.actions = SummaryAction.SummaryAction(domainString, empty, self.useconfreq)
        # Total number of system actions.
        self.numActions = len(self.actions.action_names)
        
        self.episodes = dict.fromkeys(OntologyUtils.available_domains, None)
        self.episodes[self.domainString] = Episode(self.domainString)
        
    def act_on(self, state, preference=None):
        '''
        Main policy method: mapping of belief state to system action.
        
        This method is automatically invoked by the agent at each turn after tracking the belief state.
        
        May initially return 'hello()' as hardcoded action. Keeps track of last system action and last belief state.  
        
        :param state: the belief state to act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :param hyps: n-best-list of semantic interpretations
        :type hyps: list
        :returns: the next system action of type :class:`~utils.DiaAct.DiaAct`
        '''
        beliefstate = state.getDomainState(self.domainString)
        
        if self.lastSystemAction is None and self.startwithhello:
            _systemAct = 'hello()'
        else:
            _systemAct = self.nextAction(beliefstate, preference)
        self.lastSystemAction = _systemAct
        self.prevbelief = beliefstate
        
        systemAct = DiaAct.DiaAct(_systemAct)
        return systemAct
    
    def record(self, reward, domainInControl = None, weight = None, state=None, action=None):
        '''
        Records the current turn reward along with the last system action and belief state.
        
        This method is automatically executed by the agent at the end of each turn.
        
        To change the type of state/action override :func:`~convertStateAction`. By default, the last master action is recorded. 
        If you want to have another action being recorded, eg., summary action, assign the respective object to self.actToBeRecorded in a derived class. 
        
        :param reward: the turn reward to be recorded
        :type reward: np.array
        :param domainInControl: the domain string unique identifier of the domain the reward originates in
        :type domainInControl: str
        :param weight: used by committee: the weight of the reward in case of multiagent learning
        :type weight: float
        :param state: used by committee: the belief state to be recorded
        :type state: dict
        :param action: used by committee: the action to be recorded
        :type action: str
        :returns: None
        '''
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            self.episodes[domainInControl] = Episode(dstring=domainInControl)
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.lastSystemAction
            
        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded
            
        cState, cAction = self.convertStateAction(state, action)
        
        if weight == None:
            self.episodes[domainInControl].record(state=cState, action=cAction, reward=reward)
        else:
            self.episodes[domainInControl].record(state=cState, action=cAction, reward=reward, ma_weight = weight)
            
        self.actToBeRecorded = None
        return
    
    def finalizeRecord(self, reward, domainInControl = None):
        '''
        Records the final reward along with the terminal system action and terminal state. To change the type of state/action override :func:`~convertStateAction`.
        
        This method is automatically executed by the agent at the end of each dialogue.
        
        :param reward: the final reward
        :type reward: np.array
        :param domainInControl: used by committee: the unique identifier domain string of the domain this dialogue originates in, optional
        :type domainInControl: str
        :returns: None
        '''
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return
        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())        
        self.episodes[domainInControl].record(state=terminal_state, action=terminal_action, reward=reward)
        return
    
    def convertStateAction(self, state, action):
        '''
        Converts the given state and action to policy-specific representations. 
        
        By default, the generic classes :class:`~State` and :class:`~Action` are used. To change this, override method in sub-class.
        
        :param state: the state to be encapsulated
        :type state: anything
        :param action: the action to be encapsulated
        :type: action: anything 
        '''
        return State(state), Action(action)
      
#########################################################
# interface methods
#########################################################    
    
    def nextAction(self, beliefstate, preference=None):
        '''
        Interface method for selecting the next system action. Should be overridden by sub-class.
        
        This method is automatically executed by :func:`~act_on` thus at each turn.
        
        :param beliefstate: the state the policy acts on
        :type beliefstate: dict
        :returns: the next system action
        '''
        pass
    
    def train(self):
        '''
        Interface method for initiating the training. Should be overridden by sub-class. 
        
        This method is automatically executed by the agent at the end of each dialogue if learning is True.
        
        This method is called at the end of each dialogue by :class:`~policy.PolicyManager.PolicyManager` if learning is enabled for the given domain policy.
        '''
        pass
    
    def savePolicy(self, FORCE_SAVE=False):
        '''
        Saves the learned policy model to file. Should be overridden by sub-class.
        
        This method is automatically executed by the agent either at certain intervals or at least before shutting down the agent.
        
        :param FORCE_SAVE: used to force cleaning up of any learning and saving when we are powering off an agent.
        :type FORCE_SAVE: bool
        '''
        pass
    
    def restart(self):
        '''
        Restarts the policy. Resets internal variables.
        
        This method is automatically executed by the agent at the end/beginning of each dialogue.
        '''
        self.summaryAct = None          
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        
        self.episodes = dict.fromkeys(OntologyUtils.available_domains, None)
        self.episodes[self.domainString] = Episode(dstring=self.domainString)

        self.actions.reset() # ic340: this should be called from every restart impelmentation


#########################################################
# Episode classes
#########################################################

class Episode(object):
    '''
    An episode encapsulates the state-action-reward triplet which may be used for learning. Every entry represents one turn. 
    The last entry should contain :class:`~TerminalState` and :class:`~TerminalAction`  
    '''
    def __init__(self, dstring=None):
        self.strace = []
        self.atrace = []
        self.rtrace = []
        self.totalreward = 0
        self.totalMAweight = 0
        self.learning_from_domain = dstring
    
    def record(self, state, action, reward, ma_weight = None):
        '''
        Stores the state action reward in internal lists.
        
        :param state: the last belief state
        :type state: :class:`~State`
        :param action: the last system action
        :type action: :class:`~Action`
        :param reward: the reward of the last turn
        :type reward: int
        :param ma_weight: used by committee: the weight assigned by multiagent learning, optional
        :type ma_weight: float
        '''
        self.totalreward += reward
        
        if ma_weight is not None:
            self.totalMAweight += ma_weight

        self.strace.append(state)
        self.atrace.append(action)
        self.rtrace.append(reward)

    def check(self):
        '''
        Checks whether length of internal state action and reward lists are equal.
        '''
        assert(len(self.strace)==len(self.atrace))
        assert(len(self.strace)==len(self.rtrace))
        
    def tostring(self):
        '''
        Prints state, action, and reward lists to screen.
        '''
        actionString = ','.join(map(lambda x:str(x.act), deepcopy(self.atrace)))
        stateString = '\n'.join(map(lambda x:str(x.state), deepcopy(self.strace)))
        print "Actions: ", actionString
        print "States ", stateString
        print "Rewards: ", self.rtrace
        
    def getWeightedReward(self):
        '''
        Returns the reward weighted by normalised accumulated weights. Used for multiagent learning in committee.
        
        :returns: the reward weighted by normalised accumulated weights
        '''
        reward = self.totalreward
        if self.totalMAweight != 0:
            normWeight = self.totalMAweight/(len(self.strace)-1) # we subtract 1 as the last entry is TerminalState
            reward *= normWeight
        return reward
        
class EpisodeStack(object):
    '''
    A handler for episodes. Required if stack size is to become very large - may not want to hold all episodes in memory, but
    write out to file. 
    '''
    def __init__(self, block_size=100):
        self.block_size = block_size
        self.write_batches_to = '_gptraining/'
        self.reset_stack()
        
    def retrieve_episode(self, episode_key):
        '''NB: this should probably be an iterator, using yield, rather than return
        '''
        return self.episodes[episode_key]       # no saftey checks at present
    
    def reset_stack(self):
        self.episodes = {}  # TODO - actually implement some mechanism here to write and retrieve if block size really gets big
        self.episode_count = 0
    
    def episode_keys(self):
        return self.episodes.keys()
    
    def get_stack_size(self):
        return self.episode_count
    
    def add_episode(self, domain_episodes):
        '''Items on stack are dictionaries of episodes for each domain (since with BCM can learn from 2 or more domains if a 
        multidomain dialogue happens)
        '''
        self.episodes[self.episode_count] = domain_episodes
        self.episode_count += 1
    
    def write_episodes(self, episode):
        # TODO - possible method may be to write out to file and replace Episode() instance with path in dict ... 
        pass
    
    def load_episodes(self, episode_id):
        pass
    
class State(object):
    '''
    Dummy class representing one state. Used for recording and may be overridden by sub-class.
    '''
    def __init__(self,state):
        self.state = state
        
class Action(object):
    '''
    Dummy class representing one action. Used for recording and may be overridden by sub-class.
    '''
    def __init__(self,action):
        self.act = action
    
class TerminalState(object):
    '''
    Dummy class representing one terminal state. Used for recording and may be overridden by sub-class.
    ''' 
    def __init__(self):
        self.state = "TerminalState"
    
class TerminalAction(object):
    '''
    Dummy class representing one terminal action. Used for recording and may be overridden by sub-class.
    '''
    def __init__(self):
        self.act = "TerminalAction"
    
