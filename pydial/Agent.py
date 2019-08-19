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
Agent.py - wrapper for all components required in a dialogue system
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Contains 3 classes::
    DialogueAgent, AgentFactoy, CallValidator   

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :class:`utils.DiaAct.DiaAct` |.|
    import :class:`utils.DiaAct.DiaActWithProb` |.|
    import :mod:`semo.SemOManager` |.|
    import :mod:`semanticbelieftracking.SemanticBeliefTrackingManager` |.|
    import :mod:`policy.PolicyManager` |.|
    import :mod:`evaluation.EvaluationManager` |.|
    import :mod:`topictracking.TopicTracking` |.|
    import :mod:`ontology.Ontology` |.|

************************

'''
from utils import Settings, ContextLogger
from topictracking import TopicTracking
from ontology import Ontology
from utils.DiaAct import DiaAct, DiaActWithProb

import time, re
logger = ContextLogger.getLogger('')

__author__ = "cued_dialogue_systems_group"
__version__ = Settings.__version__

#----------------------------------------------------------------
# DIALOGUE AGENT
#----------------------------------------------------------------
class DialogueAgent(object):
    ''' 
    Contains all components required for multi domain dialogue: {topic tracking, semi belief tracking, policy, semo}
    - each of these components is a manager for that ability for all domains. 
    - DialogueAgent() controls the flow of calls/information passing between all of these components in order to execute a dialog
    '''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, agent_id='Smith', hub_id='dialogueserver'):        
        
        # Define all variables in __init__:
        self.prompt_str = None
        self.reward = None
        self.currentTurn = None
        self.maxTurns = None
        self.ENDING_DIALOG = None
        self.SUBJECTIVE_RETRIEVAL_ATTEMPS = None
        self.TASK_RETRIEVAL_ATTEMPTS = None
        self.constraints = None
        self.task = None
        self.taskId = None
        self.subjective = None
        self.session_id = None        
        self.callValidator = CallValidator()
        
        # DEFAULTS:
        # meta params - note these define the 'state' of the dialogue, along with those defined in restart_agent()
        assert(hub_id in ['texthub', 'simulate', 'dialogueserver'])
        self.hub_id = hub_id  # defines certain behaviour of the agent. One of [texthub, simulate, dialogueserver]
        self.agent_id = agent_id
        self.NUM_DIALOGS = 0
        self.SYSTEM_CAN_HANGUP = False
        self.SAVE_FREQUENCY = 10   # save the policy after multiples of this many dialogues 
        self.MAX_TURNS_PROMPT = "The dialogue has finshed due to too many turns"
        self.NO_ASR_MSG = "I am afraid I did not understand. Could you please repeat that."
        self.maxTurns_per_domain = 30
        self.traceDialog = 2
        self.sim_level = 'dial_act'
        
        # CONFIGS:
        if Settings.config.has_option('agent', 'savefrequency'):
            self.SAVE_FREQUENCY = Settings.config.getint('agent', 'savefrequency')
        if Settings.config.has_option("agent","systemcanhangup"):
            self.SYSTEM_CAN_HANGUP = Settings.config.getboolean("agent", "systemcanhangup")
        if Settings.config.has_option("agent", "maxturns"):
            self.maxTurns_per_domain = Settings.config.getint("agent", "maxturns")
        if Settings.config.has_option("GENERAL", "tracedialog"):
            self.traceDialog = Settings.config.getint("GENERAL", "tracedialog")
        if Settings.config.has_option("usermodel", "simlevel"):
            self.sim_level = Settings.config.get("usermodel", "simlevel")

        # TOPIC TRACKING:
        #-----------------------------------------
        self.topic_tracker = TopicTracking.TopicTrackingManager()
        
        
        # SemI + Belief tracker
        self.semi_belief_manager = self._load_manger('semanticbelieftrackingmanager','semanticbelieftracking.SemanticBeliefTrackingManager.SemanticBeliefTrackingManager')
        
        # Policy.
        #-----------------------------------------
        self.policy_manager = self._load_manger('policymanager','policy.PolicyManager.PolicyManager')

        # SemO.
        #-----------------------------------------
        if self.hub_id == 'simulate':      # may or may not have NLG in simulate (default is not to)
            generate_prompts = False
            if Settings.config.has_option('simulate', 'generateprompts'):
                generate_prompts = Settings.config.getboolean('simulate', 'generateprompts')
        else:
            generate_prompts = True  # default for Texthub and DialogueServer
        if generate_prompts:
            self.semo_manager = self._load_manger('semomanager','semo.SemOManager.SemOManager')
        else:
            self.semo_manager = None
            
        # Evaluation Manager.
        #-----------------------------------------
        self.evaluation_manager = self._load_manger('evaluationmanager','evaluation.EvaluationManager.EvaluationManager')
        
        # Restart components - NB: inefficient - will be called again before 1st dialogue - but enables _logical_requirements()
        self.restart_agent(session_id=None)
        
        # Finally, enforce some cross module requirements:
        self._logical_requirements()
    
    def start_call(self, session_id, domainSimulatedUsers=None, maxNumTurnsScaling=1.0, start_domain=None, preference=None):
        '''
        Start a new call with the agent.
        Works through  policy > semo -- for turn 0
        Start domain is used if external topic tracking is used.
        
        Input consists of a n-best list of either ASR hypotheses (with confidence) or (mostly only in case of simulation) pre-interpreted DiaActWithProb objects. 

        :param session_id: session id
        :type session_id: string

        :param domainSimulatedUsers: simulated users in different domains
        :type domainSimulatedUsers: dict

        :param maxNumTurnsScaling: controls the variable turn numbers allowed in a dialog, based on how many domains are involved (used only for simulate)
        :type maxNumTurnsScaling: float

        :param start_domain: used by DialPort/external topictracking with DialogueServer to hand control to certain domain 
        :type start_domain: str
        
        :return: string -- the system's reponse
        ''' 
        self._check_agent_not_on_call()
        self.NUM_DIALOGS += 1
        logger.dial(">> NEW DIALOGUE SESSION. Number: "+str(self.NUM_DIALOGS))
        
        # restart agent:
        self.restart_agent(session_id, maxNumTurnsScaling, start_domain=start_domain)
        
        self.callValidator.init() 
        
        # SYSTEM STARTS DIALOGUE first turn:
        #---------------------------------------------------------
        
        self._print_turn()
        
        currentDomain = self.topic_tracker.operatingDomain
        last_sys_act = self.retrieve_last_sys_act(currentDomain)
            
        # SYSTEM ACT:
        # 1. Belief state tracking -- (currently just in single domain as directed by topic tracker)
        logger.debug('active domain is: '+currentDomain)
        
        state = self.semi_belief_manager.update_belief_state(ASR_obs=None, sys_act=last_sys_act,
                                                     dstring=currentDomain, turn=self.currentTurn,hub_id = self.hub_id)
        
        
        # 2. Policy -- Determine system act/response
        sys_act = self.policy_manager.act_on(dstring=self.topic_tracker.operatingDomain, 
                                                  state=state, preference=preference)
        
        self._print_sys_act(sys_act)

        # EVALUATION: - record the system action taken in the current domain if using tasks for evaluation (ie DialogueServer)
        self._evaluate_agents_turn(domainSimulatedUsers, sys_act)  
            
        # SEMO:
        self.prompt_str = self._agents_semo(sys_act)

        
        self.callValidator.validate(sys_act)
        
        sys_act.prompt = self.prompt_str
        state.setLastSystemAct(sys_act)
        
        #---Return the generated prompt---------------------------------------------------
        return sys_act
    
    
    def continue_call(self, asr_info, domainString=None, domainSimulatedUsers=None, preference=None):
        '''
        Works through topictracking > semi belief > policy > semo > evaluation -- for turns > 0
        
        Input consists of a n-best list of either ASR hypotheses (with confidence) or (mostly only in case of simulation) pre-interpreted DiaActWithProb objects.
              
        :param asr_info: information fetched from the asr
        :type asr_info: list of string or DiaActWithProb objects

        :param domainString: domain name
        :type domainString: string

        :param domainSimulatedUsers: simulated users in different domains
        :type domainSimulatedUsers: dict
        
        :return: DiaAct -- the system's reponse dialogue act with verbalization
        ''' 

        logger.dial("user input: {}".format([(x.to_string() if isinstance(x,DiaActWithProb) else x[0], round(x.P_Au_O, 3) if isinstance(x,DiaActWithProb) else x[1]) for x in asr_info]))
        
        # Check if user says bye and whether this is already valid
        self.callValidator.validate() # update time once more
        if self.callValidator.check_if_user_bye(asr_info) and not self.callValidator.isValid:
            logger.info("User tries to end dialogue before min dialogue length.")
            return self.callValidator.getNonValidPrompt() + " " + self.prompt_str

        # 0. Increment turn and possibly set ENDING_DIALOG if max turns reached:
        #--------------------------------------------------------------------------------------------------------------
        if self._increment_turn_and_check_maxTurns():
            sys_act = DiaAct('bye()')
            sys_act.prompt_str = self.MAX_TURNS_PROMPT
            return sys_act
        
        # 1. USER turn:
        #--------------------------------------------------------------------------------------------------------------
        
        # Make sure there is some asr information:
        if not len(asr_info):
            sys_act = DiaAct('null()')
            sys_act.prompt_str = self.NO_ASR_MSG
            return sys_act
        
        # TOPIC TRACKING: Note: can pass domainString directly here if cheating/developing or using simulate
        currentDomain = self._track_topic_and_hand_control(domainString=domainString, userAct_hyps=asr_info)
        prev_sys_act = self.retrieve_last_sys_act(currentDomain)
        
        
        # 2. SYSTEM response:
        #--------------------------------------------------------------------------------------------------------------
        
        # SYSTEM ACT:
                # 1. Belief state tracking -- (currently just in single domain as directed by topic tracker)
        logger.debug('active domain is: '+currentDomain)
        
        state = self.semi_belief_manager.update_belief_state(ASR_obs=asr_info, sys_act=prev_sys_act,
                                                     dstring=currentDomain, turn=self.currentTurn,hub_id = self.hub_id, sim_lvl=self.sim_level)

        self._print_usr_act(state, currentDomain)

        # 2. Policy -- Determine system act/response
        sys_act = self.policy_manager.act_on(dstring=currentDomain, 
                                                  state=state, preference=preference)

        # Check ending the call:
        sys_act = self._check_ENDING_CALL(state, sys_act)  # NB: this may change the self.prompt_str
        
        self._print_sys_act(sys_act)        
        
        # SEMO:
        # print(sys_act)
        self.prompt_str = self._agents_semo(sys_act)

        sys_act.prompt = self.prompt_str
        
        
        # 3. TURN ENDING
        #-----------------------------------------------------------------------------------------------------------------
        
        # EVALUATION: - record the system action taken in the current domain if using tasks for evaluation (ie DialogueServer)
        self._evaluate_agents_turn(domainSimulatedUsers, sys_act, state)
        
        self.callValidator.validate(sys_act)
        
        sys_act.prompt = self.prompt_str
        state.setLastSystemAct(sys_act)
        
        #---Return the generated prompt---------------------------------------------------
        return sys_act
       
    
    def end_call(self, domainSimulatedUsers=None, noTraining=False, preference=None):
        '''
        Performs end of dialog clean up: policy learning, policy saving and housecleaning. The NoTraining parameter is used in 
        case of an abort of the dialogue where you still want to gracefully end it, e.g., if the dialogue server receives 
        a clean request. 

        :param domainSimulatedUsers: simulated users in different domains
        :type domainSimulatedUsers: dict
        
        :param noTraining: train the policy when ending dialogue
        :type noTraining: bool
        
        :return: None
        '''
        self.evaluation_manager.record_preference(preference.cpu().numpy())
        # Finalise any LEARNING:
        finalInfo = {}
        if self.hub_id=='simulate' and domainSimulatedUsers is not None:
            usermodels = {}
            for operatingDomain in domainSimulatedUsers:
                user_model_holder = domainSimulatedUsers[operatingDomain]
                if user_model_holder is not None:
                    usermodels[operatingDomain] = user_model_holder.um
            finalInfo['usermodel'] = usermodels

        finalInfo['task'] = self.task
        finalInfo['subjectiveSuccess'] = self.subjective
        final_rewards = self.evaluation_manager.finalRewards(finalInfo)
        self.policy_manager.finalizeRecord(domainRewards = final_rewards)
        if not noTraining:
            if self.callValidator.isTrainable:
                self.policy_manager.train(self.evaluation_manager.doTraining())
        # Print dialog summary.
        self.evaluation_manager.print_dialog_summary()
        # Save the policy:
        self._save_policy()
        self.session_id = None  # indicates the agent is not on a call. 
        self.ENDING_DIALOG = False
        

    def restart_agent(self, session_id, maxNumTurnsScaling=1.0, start_domain=None):
        '''
        Restart the agnet

        :param session_id: unique session identifier for the dialogue
        :type session_id: str
        
        :param maxNumTurnsScaling: controls the variable number of turns allowed for the dialog, based on how many domains are involved (used only for simulate)
        :type maxNumTurnsScaling: float
        
        :param start_domain: used by DialPort/external topictracking with DialogueServer to hand control to certain domain 
        :type start_domain: str

        :return: None
        '''
        self.currentTurn = 0
        Settings.global_currentturn = self.currentTurn #TODO: this is used in the action mask. would this work for multiple domain dialogues?
        self.maxTurns = self.maxTurns_per_domain * maxNumTurnsScaling
        self.ENDING_DIALOG = False
        self.SUBJECTIVE_RETRIEVAL_ATTEMPS = 0
        self.TASK_RETRIEVAL_ATTEMPTS = 0
        self.ood_count = 0
        # Init the specific dialogue parameters:
        self.constraints = None # used to conditionally set the belief state of a new domain
        self.task = None  # The task this dialogue involved
        self.taskId = None  # The task this dialogue involved
        self.subjective = None   # The 0/1 DTMF feedback from MTurk user - received or not?
        self.session_id = session_id
        
        # Restart all STATEFULL objects: {topic tracker, belief, policy, evaluation}
        self.topic_tracker.restart()
        self.policy_manager.restart()
        self.semi_belief_manager.restart()
        self.evaluation_manager.restart()
        
        # Give initial control to starting domain/topictracker: 
        if start_domain is not None:
            self.topic_tracker.operatingDomain = start_domain
        self._hand_control(domainString=self.topic_tracker.operatingDomain, previousDomainString=None)
    
    
    def retrieve_last_sys_act(self, domainString=None):
        '''
        Retreives the sys act from domain domainString if a domain switch has occurred

        :param domainString: domain name
        :type domainString: string
              
        :return: string -- the system's dialogue act reponse
        '''
        
        if domainString is None:
            domainString = self.topic_tracker.operatingDomain
        sys_act = self.policy_manager.getLastSystemAction(domainString)
        return sys_act

    def _logical_requirements(self):
        '''
        Ensure system always says hello at first turn 

        :return: None
        '''
        if self.topic_tracker.USE_SINGLE_DOMAIN:
            if self.policy_manager.domainPolicies[self.topic_tracker.operatingDomain].startwithhello is False:
                logger.warning('Enforcing hello() at first turn in singledomain system')
                self.policy_manager.domainPolicies[self.topic_tracker.operatingDomain].startwithhello = True
        return      
            
    
    def _track_topic_and_hand_control(self, userAct_hyps=None, domainString=None):
        """
        userAct_hyps can possibly be various things: for example just semantic hyps with probs, or ASR n-best

        :param userAct_hyps: hypotheses of user's dialogue act
        :type userAct_hyps: list

        :param domainString: domain name
        :type domainString: string

        :return: either None or a list of conditional constraints
        """
        # 1. Track the topic
        newDomainString = self.topic_tracker.track_topic(userAct_hyps, domainString)  
        
        # 2. Hand control to [__ASSUMPTION__] single domain  (NB:tracker may end up returning a list --- committee) 
        self._hand_control(domainString=newDomainString, previousDomainString=self.topic_tracker.previousDomain)
               
        return newDomainString
    
    def _hand_control(self, domainString, previousDomainString=None):
        """
        Hands control of dialog to 'domainString' domain. Boots up the ontology/belief/policy as required.  

        :param domainString: domain name
        :type domainString: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        """
        # Ensure that the ontology for domainString is loaded:
        
#         logger.info('agent: _hand_control')
        
        Ontology.global_ontology.ensure_domain_ontology_loaded(domainString)
        
        if domainString is None:
            logger.warning("Topic tracker was unable to resolve domain. Remaining with previous turns operating domain: "\
                    + self.topic_tracker.operatingDomain)
        elif not self._check_currently_booted(domainString):
            # The Bootup part - when first launching a new dialogue manager:
            logger.info('Launching Dialogue Manager for domain: '+domainString)
            # 1. Note that this domain is now involved in current dialog:
            self.topic_tracker.in_present_dialog.append(domainString)
            self._bootup(domainString, previousDomainString)
            return 
        elif previousDomainString is not None: 
            # then we are switching domains: 
            if domainString not in self.topic_tracker.in_present_dialog:
                # note that this domain is now involved in current dialog:
                self.topic_tracker.in_present_dialog.append(domainString)
                self.semi_belief_manager.hand_control(domainString, previousDomainString)
                logger.info('Handing control from {} to running - {} - dialogue manager'.format(previousDomainString,domainString))                 
                return 
            else:
                # moving back to a domain that occured in an earlier turn of dialog. 
                logger.info('Handing control <BACK> to running - '+domainString+' - dialogue manager')
        else:
            logger.info('Domain '+domainString+' is both already running and has control') 
        return

    def _bootup(self, domainString, previousDomainString=None):
        """
        Note: only time bootup is called, self.topic_tracker.operatingDomain is set to the domain already

        :param domainString: domain name
        :type domainString: string

        :param previousDomainString: previous domain name
        :type previousDomainString: string

        :return: None
        """ 
        self.semi_belief_manager.bootup(domainString, previousDomainString)
        self.policy_manager.bootup(domainString)
        return   

    def _check_currently_booted(self, dstring):
        """
        Pertains to whole simulate run over multiple dialogs. Has the dialog manager and belief tracker for this domain been 
        booted up?       

        :param dstring: domain name string
        :param dstring: string

        :return: bool -- whether the system is booted or not

        """
        policy_booted = dstring in [domain for domain, value in self.policy_manager.domainPolicies.iteritems() if value is not None]
        belief_booted = dstring in [domain for domain, value in self.semi_belief_manager.domainSemiBelieftrackers.iteritems() if value is not None]
        return policy_booted and belief_booted
    
    def _evaluate_agents_turn(self, domainSimulatedUsers=None, sys_act=None, state=None):
        '''  
        This function needs to record per exchange rewards and pass them to dialogue management.
        
        NB: asssume that the initiative is with the user only, and that we evaluate each exchange - which consists of a user turn,
        followed by a system response.
        Currently, the process is as follows: if singledomain is False -- then Generic policy starts and we ignore the first turn
        If singledomain==True - then we enforce the system to say hello() at turn 0, and ignore the rest of this method
        --> in both cases we form User+System response pairs and evaluate these
        
        .. warning:: assumes that ONLY the user has initiative to change domains 

        :param domainSimulatedUsers: simulated users in different domains
        :type domainSimulatedUsers: dict

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: None
        '''
        
        
        if self.currentTurn==0:
            logger.debug('Note that we are ignoring any evaluation of the systems first action')
            return   
        operatingDomain = self.topic_tracker.operatingDomain       # Simply for easy access:
        
        
#         # 0. If using RNN evaluator, extract turn level feature:
#         #---------------------------------------------------------------------------------------------------------
#         
#         if self.evaluation_manager.domainEvaluators[operatingDomain] is not None:
#             if self.evaluation_manager.domainEvaluators[operatingDomain].success_measure == "rnn":
#                 # This is the order that makes sense I think - userAct THEN sys RESPONSE = domain pair
#                 turnNo = self.evaluation_manager.domainEvaluators[operatingDomain].final_turns +1
#                 # turn count for this domain only - note turn count will be incremented by per_turn_reward
#                 belief = self.semi_belief_manager.getDomainBelief(operatingDomain)
#                 prev_sys_act = self.retrieve_last_sys_act(operatingDomain)  
#                     # TODO -double check that this is the correct act above
#                 self.evaluation_manager.domainEvaluators[operatingDomain].rnn.set_turn_feature(belief,prev_sys_act,turnNo)
            
        
        # 1. Get reward
        #---------------------------------------------------------------------------------------------------------
        self.reward = None
        turnInfo = {}
        turnInfo['sys_act']=sys_act.to_string()
        turnInfo['state']=state
        turnInfo['prev_sys_act']=state.getLastSystemAct(operatingDomain)
        if self.hub_id=='simulate':
            user_model_holder = domainSimulatedUsers[operatingDomain]
            if user_model_holder is None:
                logger.warning('Simulated user not present for domain %s - passing reward None thru to policy' % operatingDomain)
            else:
                turnInfo['usermodel'] = user_model_holder.um
                
        self.reward = self.evaluation_manager.turnReward(domainString=operatingDomain, turnInfo=turnInfo)
                
        # 2. Pass reward to dialogue management:
        #--------------------------------------------------------------------------------------------------------- 
        self.policy_manager.record(domainString=operatingDomain, reward=self.reward) 
           
        return
    
        
    def _print_turn(self):
        '''
        Prints the turn in different ways for different hubs (dialogueserver, simulate or texthub)

        :return: None
        '''
        if self.hub_id=='dialogueserver':
            logger.dial('Turn %d' % self.currentTurn)
        else:
            if self.traceDialog>1: print '   Turn %d' % self.currentTurn
            logger.dial('** Turn %d **' % self.currentTurn)
        return
    
    def _print_sys_act(self, sys_act):
        '''Prints the system act in different ways for different hubs (dialogueserver, simulate or texthub)

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: None
        '''
        if self.hub_id=='dialogueserver':
            logger.dial('Sys > {}'.format(sys_act))
        else:
            if self.traceDialog>1: print '   Sys > {}'.format(sys_act)
            logger.dial('| Sys > {}'.format(sys_act))
    
    def _print_usr_act(self, state, currentDomain):
        '''Prints the system act in different ways for different hubs (dialogueserver, simulate or texthub)

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: None
        '''
        if self.traceDialog>2: state.printUserActs(currentDomain)
    
    def _check_ENDING_CALL(self, state = None, sys_act = None):
        '''
        Sets self.ENDING_DIALOG as appropriate -- checks if USER ended FIRST, then considers SYSTEM. 

        :param state: system's state (belief)
        :type state: :class:`~utils.DialgoueState.DialgoueState`

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: bool -- whether to end the dialogue or not
        '''
        sys_act = self._check_USER_ending(state, sys_act=sys_act)
        if not self.ENDING_DIALOG:
            # check if the system can end the call
            self._check_SYSTEM_ending(sys_act)
        return sys_act
    
    def _check_SYSTEM_ending(self, sys_act):
        ''' 
        Checks if the *system* has ended the dialogue

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: None
        '''
        
        
        if self.SYSTEM_CAN_HANGUP:      # controls policys learning to take decision to end call. 
            if sys_act.to_string() in ['bye()']:
                self.ENDING_DIALOG = True    # SYSTEM ENDS
        else:
            # still possibly return true if *special* domains [topictracker, wikipedia] have reached their own limits
            if sys_act.to_string() in ['bye(toptictrackertimedout)', 'bye(wikipediatimedout)', 'bye(topictrackeruserended)']:
                self.ENDING_DIALOG = True   # SYSTEM ENDS
            
    
    def _check_USER_ending(self, state = None, sys_act = None):
        '''Sets boolean self.ENDING_DIALOG if user has ended call. 
        
        .. note:: can change the semo str if user ended.

        :param state: system's state (belief)
        :type state: :class:`~utils.DialgoueState.DialgoueState`

        :param sys_act: system's dialogue act
        :type sys_act: string

        :return: bool -- whether to end the dialogue or not
        '''
#         assert(not self.ENDING_DIALOG)

        self.ENDING_DIALOG = state.check_user_ending()
        if self.ENDING_DIALOG:  
            if self.semo_manager is not None:   
                if 'bye' not in self.prompt_str:
                    logger.warning('Ignoring system act: %s and saying goodbye as user has said bye' % sys_act)
                    self.prompt_str = 'Goodbye. '       # TODO - check how system can say bye --otherwise user has said bye,
                    #  and we get some odd reply like 'in the south. please enter the 5 digit id ...'
            sys_act = DiaAct('bye()')
        return sys_act

    def _agents_semo(self, sys_act):
        '''
        Wrapper for semo -- agent used in simulate for example may not be using semo. 
        
        :param sys_act: system's dialogue act
        :type sys_act: string

        :return string -- system's sentence reponse
        '''
        if self.semo_manager is not None:
            logger.dial("Domain with CONTROL: "+self.topic_tracker.operatingDomain)
            prompt_str = self.semo_manager.generate(sys_act, domainTag=self.topic_tracker.operatingDomain)
        else:
            prompt_str = None 
        return prompt_str                      


    def _save_policy(self, FORCE_SAVE=False):
        """
        A wrapper for policy_manager.savePolicy()  - controls frequency with which policy is actually saved
        
        :param FORCE_SAVE: whether to force save the policy or not 
        :type FORCE_SAVE: bool
    
        :return: None
        """
        if FORCE_SAVE:
            self.policy_manager.savePolicy(FORCE_SAVE)
        elif self.NUM_DIALOGS % self.SAVE_FREQUENCY == 0:
            self.policy_manager.savePolicy()
        else:
            pass # neither shutting down agent, nor processed enough dialogues to bother saving. 
        return

    # Agent Utility functions:
    #--------------------------------------
    def _increment_turn_and_check_maxTurns(self):
        '''
        Returns boolean from :method:_check_maxTurns - describing whether dialog has timed (ie turned!) out. 

        :return: bool -- call _check_maxTurns() to check whether the conversation reaches max turns
        '''
        self._increment_turn()
        self._print_turn()
        return self._check_maxTurns()
    
    def _increment_turn(self):
        ''' 
        Count turns.

        :return: None
        '''
        self.currentTurn += 1
        Settings.global_currentturn = self.currentTurn
    
    def _check_maxTurns(self):
        '''
        Checks that we haven't exceeded set max number. Note: sets self.prompt_str if max turns reached. 
        Returns a boolean on this check of num turns.

        :return: bool -- check whether the conversation reaches max turns
        ''' 
        if self.currentTurn > self.maxTurns:
            logger.dial("Ending dialog due to MAX TURNS being reached: "+str(self.currentTurn))
            self.ENDING_DIALOG = True
            return True
        return False
    
    def _check_agent_not_on_call(self):
        ''' 
        need a check here that the agent is indeed not talking to anyone ...

        :return: None
        '''
        if self.session_id is not None:
            logger.error("Agent is assumed to be only on one call at a time")
            
            
    def _load_manger(self, config, defaultManager):
        '''
        Loads and instantiates the respective manager (e.g. policymanager, semomanager, etc) as configured in config file. The new object is returned.
        
        :param config: the config option which contains the manager configuration
        :type config: str
        :param defaultManager: the config string pointing to default manager
        :type defaultManager: str
        :returns: the new manager object
        '''
        
        manager = defaultManager
            
        if Settings.config.has_section('agent') and Settings.config.has_option('agent', 'config'):
            manager = Settings.config.has_option('agent', 'config')
        try:
            # try to view the config string as a complete module path to the class to be instantiated
            components = manager.split('.')
            packageString = '.'.join(components[:-1]) 
            classString = components[-1]
            mod = __import__(packageString, fromlist=[classString])
            klass = getattr(mod, classString)
            return klass()
        except ImportError as e:
            logger.error('Manager "{}" could not be loaded: {}'.format(manager, e))



#******************************************************************************************************************
# AGENT FACTORY
#******************************************************************************************************************
class AgentFactory(object):
    '''
    Based on the config (Settings.config) - a primary agent (called Smith) is created. 
    This agent can be duplicated as required by concurrent traffic into the dialogue server. 
    Duplicated agents are killed at end of their calls if more agents are running
    than a specified minimum (MAX_AGENTS_RUNNING)
    '''
    def __init__(self, hub_id='dialogueserver'):
        self.MAX_AGENTS_RUNNING = 2  # always start with 1, but we dont kill agents below this number
        self.init_agents(hub_id)
        self.session2agent = {}
        self.historical_sessions = []

    def init_agents(self, hub_id):
        '''
        Creates the first agent. All other agents created within the factory will be deep copies of this agent. 

        :param hub_id: hub id
        :type hub_id: string

        :return: None
        '''
        self.agents = {}
        self.agents['Smith'] = DialogueAgent(agent_id='Smith', hub_id=hub_id)  # primary agent, can be copied
    
    def start_call(self, session_id, start_domain=None, preference=None):
        '''
        Locates an agent to take this call and uses that agents start_call method. 

        :param session_id: session_id
        :type session_id: string
               
        :param start_domain: used by DialPort/external topictracking with DialogueServer to hand control to certain domain 
        :type start_domain: str

        :return: start_call() function of agent object, string -- the selected agent, agent id
        '''
        agent_id = None
        
        # 1. make sure session_id is not in use by any agent
        if session_id in self.session2agent.keys():
            agent_id = self.session2agent[session_id]
            logger.info('Attempted to start a call with a session_id %s already in use by agent_id %s .' % (session_id, agent_id))
        # 2. check if there is an inactive agent
        if agent_id is None:
            for a_id in self.agents.keys():
                if self.agents[a_id].session_id is None:
                    agent_id = a_id
                    break
        # 3. otherwise create a new agent for this call
        if agent_id is None:
            agent_id = self.new_agent()
        else:
            logger.info('Agent {} has been reactivated.'.format(agent_id))
        # 4. record that this session is with this agent, and that it existed:
        self.session2agent[session_id] = agent_id
        self.historical_sessions.append(session_id)
        # 5. start the call with this agent:
        return self.agents[agent_id].start_call(session_id, start_domain=start_domain, preference=preference), agent_id
    
    def continue_call(self, agent_id, asr_info, domainString=None, preference=None):
        '''
        wrapper for continue_call for the specific Agent() instance identified by agent_id

        :param agent_id: agent id
        :type agent_id: string
        
        :param asr_info: information fetched from the asr
        :type asr_info: list

        :param domainString: domain name
        :type domainString: string

        :return: string -- the system's response
        '''
        prompt_str = self.agents[agent_id].continue_call(asr_info, domainString, preference)
        return prompt_str
    
    def end_call(self, agent_id=None, session_id=None, noTraining=False, preference=None):
        '''
        Can pass session_id or agent_id as we use this in cases 
            1) normally ending a dialogue, (via agent_id) 
            2) cleaning a hung up call      (via session_id)

        :param agent_id: agent id
        :type agent_id: string

        :param session_id: session_id
        :type session_id: string
        
        :return: None
        '''
        # 1. find the agent if only given session_id
        if agent_id is None: # implicitly assume session_id is given then
            agent_id = self.retrieve_agent(session_id)
        logger.info('Ending agents %s call' % agent_id)
        # 2. remove session from active list
        session_id = self.agents[agent_id].session_id
        del self.session2agent[session_id]        
        # 3. end agents call
        self.agents[agent_id].end_call(noTraining=noTraining)
        # 4. can we also delete agent?
        self.kill_agent(agent_id)
        
    def agent2session(self, agent_id):
        '''
        Gets str describing session_id agent is currently on

        :param agent_id: agent id
        :type agent_id: string
        
        :return: string -- the session id
        '''
        return self.agents[agent_id].session_id
    
            
    def retrieve_agent(self, session_id):
        '''
        Returns str describing agent_id. 
        
        :param session_id: session_id
        :type session_id: string
        
        :return: string -- the agent id
        '''
        if session_id not in self.session2agent.keys():
            logger.error('Attempted to get an agent for unknown session %s' % session_id)
        return self.session2agent[session_id]
    
    def query_ENDING_DIALOG(self, agent_id):
        '''
        Wrapper for specific Agent.ENDING_DIALOG() -- with some basic initial checks.
        
        :param agent_id: agent id
        :type agent_id: string
        
        :return: bool -- whether to end the dialogue or not
        '''
        if agent_id is None:
            return False
        if agent_id not in self.agents.keys():
            logger.error('Not an existing agent: '+str(agent_id))
        return self.agents[agent_id].ENDING_DIALOG
    
    def new_agent(self):
        '''
        Creates a new agent to handle some concurrency. 
        Here deepcopy is used to creat clean copy rather than referencing, 
        leaving it in a clean state to commence a new call. 
        
        :return: string -- the agent id
        '''
        agent_id = 'Smith' + str(len(self.agents))
#         self.agents[agent_id] = copy.deepcopy(self.agents['Smith']) # alternative to copying is a new DialogueAgent() object  
        self.agents[agent_id] = DialogueAgent(agent_id)
        self.agents[agent_id].restart_agent(session_id=None) # VERY IMPORTANT AFTER copying!!
        logger.info('Agent {} has been created.'.format(agent_id))
        return agent_id
    
    def kill_agent(self, agent_id):
        '''
        Delete an agent if the total agent number is bigger than self.MAX_AGENTS_RUNNING.
                 
        :param agent_id: agent id
        :type agent_id: string
        
        :return: None
        '''
        if agent_id == 'Smith':
            return # never kill our primary agent
        agent_number = int(agent_id.strip('Smith'))
        if agent_number > self.MAX_AGENTS_RUNNING:
            del self.agents[agent_id]
            # TODO - WHEN LEARNING IS INTRODUCED -- will need to save policy etc before killing 
    
    def power_down_factory(self):
        '''
        Finalise agents, print the evaluation summary and save the policy we close dialogue server.
        
        :return: None
        '''

        for agent_id in self.agents.keys():
            logger.info('Summary of agent: %s' % agent_id)
            self.agents[agent_id].evaluation_manager.print_summary()
            self.agents[agent_id]._save_policy(FORCE_SAVE=True) #always save at end, can otherwise miss some dialogs by saving every 10
        logger.info('Factory handled these sessions: %s' % self.historical_sessions)


class CallValidator(object):
    '''
    Used to validate calls, e.g., when using PyDial within user experiments. 
    
    Calls may be validated after a minimum of length in seconds or turns or if the system offers a venue. The flag isTrainable may be used to distinguish between dialogues whose formal conditions for validity are fulfilled
    but who will introduce to much noise in the training process, e.g., if you allow for users to regularly abort the dialogue after 2 minutes but only want to use the dialogue for training if a minimum of 3 turns have
    carried out. 
    '''
    def __init__(self):
        self.startTime = 0
        self.turns = 0
        self.isValid = False
        self.mindialoguedurationprompt = "You cannot finish the dialogue yet, please try just a bit more." # Prompt when user says bye before miduration and without system having informed
        self.mindialogueduration = 0 # Minimun duration length in seconds before giving token. Default is 0 (disabled)
        self.mindialogueturns = 0 # Minimum number of turns before giving token. Default is 0 (disabled)
        self.isTrainable = False
        
        self.doValidityCheck = False
        
        if Settings.config.has_option('agent', 'mindialoguedurationprompt'):
            self.mindialoguedurationprompt = Settings.config.get('agent', 'mindialoguedurationprompt')
        if Settings.config.has_option('agent', 'mindialogueturns'):
            self.mindialogueturns = Settings.config.getint('agent', 'mindialogueturns')
        if Settings.config.has_option('agent', 'mindialogueduration'):
            self.mindialogueduration = Settings.config.getint('agent', 'mindialogueduration')
            
        if self.mindialogueduration > 0 or self.mindialogueturns > 0:
            self.doValidityCheck = True
    
    def init(self):
        self.startTime = time.time()
        self.turns = 0
        self.isValid = False
        self.isTrainable = False
        
    def validate(self, sys_act = None):
        timePassed = time.time()  - self.startTime
        
        timeStatus = timePassed >= self.mindialogueduration
        sysActStatus = False
        if sys_act is not None:
            self.turns += 1
            sys_act_string = sys_act.to_string()
            sysActStatus = 'bye' in sys_act_string or 'inform' in sys_act_string # if system says goodbye (eg if maxTurns reached) or sys provides one result (needs not to be the correct one)
        
        turnStatus = self.turns >= self.mindialogueturns
        
        if timeStatus:
            if self.doValidityCheck:
                logger.info("Call is valid due to time ({} sec passed).".format(timePassed))
            self.isValid = True
            self.isTrainable = True
        
        if sysActStatus:
            if self.doValidityCheck:
                logger.info("Call is valid due to system act.")
            self.isValid = True
            self.isTrainable = True
            
        if turnStatus:
            if self.doValidityCheck:
                logger.info("Call is valid due to number of turns ({} turns).".format(self.turns))
            self.isValid = True
            self.isTrainable = True
        
        return self.isValid
    
    def getNonValidPrompt(self):
        return self.mindialoguedurationprompt
    
    
    def check_if_user_bye(self,obs):
        """
        Checks using a regular expression heuristic if the user said good bye. In accordance with C++ system, prob of respective n-best entry must be > 0.85.
        """
        for ob in obs:
            # ASR / Texthub input
            if isinstance(ob, tuple):
                #sentence,sentence_prob = ob[0],ob[1]
                sentence,_ = ob[0],ob[1]
            # simulated user input
            elif isinstance(ob,DiaAct):
                sentence = ob.act
            elif isinstance(ob,str):
                sentence,_ = ob, None    
            rBYE = "(\b|^|\ )(bye|goodbye|that'*s*\ (is\ )*all)(\s|$|\ |\.)"
            if self._check(re.search(rBYE,sentence, re.I)):
#                 if sentence_prob > 0.85:
                return True # one best input so prob is not relevant
        return False
        
    def _check(self,re_object):
        """
        """
        if re_object is None:
            return False
        for o in re_object.groups():
            if o is not None:
                return True
        return False

#END OF FILE
