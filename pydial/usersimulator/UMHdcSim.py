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
UMHdcSim.py - Handcrafted simulated user behaviour 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`usersimulator.UserModel` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.dact` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

**Relevant config variables** (values are defaults)::

    [um]
    usenewgoalscenarios = True 
    answerreqalways = False

    [goalgenerator]
    patience = 10

'''

__author__ = "cued_dialogue_systems_group"
import copy, os

import UserModel, UMSimulator
from ontology import Ontology
from utils import Settings, DiaAct, dact, ContextLogger
logger = ContextLogger.getLogger('')

# The following parameters are originally specified in usin_params_caminfo.txt of C++ system,
# But due to a bug when loading the above file into the memory, the parameters below are being used.
# rand_decision_probs = {'InformCombination':     0.6,
#                        'AddSlotToReq':          0.333,
#                        'Greeting':              0.5,
#                        'YesAfterReqmore':       0.25,
#                        'ConstraintRelax':       0.667,
#                        'TellAboutChange':       0.5,
#                        'ByeOrStartOver':        0.333,
#                        'DealWithPending':       0.5,
#                        'AddVenueNameToRequest': 0.05,
#                        'NoSlotWithDontcare':    0.8,
#                        'RequestSlotAtStart':    0.05,
#                        'Repeat':                0.0,
#                        'InformToConfirm':       0.05,
#                        'CombAffirmWithAgdItem': 0.05,
#                        'NullResp':              0.0,
#                        'OverruleCorrection':    0.1,
#                        'ConfirmRandomConstr':   0.1,
#                        'ReqAltsAfterVenRec1':   0.143,
#                        'ReqAltsAfterVenRec2':   0.143,
#                        'NewRequestResp1':       0.2,
#                        'NewRequestResp2':       0.2,
#                        'CorrectingAct1':        0.45,
#                        'CorrectingAct2':        0.4,
#                        'ThankAck1':             0.1,
#                        'ThankAck2':             0.1,
#                        'NoAfterReqmore':        0.7}

# # from eddy
'''rand_decision_probs = {'InformCombination':     0.6,
                       'AddSlotToReq':          0.333,
                       'NoAfterReqmore':        0.7,
                       'YesAfterReqmore':       0.25,
                       'Greeting':              0.5,
                       'ConstraintRelax':       0.667,
                       'TellAboutChange':       0.5,
                       'ByeOrStartOver':        0.333,
                       'DealWithPending':       0.5,
                       'AddVenueNameToRequest': 0.05,
                       'NoSlotWithDontcare':    0.8,
                       'Repeat':                0.0,
                       'InformToConfirm':       0.05,
                       'CombAffirmWithAgdItem': 0.05,
                       'NullResp':              0.0,
                       'OverruleCorrection':    0.1,
                       'ConfirmRandomConstr':   0.1,
                       'ReqAltsAfterVenRec1':   0.143, #0.8
                       'ReqAltsAfterVenRec2':   0.143,
                       'NewRequestResp1':       0.2,
                       'NewRequestResp2':       0.2,
                       'CorrectingAct1':        0.45,
                       'CorrectingAct2':        0.4,
                       'ThankAck1':             0.1,
                       'ThankAck2':             0.1,
                       'AffirmCombination':     1.0}'''

# from david
# rand_decision_probs = {'InformCombination':     0.6,
#                        'AddSlotToReq':          0.333,
#                        'NoAfterReqmore':        0.7,
#                        'YesAfterReqmore':       0.25,
#                        'Greeting':              0.5,
#                        'ConstraintRelax':       0.667,
#                        'TellAboutChange':       0.5,
#                        'ByeOrStartOver':        0.333,
#                        'DealWithPending':       0.5,
#                        'AddVenueNameToRequest': 0.05,
#                        'NoSlotWithDontcare':    0.8,
#                        'Repeat':                0.0,
#                        'InformToConfirm':       0.05,
#                        'CombAffirmWithAgdItem': 0.05,
#                        'NullResp':              0.05,
#                        'OverruleCorrection':    0.0,
#                        'ConfirmRandomConstr':   0.1,
#                        'ReqAltsAfterVenRec1':   0.1,
#                        'ReqAltsAfterVenRec2':   0.143,
#                        'NewRequestResp1':       0.143,
#                        'NewRequestResp2':       0.2,
#                        'CorrectingAct1':        0.2,
#                        'CorrectingAct2':        0.45,
#                        'ThankAck1':             0.4,
#                        'ThankAck2':             0.1,
#                        'NoAfterReqmore':        0.7}


class UMHdcSim(UMSimulator.UMSimulator):
    '''Handcrafted behaviour of simulated user
    '''
    def __init__(self, domainString, max_patience = 5):
        super(UMHdcSim,self).__init__(domainString,max_patience)
        
        # DEFAULTS:
        self.answer_req_always = True
        self.use_new_goal_scenarios = False
        self.sampleDecisiconProbs = False
        self.patience_old_style = False
        self.old_style_parameter_sampling = True
        config_file_path = 'config/defaultUM.cfg'
        self.rand_decision_probs = {'InformCombination':     0.6,
                       'AddSlotToReq':          0.333,
                       'NoAfterReqmore':        0.7,
                       'YesAfterReqmore':       0.25,
                       'Greeting':              0.5,
                       'ConstraintRelax':       0.667,
                       'TellAboutChange':       0.5,
                       'ByeOrStartOver':        0.333,
                       'DealWithPending':       0.5,
                       'AddVenueNameToRequest': 0.05,
                       'NoSlotWithDontcare':    0.8,
                       'Repeat':                0.0,
                       'InformToConfirm':       0.05,
                       'CombAffirmWithAgdItem': 0.05,
                       'NullResp':              0.0,
                       'OverruleCorrection':    0.1,
                       'ConfirmRandomConstr':   0.1,
                       'ReqAltsAfterVenRec1':   0.143,
                       'ReqAltsAfterVenRec2':   0.143,
                       'NewRequestResp1':       0.2,
                       'NewRequestResp2':       0.2,
                       'CorrectingAct1':        0.45,
                       'CorrectingAct2':        0.4,
                       'ThankAck1':             0.1,
                       'ThankAck2':             0.1,
                       'AffirmCombination':     1.0}
        
        # CONFIG:
        if Settings.config.has_option('usermodel', 'usenewgoalscenarios'):
            self.use_new_goal_scenarios = Settings.config.getboolean('usermodel', 'usenewgoalscenarios')
        if Settings.config.has_option('usermodel', 'answerreqalways'):
            self.answer_req_always = Settings.config.getboolean('usermodel', 'answerreqalways')

        if Settings.config.has_option('usermodel', 'informcombinationprob'):
            self.rand_decision_probs['InformCombination'] = Settings.config.getfloat('usermodel', 'informcombinationprob')
        if Settings.config.has_option('usermodel', 'affirmcombinationprob'):
            self.rand_decision_probs['AffirmCombination'] = Settings.config.getfloat('usermodel', 'affirmcombinationprob')
        if Settings.config.has_option('usermodel', 'reqaltsaftervenrec'):
            self.rand_decision_probs['ReqAltsAfterVenRec1'] = Settings.config.getfloat('usermodel', 'reqaltsaftervenrec')
        if Settings.config.has_option('usermodel', 'sampledialogueprobs'):
            self.sampleDecisiconProbs = Settings.config.getboolean('usermodel', 'sampledialogueprobs') # deprecated, now this is set in the config file
        if Settings.config.has_option('usermodel', 'oldstylepatience'):
            self.patience_old_style = Settings.config.getboolean('usermodel', 'oldstylepatience')
        if Settings.config.has_option('usermodel', 'oldstylesampling'):
            self.old_style_parameter_sampling = Settings.config.getboolean('usermodel', 'oldstylesampling')
        if Settings.config.has_option('usermodel', 'configfile'):
            config_file_path = Settings.config.get('usermodel', 'configfile')
            
        config_file_path = self._check_config_file_path(config_file_path)
            
        self._read_UM_config(config_file_path)

        self.agenda = UserModel.UMAgenda(self.dstring)
        self.last_user_act = None
        self.last_sys_act = None
        self.relax_constraints = None
        self.first_venue_recommendation = None

        self.receive_options = {'badact': self._receive_badact,
                   'reqmore': self._receive_reqmore,
                   'null': self._receive_badact,
                   'hello': self._receive_hello,
                   'request': self._receive_request,
                   'confirm': self._receive_confirm,
                   'inform': self._receive_inform,
                   'repeat': self._receive_repeat,
                   'select': self._receive_select,
                   'confreq': self._receive_confreq,
                   'bye': self._receive_bye,
                   'affirm': self._receive_affirm,
                   'negate': self._receive_negate}
        self.sampling_probs = None
        for key in self.rand_decision_probs:
            if type(self.rand_decision_probs[key]) is list:
                self.sampling_probs = copy.deepcopy(self.rand_decision_probs)
                break

    def init(self, goal, um_patience):
        """
        """
        # First create the AGENDA:
        self.agenda.init(goal)
        self.last_user_act = DiaAct.DiaAct('null()')
        self.last_sys_act = DiaAct.DiaAct('null()')
        
        #if self.sampleDecisiconProbs:
        if self.sampling_probs:
            self._sampleProbs()
            
        self.max_patience = um_patience
        
        self.relax_constraints = False
        self.first_venue_recommendation = True

        for const in goal.constraints:
            slot = const.slot
            value = const.val
            goal.add_prev_used(slot, value)

    def receive(self, sys_act, goal):
        """
        """
        self.last_sys_act = sys_act

        if goal.is_completed() and self.agenda.size() == 0 and sys_act.dact['act'] != 'reqmore'\
                and Settings.random.rand() < 0.85:
            # Goal already completed: say goodbye.
            self.agenda.clear()
            self.agenda.push(DiaAct.DiaAct('bye()'))
            return

        # Generate repeat act with small probability:
        #   assume the user did not hear the system utterance,
        #   let alone make any updates to their (user goal) state,
        #   and respond with a repeat act.
        if goal.patience > 1 and sys_act.act != 'repeat' and sys_act.act != 'badact' and\
                        sys_act.act != 'null':
            if Settings.random.rand() < self.rand_decision_probs['Repeat']:
                self.agenda.push(DiaAct.DiaAct('repeat()'))
                return

        # Generate null action with small probability:
        #   user generates (silence or) something incomprehensible
        if Settings.random.rand() < self.rand_decision_probs['NullResp']:
            self.agenda.push(DiaAct.DiaAct('null()'))
            return

        if sys_act.act in self.receive_options and sys_act.act != 'null': 
            self.receive_options[sys_act.act](sys_act, goal)
        else:
            logger.warning('Unknown acttype in UMHdcSim.receive(): ' + sys_act.act)
            self._receive_badact(goal)

        logger.debug(str(self.agenda.agenda_items))
        logger.debug(str(goal))

    def respond(self, goal):
        '''
        This method is called to get the user response.

        :param goal: of :class:`UMGoal` 
        :type goal: instance
        :returns: (instance) of :class:`DiaActWithProb`
        '''
        # If agenda is empty, push ByeAct on top.
        if self.agenda.size() == 0:
            self.agenda.push(DiaAct.DiaAct('bye()'))

        # Pop the top act off the agenda to form the user response.
        dap = self.agenda.pop()
        logger.debug(str(dap))

        # if len(dap.items) > 1:
        #     logger.warning('Multiple semantic items in agenda: ' + str(dap))
        dap_item = None
        if len(dap.items) > 0:
            dap_item = dap.items[0]

        # If it created negate(name="!x") or deny(name="x", name="!x") or confirm(name="!x") just reqalts()
        for item in dap.items:
            if item.op == "!=":
                dap = DiaAct.DiaAct('reqalts()')
                break

        # Checking agenda for redundant constraints.
        self.agenda.filter_constraints(dap)

        if dap.act in ['thankyou', 'silence', 'repeat', 'ack', 'deny', 'confirm']:
            return self._normalise_act_no_rules(dap)

        if self.last_sys_act.act == 'reqmore':
            return self._normalise_act_no_rules(dap)

        # Ckecing whether we might remove the slot name for value dontcare in the planned act.
        if dap.act == 'inform' and not dap.items:
            logger.error('Error inform act with no slots is on agenda.')

        # In response to a request about a particular slot users often do not specify hte slot
        # especially when the value is dontcare.
        if self.last_sys_act.act in ['request', 'confreq', 'select']:
            if dap.act == 'inform' and dap_item is not None and dap_item.val == 'dontcare':
                f = Settings.random.rand()
                if f < self.rand_decision_probs['NoSlotWithDontcare']:
                    dap_item.slot = None

        # Checking whether we might add a venue name ot the planned act.
        if dap.act == 'request' and len(dap.items) == 1:
            rec_ven = goal.requests['name']
            # If venue recommended, randomly decide to include the venue name in the request.
            if rec_ven is not None:
                if Settings.random.rand() < self.rand_decision_probs['AddVenueNameToRequest']:
                    dap.append('name', rec_ven)
            # else:
            #     logger.error('Requesting slot without venue recommended.')

        # Checking whether we might include additional constraints in the planned act.
        # When specifying a constraint, combine the act with additional constraints with some probability.
        if dap.act in ['inform', 'negate', 'hello', 'affirm']:
#             print "dialogue act", dap.act, dap.items
            inf_comb_count = 0
            while self.agenda.size() > 0 and \
                    (self.agenda.agenda_items[-1].act == 'inform' or \
                     self.agenda.agenda_items[-1].act == 'request' and dap.act == 'hello'):
                if Settings.random.rand() < self.rand_decision_probs['InformCombination']:
                    inf_comb_count += 1
                    next_dap = self.agenda.pop()
                    for dip in next_dap.items:
                        dap.append(dip.slot, dip.val, dip.op == '!=')
                else:
                    break

        # Checking whether we might request a slot when specifying the type of venue.
        # When specifying the requestType constraint at the beginning of a dialogue,
        # occasionally request an additional requested slot
        if dap.act == 'request' and len(dap.items) > 0 and dap_item.slot in ['type', 'task', 'restaurant']:
            logger.warning('Not completely implemented: RequestSlotAtStart')

        usr_output = self._normalise_act_no_rules(dap)
        self.last_user_act = usr_output
        return usr_output

    def _receive_badact(self,goal):
        if goal.patience < 1:
            self.agenda.push(DiaAct.DiaAct('bye()'))
        else:
            self.agenda.push(copy.deepcopy(self.last_user_act))

    def _receive_hello(self, sys_act, goal):
        if not len(sys_act.items):
            if Settings.random.rand() < self.rand_decision_probs['Greeting']:
                self.agenda.push(DiaAct.DiaAct('hello()'))

    def _receive_bye(self, sys_act, goal):
        self.agenda.clear()
        self.agenda.push(DiaAct.DiaAct('bye()'))

    def _receive_reqmore(self, sys_act, goal):
        # Check if there are pending items on the agenda apart from bye().
        # If yes, just return and wait for them to be dealt with, or turn the top item of the agenda into an affirm.
        if self.agenda.size() > 1:
            next_dap = self.agenda.agenda_items[-1]
            if not next_dap.contains('type', goal.request_type):  # was hard coded to 'restaurant'
                if Settings.random.rand() < self.rand_decision_probs['CombAffirmWithAgdItem']:
                    # Responding with affirm and combine with next agenda item.
                    # Create an affirm act and combine it with the top item of the agenda if that specifies a constraint.
                    # e.g. inform(type=bar) -> affirm(type=bar) or request(bar) -> affirm(=bar)
                    resp_act = DiaAct.DiaAct('affirm()')
                    # Combine this affirm with a constraint from the top of the agenda if possible.
                    if next_dap.act == 'inform' or (next_dap.act == 'request' and next_dap.contains_slot('type')):
                        for item in next_dap.items:
                            new_item = copy.copy(item)
                            if next_dap.act == 'request':
                                new_item.val = new_item.slot
                                new_item.slot = None
                            resp_act.append(new_item.slot, new_item.val, negate=(new_item.op == '!='))
                        self.agenda.pop()
                        self.agenda.push(resp_act)
                return

        # Check if there is an unsatisfied request on the goal
        if goal.are_all_requests_filled():
            # If all requests are filled,
            if Settings.random.rand() < self.rand_decision_probs['NoAfterReqmore']:
                # Occasionally just say no. A good policy can save turns by ending the conversation at this point.
                self.agenda.push(DiaAct.DiaAct('negate()'))
            else:
                informs = self.agenda.get_agenda_with_act('inform')
                confirms = self.agenda.get_agenda_with_act('confirm')

                # If there are still items on the agenda which need to be transmitted to the system,
                # then don't hangup!
                if len(informs) + len(confirms) > 0:
                    return

                # Say bye if there's nothing else to do.
                self.agenda.clear()
                self.agenda.push(DiaAct.DiaAct('bye()'))

        else:
            # If there are unsatisfied requests,
            if goal.is_venue_recommended():
                # If a venue has been recommended already, then ask for empty requests, like phone, addr.
                unsatisfied = goal.get_unsatisfied_requests()
                for info in unsatisfied:
                    self.agenda.push(DiaAct.DiaAct('request(%s)' % info))

            # If no venue has been recommended yet, then asking reqmore() is pretty stupid.
            # Make the system loose a point by answering "yes!"
            else:
                if Settings.random.rand() < self.rand_decision_probs['YesAfterReqmore']:
                    # Nothing recommended yet, so just say yes.
                    self.agenda.push(DiaAct.DiaAct('affirm()'))
                else:
                    # Nothing recommended yet, so just ask for the request_type and all constraints.
                    usr_resp = DiaAct.DiaAct('inform(type=%s)' % goal.request_type )
                    for const in goal.constraints:
                        if const.val != 'dontcare':
                            slots_to_add = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot=const.slot, value=const.val)
                            for slot in slots_to_add:
                                value = goal.get_correct_const_value(slot)
                                if value is not None: usr_resp.append(slot, value)
                                value = goal.get_correct_const_value(slot, negate=True)
                                if value is not None: usr_resp.append(slot, value, negate=True)

                    self.agenda.push(usr_resp)

    def _receive_confirm(self, sys_act, goal):
        # Check the given information.
        if not self._receive_implicit_confirm(sys_act, goal):
            # The given info was not ok, so stop processing confirm act.
            return

        # Check explicit confirmation.
        if not self._receive_direct_implicit_confirm(sys_act, goal):
            # Item in system act needed correction: stop processing confirm act here.
            return

        # Given information is ok. Put an affirm on the agenda if next item on the agenda is an inform act,
        # can affirm and inform in one go.
        # affirm(), or affirm(a=x) if next agneda item is inform.
        new_affirm_act = DiaAct.DiaAct('affirm()')
        if self.agenda.size() > 0:
            top_item = self.agenda.agenda_items[-1]
            if top_item.act == 'inform':
                if Settings.random.rand() < self.rand_decision_probs['AffirmCombination']:
                    for item in top_item.items:
                        new_affirm_act.append(item.slot, item.val, negate=(item.op == '!='))
                    self.agenda.pop()
        self.agenda.push(new_affirm_act)

    def _receive_repeat(self, sys_act, goal):
        if self.last_user_act is not None and self.last_user_act.act != 'silence':
            self.agenda.push(copy.deepcopy(self.last_user_act))

    def _receive_select(self, sys_act, goal):
        # Receive select(slot=x, slot=y)
        slot = sys_act.items[0].slot
        value = sys_act.items[0].val
        if slot == 'name':
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, value)))
            # logger.error('select on name slot.')

        if not goal.contains_slot_const(slot):
            # If slot is not in the goal, get the correct value for it.
            logger.warning('Slot %s in the given system act %s is not found in the user goal.' % (slot, str(sys_act)))
            #random_val = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot)
            goal.add_const(slot, 'dontcare') # ic340 shouldnt this be dontcare instead of adding a random constrain? (or at least do it with a prob) su259: I agree and have changed it to dontcare
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, 'dontcare')))
        else:
            correct_val = goal.get_correct_const_value(slot)
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, correct_val)))
            return

    def _receive_confreq(self, sys_act, goal):
        '''
        confreq(a=x,...,c=z,d): implicit confirm + request d.
        :param sys_act:
        :param goal:
        :return:
        '''
        # Split into confirm(a=x,...c=z) and request(d)
        req_act = DiaAct.DiaAct('request()')
        for item in sys_act.items:
            if item.val is None or item.slot == 'option':
                req_act.append(item.slot, None)

        conf_act = DiaAct.DiaAct('confirm()')
        for item in sys_act.items:
            if item.val is not None and item.slot != 'option':
                conf_act.append(item.slot, item.val, negate=(item.op == '!='))

        # Process the implicit confirm() part. If it leads to any action then ignore the request() part.
        if self._receive_implicit_confirm(conf_act, goal):
            # Implicit confirmed items were ok.
            if self._receive_direct_implicit_confirm(conf_act, goal):
                # Implicit confirmed items were ok. Now process the request.
                self._receive_request(req_act, goal)

    def _receive_negate(self, sys_act, goal):
        self._receive_inform(sys_act, goal)

    def _receive_request(self, sys_act, goal):
        items = sys_act.items
        requested_slot = items[0].slot

        # Check if any options are given.
        if len(items) > 1:
            logger.error('request(a,b,...) is not supported: ' + sys_act)

        '''
        First check if the system has actually already recommended the name of venue.
        If so, check if the user is still trying to get requested info.
        In that case, don't respond to the request (in at least some of the cases)
        but ask for the requested info.
        '''
         
        # Check if there is an unsatisfied request on the goal
        if 'name' in goal.requests and goal.requests['name'] is not None:
            for info in goal.requests:
                if goal.requests[info] is None:
                    self.agenda.push(DiaAct.DiaAct('request(name="%s",%s)' % (goal.requests['name'], info)))
                    return

        '''
        request(venue), request(task), ...
        Check if there is an act of the form request(bar|restaurant|hotel) on the agenda.
        If so, then jus return, otherwise push that act onto the agenda
        '''
        
        if requested_slot in ['venue', 'task', 'type']:
            # Check if there is a suitable response on the agenda (eg. request(bar)).
            if self.agenda.contains_act('request'):
                # :todo: this might be problem. because now any request act on the agenda trigger a return!
                return
            self.agenda.push(DiaAct.DiaAct('inform(type="%s")' % goal.request_type))
            return

        '''
        request(info)
        "Do you know the phone number of the place you are looking for?", etc.
        Just say no.
        '''
       
        if Ontology.global_ontology.is_only_user_requestable(self.dstring, slot=requested_slot):
            self.agenda.push(DiaAct.DiaAct('negate()'))
            return

        '''
        request(hotel), request(bar), request(restaurant)
        This type of system action is not produced by handcrafted DM.
        '''

        '''
        Handle invalid requests that do not match the user goal
        '''
        # Check if the requested slot makes no sense for the user's query,
        # eg. system asks for "food" when user requests "hotel".
        # In this case, re-request the type.
        if not Ontology.global_ontology.is_valid_request(self.dstring, request_type=goal.request_type, slot=requested_slot): 
            self.agenda.push(DiaAct.DiaAct('inform(type="%s")' % goal.request_type))
            return

        '''
        Handle valid requests
        '''
        requested_value = goal.get_correct_const_value(requested_slot)
        answer_slots = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot=requested_slot, value=requested_value)
        
        '''
        Case 1: Requested slot is somewhere on the agenda.
        '''
        # Go through the agenda and locate any corresponding inform() acts.
        # If you find one, move it to the top of the agenda.
        
        logger.debug("CASE1")
        action_taken = False
        inform_items = self.agenda.get_agenda_with_act('inform')
        for agenda_act in inform_items:
            for item in agenda_act.items:
                if item.slot in answer_slots:
                    # Found corresponding inform() on agenda and moving it to top.
                    action_taken = True
                    self.agenda.remove(agenda_act)
                    self.agenda.push(agenda_act)

        if action_taken:
            logger.debug('CASE1')
            return

        '''
        Case 2: Requested slot is not on the agenda, but there is another request() or inform() on the agenda.
        '''
        
        logger.debug("CASE2")
        if not self.answer_req_always:
            if self.agenda.get_agenda_with_act('inform') != [] or self.agenda.get_agenda_with_act('request') != []:
                logger.debug('CASE2')
                return

        '''
        Case 3: There is nothing on the agenda that would suit this request,
                but there is a corresponding constraint in the user goal.
        '''
        logger.debug("CASE3")
        if goal.contains_slot_const(requested_slot):
            logger.debug('CASE3')
            new_act = DiaAct.DiaAct('inform()')
            for slot in answer_slots:
                correct_val = goal.get_correct_const_value(slot)
                if correct_val is not None:
                    new_act.append(slot, correct_val)
                wrong_val = goal.get_correct_const_value(slot, negate=True)
                if wrong_val is not None:
                    new_act.append(slot, wrong_val, negate=True)
            self.agenda.push(new_act)
            return
         
        '''
        Case 4: There is nothing on the agenda or on the user goal.
        '''
        logger.debug('###4 ---- into case 4 --- prob going to say dontcare ...')
        # Either repeat last user request or invent a value for the requested slot.
        f = Settings.random.rand()
        if f < self.rand_decision_probs['NewRequestResp1']:
            # Decided to randomly repeat one of the goal constraints.
            # Go through goal and randomly pick a request to repeat.
            random_val = goal.get_correct_const_value(requested_slot) # copied here from below because random_val was not defined. IS THIS CORRECT?
            if len(goal.constraints) == 0:
                # No constraints on goal: say dontcare.
                self.agenda.push(DiaAct.DiaAct('inform(=dontcare)'))
                goal.add_const(slot=requested_slot,value=random_val)
                goal.add_prev_used(requested_slot,random_val)
                logger.debug('###4.1 just added to goal.prev_slot_values '+ str(requested_slot)+' '+str(random_val))
            else: 
                logger.debug('###4.2')
                sampled_act = Settings.random.choice(goal.constraints)
                sampled_slot, sampled_op, sampled_value = sampled_act.slot, sampled_act.op, sampled_act.val
                self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (sampled_slot, sampled_value)))

        elif f < self.rand_decision_probs['NewRequestResp1'] + self.rand_decision_probs['NewRequestResp2']:
            # Pick a constraint from the list of options and randomly invent a new constraint.
            #random_val = goal.getCorrectValueForAdditionalConstraint(requested_slot) # wrong method from dongho?
            random_val = goal.get_correct_const_value(requested_slot)
            if random_val is None:
                # TODO  Not sure what options is meant to be/do? --> will just reply 'dontcare' for 
                # requests about slots not in goal.
                """
                # None found. Getting a random value.
                if len(options) == 0:
                    random_val = self.domain.getRandomValueForSlot(requested_slot)
                # Get a value from the option list if possible
                else:
                    random_val = Settings.random.choice(options)
                """

                if random_val is None:
                    # Again, none found. Setting it to dontcare.
                    random_val = 'dontcare'
                    self.agenda.push(DiaAct.DiaAct('inform(=%s)' % random_val ))
                    #goal.add_const(slot=requested_slot,value=random_val) #dont think this should be added to constraints
                    goal.add_prev_used(requested_slot,random_val)
                    logger.debug('###4.3 just added to goal.prev_slot_values '+str(requested_slot)+' '+str(random_val))
                else:
                    logger.debug('###4.4')
                    additional_slots = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot=requested_slot, value=random_val)
                    for slot in additional_slots:
                        rval = Ontology.global_ontology.getRandomValueForSlot(self.dstring, slot=slot)
                        self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (slot, rval)))
            else:
                goal.add_const(slot=requested_slot,value=random_val) 
                #goal.constraints[requested_slot] = random_val
                self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (requested_slot, random_val)))
                logger.debug('###4.5 -- havent added anything to prev_slot_values')
    
        else:
            # Decided to say dontcare. 
            logger.debug('###4.6')
            goal.add_const(slot=requested_slot,value='dontcare')
            goal.add_prev_used(requested_slot,'dontcare')
            self.agenda.push(DiaAct.DiaAct('inform(%s="%s")' % (requested_slot, 'dontcare')))

    def _receive_affirm(self, sys_act, goal):
        goal.affirm_pending_confirms()
        self._receive_inform(sys_act, goal)

    def _receive_inform(self, sys_act, goal):
        # Check if the given inform act contains name=none.
        # If so, se the flag RELAX_CONSTRAINTS.
        possible_venue = []
        contains_name_none = False

        logger.debug('Received an inform act. Check if it contains name=none.')
        for item in sys_act.items:
            if item.slot == 'name':
                if item.op == '=' and item.val == 'none':
                    contains_name_none = True
                    self.relax_constraints = True
                    logger.debug('Yes it does. Try to correct or relax the given constraints.')
                elif item.op == '!=':
                    possible_venue.append(item.val)
                    contains_name_none = False
                    self.relax_constraints = False
                else:
                    self.relax_constraints = False

        # Reset requested slots right after the system recommend new venue.
        for item in sys_act.items:
            if item.slot == 'name' and item.op == '=' and item.val != 'none':
                if goal.requests['name'] != item.val:
                    goal.reset_requests()
                    break

        # Check the implicitly confirmed information.
        impl_confirm_ok = self._receive_implicit_confirm(sys_act, goal, False)
        if not impl_confirm_ok:
            logger.debug('The impl confirmed inform was not ok, so stop processing inform act.')
            return

        # If we get this far then all implicitly confirmed constraints were correctly understood.
        # If they don't match an item in the database, however, say bye or try again from beginning.
        sel_venue = None
        if self.use_new_goal_scenarios:
            change_goal = False
            add_name_in_consts = False

            if contains_name_none:
                logger.debug('Definitely change the goal if there is no venue matching the current constraints.')
                change_goal = True
            elif self.first_venue_recommendation:
                self.first_venue_recommendation = False

                # Make a random choice of asking for alternatives,
                # even if the system has recommended another venue.
                f = Settings.random.rand()
                if f < self.rand_decision_probs['ReqAltsAfterVenRec1']:
                    # Ask for alternatives without changing the goal but add a !name in constraints.

                    # Insert name!=venue constraint.
                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)

                    self.agenda.push(DiaAct.DiaAct('reqalts()'))
                    return

                elif f < self.rand_decision_probs['ReqAltsAfterVenRec1'] + self.rand_decision_probs['ReqAltsAfterVenRec2']:
                    # Do change the goal and ask for alternatives.
                    change_goal = True

                else:
                    # Decide not to ask for alternatives nor change the goal at this point.
                    goal.add_name_constraint(sys_act.get_value('name'))
            else:
                # After first venue recommendation we can't ask for alternatives again.
                goal.add_name_constraint(sys_act.get_value('name'))

            if change_goal:
                # Changing the goal.
                if len(goal.constraints) == 0:
                    logger.warning('No constraints available to change.')
                    change_goal = False
                else:
                    # Collect the constraints mentioned by the system act.
                    relax_candidates = []
                    for item in sys_act.items:
                        # Remember relax candidate that has to be set to dontcare.
                        set_dontcare= False
                        if contains_name_none and item.val == 'dontcare' and item.op == '!=':
                            set_dontcare = True
                        # Update candidate list
                        if item.slot not in ['name', 'type'] and\
                                Ontology.global_ontology.is_system_requestable(self.dstring, slot=item.slot) and\
                                item.val not in [None, goal.request_type] and\
                                (item.val != 'dontcare' or item.op == '!='):
                            relax_candidates.append((item.slot, set_dontcare))

                    # Pick a constraint to relax.
                    relax_dontcare = False
                    if len(relax_candidates) > 0:
                        index = Settings.random.randint(len(relax_candidates))
                        (relax_slot, relax_dontcare) = relax_candidates[index]
                    # Randomly pick a goal constraint to relax
                    else:
                        index = Settings.random.randint(len(goal.constraints))
                        relax_const = goal.constraints[index]
                        relax_slot = relax_const.slot

                    # Randomly decide whether to change it to another value or set it to 'dontcare'.
                    if relax_slot is not None:
                        #if type(relax_slot) not in [unicode, str]:
                        #    print relax_slot
                        #    logger.error('Invalid relax_slot type: %s in %s' % (type(relax_slot), relax_slot))
                        logger.debug('Relaxing constraint: ' + relax_slot)
                        if goal.contains_slot_const('name'):
                            goal.remove_slot_const('name')

                        # DEBUG THIS SECCTION:
                        logger.debug("choosen slot to relax constraints in: "+relax_slot)
                        logger.debug("current goal: "+str(goal))
                        logger.debug("current goal.prev_slot_values: "+str(goal.prev_slot_values))

                        if Settings.random.rand() < self.rand_decision_probs['ConstraintRelax'] or relax_dontcare:
                            logger.debug("--case0--")
                            # Just set it to dontcare.
                            relax_value = 'dontcare'
                        elif relax_slot not in goal.prev_slot_values:
                            logger.debug("--case1--")
                            # TODO - check this - added this elif as all domains bar CamRestaurants were crashing here
                            relax_value = 'dontcare'
                            goal.add_prev_used(relax_slot, relax_value) # is this necessary?
                        else:
                            logger.debug("--case2--")
                            # Set it to a valid value for this slot that is different from the previous one.
                            relax_value = Ontology.global_ontology.getRandomValueForSlot(self.dstring,
                                                                                        slot=relax_slot, 
                                                                                        nodontcare=True,
                                                                                        notthese=goal.prev_slot_values[relax_slot])
                            goal.add_prev_used(relax_slot, relax_value)
                            logger.debug("relax value to "+relax_value)

                        goal.replace_const(relax_slot, relax_value)

                        # Randomly decide whether to tell the system about the change or just request an alternative.
                        if not contains_name_none:
                            if Settings.random.rand() < self.rand_decision_probs['TellAboutChange']:
                                # Decide to tell the system about it.
                                self.agenda.push(DiaAct.DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                            else:
                                # Decide not to tell the system about it.
                                # If the new goal constraint value is set to something other than dontcare,
                                # then add the slot to the list of requests, so that the user asks about it
                                # at some point in the dialogue.
                                # If it is set to dontcare, add name!=value into constraint set.
                                self.agenda.push(DiaAct.DiaAct('reqalts()'))
                                if relax_value == 'dontcare':
                                    goal.add_name_constraint(sys_act.get_value('name'), negate=True)
                                else:
                                    goal.requests[relax_slot] = None
                        else:
                            # After inform(name=none,...) always tell the system about the goal change.
                            self.agenda.push(DiaAct.DiaAct('reqalts(%s="%s")' % (relax_slot, relax_value)))
                        return

                    else:
                        # No constraint to relax.
                        change_goal = False

            else: # change_goal == False
                # If name=none, ..., name!=x, ...
                if len(possible_venue) > 0:
                    # If # of possible venues is same to the number of name!=value constraints,
                    # that is, all possible venues are excluded by name!=value constraints.
                    # The user must relax them.
                    is_there_possible_venue = False
                    for venue in possible_venue:
                        if goal.is_satisfy_all_consts(dact.DactItem('name', '=', venue)):
                            is_there_possible_venue = True

                    if not is_there_possible_venue:
                        goal.remove_slot_const('name')

                    # Remove possible venues violating name constraints.
                    copy_possible_venue = copy.copy(possible_venue)
                    for venue in copy_possible_venue:
                        if not goal.is_satisfy_all_consts(dact.DactItem('name', '=', venue)):
                            possible_venue.remove(venue)

                    # 1) Choose venue from possible_venue, which satisfy the constraints.
                    sel_venue = Settings.random.choice(possible_venue)

                    # 2) Relax appropriate constraint from goal.
                    for cslot in copy.deepcopy(goal.constraints):
                        if not sys_act.contains_slot(cslot.slot):
                            # Constraint not found in system act: relax it.
                            goal.replace_const(cslot.slot, 'dontcare')

                            # Also remove any informs about this constraint from the agenda.
                            self.agenda.filter_acts_slot(cslot.slot)


        # Endif self.user_new_goal_scenarios == True
        if self.relax_constraints:
            # The given constraints were understood correctly but did not match a venue.
            if Settings.random.rand() < self.rand_decision_probs['ByeOrStartOver']:
                self.agenda.clear()
                self.agenda.push(DiaAct.DiaAct('bye()'))
            else:
                #self.agenda.push(DiaAct.DiaAct('inform(type=restaurant)'))
                self.agenda.push(DiaAct.DiaAct('inform(type=%s)' % goal.request_type ))
            return

        '''
        If we get this far then all implicitly confirmed constraints are correct.
        Use the given information to fill goal request slots.
        '''
        for slot in goal.requests:
            if slot == 'name' and sel_venue is not None:
                goal.requests[slot] = sel_venue
            else:
                for item in sys_act.items:
                    if item.slot == slot:
                        if item.op != '!=':
                            goal.requests[slot] = item.val
                        #if item.slot == 'name' and goal.nturns_to_first_recommendation == -1:
                        #    goal.nturns_to_first_recommendation = goal.nturns

        '''
        With some probability, change any remaining inform acts on the agenda to confirm acts.
        '''
        if Settings.random.rand() < self.rand_decision_probs['InformToConfirm']:
            for agenda_item in self.agenda.agenda_items:
                if agenda_item.act == 'inform':
                    if len(agenda_item.items) == 0:
                        logger.error('Empty inform act found on agenda.')
                    elif agenda_item.items[0].val != 'dontcare':
                        agenda_item.act = 'confirm'

        # Randomly decide to respond with thankyou() or ack(), or continue.
        if self.use_new_goal_scenarios:
            f = Settings.random.rand()
            if f < self.rand_decision_probs['ThankAck1']:
                self.agenda.push(DiaAct.DiaAct('thankyou()'))
                return
            elif f < self.rand_decision_probs['ThankAck1'] + self.rand_decision_probs['ThankAck2']:
                self.agenda.push(DiaAct.DiaAct('ack()'))
                return

        '''
        If empty slots remain in the goal, put a corresponding request for the first empty slot onto the agenda.
        If there are still pending acts on the agenda apart from bye(), though, then process those first,
        at least sometimes.
        '''
        if self.agenda.size() > 1:
            if Settings.random.rand() < self.rand_decision_probs['DealWithPending']:
                return

        # If empty goal slots remain, put a request on the agenda.
        if not goal.are_all_requests_filled():
            # Specify name in case the system giving complete list of venues matching constraints
            # inform(name=none, ..., name!=x, name!=y, ...)
            user_response = DiaAct.DiaAct('request()')
            if sel_venue is not None:
                # If user picked venue from multiple venues, it needs to specify selected name in request act.
                # If only one possible venue was offered, it specifies the name with some probability.
                user_response.append('name', sel_venue)

            one_added = False
            for slot in goal.requests:
                value = goal.requests[slot]
                if value is None:
                    if not one_added:
                        user_response.append(slot, None)
                        one_added = True
                    else:
                        # Add another request with some probability
                        if Settings.random.rand() < self.rand_decision_probs['AddSlotToReq']:
                            user_response.append(slot, None)
                        else:
                            break

            self.agenda.push(user_response)

    def _receive_direct_implicit_confirm(self, sys_act, goal):
        '''
        Deals with implicitly confirmed items that are not on the user goal.
        These are okay in system inform(), but not in system confirm() or confreq().
        In this case, the user should mention that slot=dontcare.
        :param sys_act:
        :param goal:
        :return:
        '''
        for item in sys_act.items:
            slot = item.slot
            val = item.val
            if slot in ['count', 'option', 'type']:
                continue
            if not goal.contains_slot_const(slot) and val != 'dontcare':
                if not Ontology.global_ontology.is_implied(self.dstring, slot=slot, value=val):
                    self.agenda.push(DiaAct.DiaAct('negate(%s="dontcare")' % slot))
                    return False

        # Explicit confirmations okay.
        return True

    def _receive_implicit_confirm(self, sys_act, goal, fromconfirm=True):
        '''
        This method is used for checking implicitly confirmed items.
        :param sys_act:
        :param goal:
        :return: True if all the items are consistent with the user goal.
                 If there is a mismatch, then appropriate items are added to the agenda and
                 the method returns False.
        '''
        contains_name_none = False
        contains_count = False
        is_inform_requested = False
        #contains_options = False

        # First store all possible values for each unique slot in the list of items and check for name=none
        slot_values = {}
        informable_s = Ontology.global_ontology.get_informable_slots(self.dstring)
        requestable_s = Ontology.global_ontology.get_requestable_slots(self.dstring)
        for item in sys_act.items:
            if item.slot != 'count' and item.slot != 'option':
                # if item.slot in slot_values:
                #     # If rewrite, raise error.
                #     logger.error('Slot value is being rewritten: %s' + str(item))
                if item.slot not in slot_values:
                    slot_values[item.slot] = set()
                slot_values[item.slot].add(item)
            elif item.slot == 'count':
                contains_count = True
                count = item.val
            elif item.slot == 'option':
                logger.error('option in dialog act is not supported.')
                #contains_options = True

            if item.slot == 'name' and item.val == 'none':
                contains_name_none = True

            if item.slot in requestable_s and item.slot not in informable_s:
                is_inform_requested = True



        # Check if all the implicitly given information is correct. Otherwise reply with negate or deny.
        do_exp_confirm = False
        for item in sys_act.items:
            correct_val = None
            correct_slot = None
            do_correct_misunderstanding = False
            # negation = (item.op == '!=')

            if item.slot in ['count', 'option']:
                continue

            # Exclude slots that are info keys in the ontology, go straight to the next item.
            if Ontology.global_ontology.is_only_user_requestable(self.dstring, slot=item.slot) and\
                    not goal.contains_slot_const(item.slot):
                continue

            # If an implicitly confirmed slot is not present in the goal,
            # it doesn't really matter, unless:
            # a) the system claims that there is no matching venue, or
            # b) the system explicitly confirmed this slot.
            if not goal.contains_slot_const(item.slot):
                logger.debug('%s=%s is not on user goal.' % (item.slot, item.val))

                # If slot-value conflicts with the user goal.
                if not goal.is_satisfy_all_consts(slot_values[item.slot]):
                    logger.warning('Not implemented.')

            else:
                correct_val = goal.get_correct_const_value(item.slot)
                wrong_val = goal.get_correct_const_value(item.slot, negate=True)

                # Current negation: if the goal is slot!=value
                if correct_val is not None and wrong_val is not None and correct_val == wrong_val:
                    logger.debug(str(goal))
                    logger.error('Not possible')
                    
                # Conflict between slot!=value on user goal and slot=value in system act.
                if wrong_val is not None and not goal.is_satisfy_all_consts(slot_values[item.slot]):
                    if contains_name_none:
                        if item.slot == 'name':
                            # Relax constraint for !name because of name=none.
                            goal.remove_slot_const('name', negate = True) # su259: negate added; only remove slot!=value constraints
                        else:
                            # Must correct it because of name=none.
                            do_correct_misunderstanding = True
                    # System informed wrong venue.
                    else:
                        do_correct_misunderstanding = True
                        
                # ic340: Exclude the name from the confirmation. This makes the simulated user always accept a new informed
                # venue even if another venue has been informed before. This is done to fix the bug with the simulated
                # user becoming "obsessed" with a wrong venue. However, I dont know if this fix could introduce further bugs
                # (e.g. user gets informed the telephone of a different restaurant and accepts it as correct)
                # su259: moved here and added condition on do_correct_misunderstanding so that having a name in the sys act
                # other than inform_requested which is forbidden by the goal triggers a correction instead of skipping it
                if item.slot == 'name' and not is_inform_requested and not do_correct_misunderstanding:
                    continue
                    #pass


                # Conflict between slot=value on user goal, and slot=other or slot!=other in system act
                if correct_val is not None and not goal.is_satisfy_all_consts(slot_values[item.slot]):
                    # If the system act contains name=none, then correct the misunderstanding in any case.
                    if contains_name_none:
                        if item.slot != 'name':
                            do_correct_misunderstanding = True
                    # If it doesn't, then only correct the misunderstanding if the user goal constraints say so
                    elif correct_val != 'dontcare':
                        do_correct_misunderstanding = True
                        
                
            # If all constraints mentioned by user but some are not implicitly confirmed in system act,
            # confirm one such constraint with some probability
            planned_response_act = None
            if not contains_name_none:
                # Now find constraints not mentioned in system act, if any
                confirmable_consts = []
                for const in goal.constraints:
                    slots_to_confirm = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot=const.slot, value=const.val)
                    for sit in slots_to_confirm:
                        s1 = sit

                        # Check if the system act contains the slot value pair (const_slot and its value)
                        found = False
                        v1 = goal.get_correct_const_value(s1)
                        if sys_act.contains(s1, v1):
                            found = True
                        v1_neg = goal.get_correct_const_value(s1, negate=True)
                        if v1_neg is not None and sys_act.contains_slot(s1) and sys_act.get_value(s1) != v1_neg:
                            found = True

                        # found = None
                        # for sysits_it in sys_act.items:
                        #     if sysits_it.slot == s1 and (sysits_it.val == v1 or \
                        #                                  (sysits_it.val != v1 and '!' in v1)):
                        #         found = sysits_it

                        if not found and const.val not in ['dontcare','none','**NONE**']:
                            confirmable_consts.append(const)

                # Now pick a constraint to confirm
                if len(confirmable_consts) > 0:
                    rci = Settings.random.choice(confirmable_consts)
                    planned_response_act = DiaAct.DiaAct('confirm()')
                    planned_response_act.append(rci.slot, rci.val)

                    # su259: unclear why in the following the goal is gone through again, constraint slot-value from goal
                    # already collected in previous loop. Hence, simplifying this to the above code. 
                    #
                    # slots_to_exp = Ontology.global_ontology.getSlotsToExpress(self.dstring, slot=rci.slot, value=rci.val)
                    # planned_response_act = DiaAct.DiaAct('confirm()')
                    # for strit in slots_to_exp:
                    #     v1 = goal.get_correct_const_value(strit)
                    #     if v1 is not None: planned_response_act.append(strit, v1)
                    #     v1 = goal.get_correct_const_value(strit, negate=True)
                    #     if v1 is not None: planned_response_act.append(strit, v1, negate=True)

            if do_correct_misunderstanding:
                logger.debug('Correct misunderstanding for slot %s' % item.slot)
                # Depending on the patience level, say bye with some probability (quadratic function of patience level)
                if not self.patience_old_style:
                    prob1 = float(goal.patience ** 2) / self.max_patience ** 2
                    prob2 = float(2*goal.patience) / self.max_patience
                    prob = prob1 - prob2 + 1
                    if Settings.random.rand() < prob:
                        # Randomly decided to give up
                        self.agenda.clear()
                        self.agenda.push(DiaAct.DiaAct('bye()'))
                        return False

                # Pushing negate or deny onto agenda to correct misunderstanding.
                # Make a random decision as to whether say negate(a=y) or deny(a=y,a=z), or
                # confirm a constraint not mentioned in system act.
                # If the type is wrong, say request(whatever).

                if item.slot != 'type':
                    correct_it = False
                    if planned_response_act is None:
                        correct_it = True
                    else:
                        if Settings.random.rand() >= self.rand_decision_probs['OverruleCorrection']:
                            # Decided to correct the system.
                            correct_it = True

                    if correct_it:
                        if correct_val is None:
                            correct_val = 'dontcare'

                        cslot = item.slot
                        if correct_slot is not None:
                            cslot = correct_slot

                        planned_response_act = None
                        if wrong_val is not None:
                            planned_response_act = DiaAct.DiaAct('inform(%s!="%s")' % (cslot, wrong_val))
                            # planned_response_act = DiaAct.DiaAct('negate(%s="%s")' % (cslot, correct_val))
                        else:
                            f = Settings.random.rand()
                            if f < self.rand_decision_probs['CorrectingAct1']:
                                planned_response_act = DiaAct.DiaAct('negate(%s="%s")' % (cslot, correct_val))
                            elif f < self.rand_decision_probs['CorrectingAct1'] + self.rand_decision_probs['CorrectingAct2']:
                                planned_response_act = DiaAct.DiaAct('deny(%s="%s",%s="%s")' % (item.slot, item.val,
                                                                                               cslot, correct_val))
                            else:
                                planned_response_act = DiaAct.DiaAct('inform(%s="%s")' % (cslot, correct_val))

                else:
                    planned_response_act = DiaAct.DiaAct('inform(type=%s)' % goal.request_type )

                self.agenda.push(planned_response_act)

                # Resetting goal request slots.
                goal.reset_requests()
                return False

            # The system's understanding is correct so far, but with some changes,
            # the user decide to confirm a random constraints.
            elif planned_response_act is not None and not do_exp_confirm and not fromconfirm:
                if Settings.random.rand() < self.rand_decision_probs['ConfirmRandomConstr']:
                    # Decided to confirm a random constraint.
                    self.agenda.push(planned_response_act)
                    do_exp_confirm = True

            elif contains_name_none:
                # No correction required in case of name=none: set goal status systemHasInformedNameNone=True
                goal.system_has_informed_name_none = True

        # The user decide to confirm a random constraints.
        if do_exp_confirm:
            goal.fill_requests(sys_act.items)
            return False

        # Implicit confirmations okay.
        return True
    
    '''def _sampleProbs(self):
        self.rand_decision_probs['AffirmCombination'] = Settings.random.rand()
        self.rand_decision_probs['InformCombination'] = Settings.random.rand()
        if self.old_style_parameter_sampling:
            self.max_patience = Settings.random.randint(2,10)'''

    def _sampleProbs(self):
        for key in self.sampling_probs:
            if type(self.sampling_probs[key]) is list:
                self.rand_decision_probs[key] = Settings.random.uniform(self.sampling_probs[key][0], self.sampling_probs[key][1])

    def _normalise_act_no_rules(self, dap):
        #logger.debug(str(dap))
        norm_act = copy.deepcopy(dap)
        norm_act.items = []

        for item in dap.items:
            keep_it = True
            val = item.val
            slot = item.slot

            if slot == 'task':
                keep_it = False
            elif dap.act == 'request' and val is None:
                if slot == 'name':
                    keep_it = False
                    if val is None:
                        norm_act.act = 'inform'
                elif slot == 'bar' or slot == 'restaurant' or slot == 'hotel':
                    norm_act.append('type', slot)
                    keep_it = False
            elif slot is None and val is not None and val != 'dontcare':
                keep_it = False
                norm_act.append('type', val)

            if keep_it:
                norm_act.append(slot, val)

        #logger.debug(str(norm_act))
        return norm_act
    
    def _check_config_file_path(self, cfpath):
        # check if file path points to an existing file. if not, try searching for file relative to root
        if not os.path.isfile(cfpath):
            cfpath = os.path.join(Settings.root,cfpath)
            if not os.path.isfile(cfpath):
                logger.error('Error model config file "{}" does not exist'.format(cfpath))
        return cfpath
    
    def _read_UM_config(self, config_file_path):
        with open(config_file_path, 'r') as config_file:
            for line in config_file:
                if not line.startswith('#'):
                    if 'InformCombination' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['InformCombination'] = val
                    elif 'AddSlotToReq' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['AddSlotToReq'] = val
                    elif 'NoAfterReqmore' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['NoAfterReqmore'] = val
                    elif 'YesAfterReqmore' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['YesAfterReqmore'] = val
                    elif 'Greeting' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['Greeting'] = val
                    elif 'ConstraintRelax' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ConstraintRelax'] = val
                    elif 'TellAboutChange' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['TellAboutChange'] = val
                    elif 'ByeOrStartOver' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ByeOrStartOver'] = val
                    elif 'DealWithPending' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['DealWithPending'] = val
                    elif 'AddVenueNameToRequest' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['AddVenueNameToRequest'] = val
                    elif 'NoSlotWithDontcare' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['NoSlotWithDontcare'] = val
                    elif 'Repeat' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['Repeat'] = val
                    elif 'InformToConfirm' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['InformToConfirm'] = val
                    elif 'CombAffirmWithAgdItem' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['CombAffirmWithAgdItem'] = val
                    elif 'NullResp' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['NullResp'] = val
                    elif 'OverruleCorrection' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['OverruleCorrection'] = val
                    elif 'ConfirmRandomConstr' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ConfirmRandomConstr'] = val
                    elif 'ReqAltsAfterVenRec1' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ReqAltsAfterVenRec1'] = val
                    elif 'ReqAltsAfterVenRec2' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ReqAltsAfterVenRec2'] = val
                    elif 'NewRequestResp1' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['NewRequestResp1'] = val
                    elif 'NewRequestResp2' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['NewRequestResp2'] = val
                    elif 'CorrectingAct1' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['CorrectingAct1'] = val
                    elif 'CorrectingAct2' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['CorrectingAct2'] = val
                    elif 'ThankAck1' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ThankAck1'] = val
                    elif 'ThankAck2' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['ThankAck2'] = val
                    elif 'AffirmCombination' in line:
                        val = line.split('#')[0].split('=')[-1].strip()
                        if '[' in val and ']' in val:
                            val = [float(x) for x in val.replace('[', '').replace(']', '').split(',')]
                        else:
                            val = float(val)
                        self.rand_decision_probs['AffirmCombination'] = val

    

#END OF FILE
