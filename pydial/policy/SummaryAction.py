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
SummaryAction.py - Mapping between summary and master actions
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017, 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.SummaryUtils` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.Settings`

************************

'''


__author__ = "cued_dialogue_systems_group"

import sys
import SummaryUtils
from utils import ContextLogger,Settings
from ontology import Ontology
logger = ContextLogger.getLogger('')

MAX_NUM_ACCEPTED = 10


class SummaryAction(object):
    '''
    The summary action class encapsulates the functionality of a summary action along with the conversion from summary to master actions.
    
    .. Note::
        The list of all possible summary actions are defined in this class.
    '''
    def __init__(self, domainString, empty=False, confreq=False):
        '''
        Records what domain the class is instantiated for, and what actions are available

        :param domainString: (string) domain tag
        :param empty: (bool)
        :param confreq: (bool) representing if the action confreq is used
        '''

        self.domainString = domainString 
        self.action_names = []
        self._array_slot_summary = None
        self._global_summary = None

        self.inform_mask = True
        if Settings.config.has_option("summaryacts", "informmask"):
            self.inform_mask = Settings.config.getboolean('summaryacts', 'informmask')
        self.inform_count_accepted = 4
        if Settings.config.has_option("summaryacts", "informcountaccepted"):
            self.inform_count_accepted = Settings.config.getint('summaryacts', 'informcountaccepted')
        elif Settings.config.has_option("goalgenerator", "maxconstraints"):
            self.inform_count_accepted = Settings.config.getint('goalgenerator', 'maxconstraints') + 1
        self.request_mask = True
        if Settings.config.has_option("summaryacts", "requestmask"):
            self.request_mask = Settings.config.getboolean('summaryacts', 'requestmask')
        self.bye_mask = True
        if Settings.config.has_option("summaryacts", "byemask"):
            self.request_mask = Settings.config.getboolean('summaryacts', 'byemask')

        if not empty:
            for slot in Ontology.global_ontology.get_system_requestable_slots(domainString):
                self.action_names.append("request_" + slot)
                self.action_names.append("confirm_" + slot)
                self.action_names.append("select_" + slot)
                if confreq:
                    for slot2 in Ontology.global_ontology.get_system_requestable_slots(domainString):
                        self.action_names.append("confreq_" + slot + "_" + slot2)
            self.action_names += [ "inform",
                                   "inform_byname",
                                   "inform_alternatives",
                                   "bye",
                                   "repeat",
                                   "reqmore",
                                   "restart"
                                 ]
        self.reset()

    def reset(self):
        self.alternatives_requested = False

    def Convert(self, belief, action, lastSystemAction):
        '''
        Converts the given summary action into a master action based on the current belief and the last system action.

        :param belief: (dict) the current master belief
        :param action: (string) the summary action to be converted to master action
        :param lastSystemAction: (string) the system action of the previous turn
        '''

        self._array_slot_summary = SummaryUtils.arraySlotSummary(belief, self.domainString)
        self._global_summary = SummaryUtils.globalSummary(belief, self.domainString)
        logger.dial('system summary act: {}.'.format(action))

        if action == "inform":
            output = self.getInformByConstraints(belief)
        elif "request_" in action:
            output = self.getRequest(action.split("_")[1])
        elif "select_" in action:
            output = self.getSelect(action.split("_")[1])
        elif "confirm_" in action:
            output = self.getConfirm(action.split("_")[1])
        elif "confreq_" in action:
            output = self.getConfReq(action.split("_")[1], action.split("_")[2])
        elif action == "inform_byname":
            output = self.getInformByName(belief)
        elif action == "inform_alternatives":
            output = self.getInformAlternatives(belief)
        elif action == "bye":
            output = self.getBye()
        elif action == "repeat":
            output = lastSystemAction
        elif action == "reqmore":
            output = self.getReqMore()
        elif action == "restart":
            output = self.getRestart()
        else:
            output = ""
            logger.error("Unknown action: " + action)
        return output

    # MASK OVER SUMMARY ACTION SET
    # ------------------------------------------------------------------------------------

    def getNonExecutable(self, belief, lastSystemAction):
        '''
        Set of rules defining the mask over the action set, given the current belief state
        :param belief: (dict) the current master belief
        :param lastSystemAction: (string) the system action of the previous turn
        :return: list of non-executable (masked) actions
        '''

        array_slot_summary = SummaryUtils.arraySlotSummary(belief, self.domainString)
        global_summary = SummaryUtils.globalSummary(belief, self.domainString)
        if global_summary['GLOBAL_BYALTERNATIVES'] and not global_summary['GLOBAL_THANKYOU'] and not global_summary['GLOBAL_ACK']:
            self.alternatives_requested = True

        nonexec = []

        for action in self.action_names:
            mask_action = False

            if action == "inform":
                acceptance_list = SummaryUtils.getTopBeliefs(belief, domainString=self.domainString)
                discriminable = SummaryUtils.acceptanceListCanBeDiscriminated(acceptance_list,
                                                                                                     self.domainString)
                if not global_summary['GLOBAL_BYCONSTRAINTS']:
                    mask_action = True
                if global_summary['GLOBAL_COUNTACCEPTED'] < self.inform_count_accepted and discriminable:
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif action == "inform_byname":
                if not global_summary['GLOBAL_BYNAME']:
                    mask_action = True
                if belief['features']['lastInformedVenue'] == '' \
                        and SummaryUtils.getTopBelief(belief['beliefs']['name'])[0] == '**NONE**' :
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif action == "inform_alternatives":
                if not self.alternatives_requested:
                    mask_action = True
                if belief['features']['lastInformedVenue'] == '':
                    mask_action = True
                if mask_action and self.inform_mask:
                    nonexec.append(action)

            elif action == "bye":
                if not global_summary['GLOBAL_FINISHED']:
                    mask_action = True
                if mask_action and self.bye_mask:
                    nonexec.append(action)

            elif action == "repeat":
                if not global_summary['GLOBAL_REPEAT'] or lastSystemAction is None:
                    mask_action = True
                mask_action = True  # ic340: this action is "deactivated" because simuser doesnt know how to react to it
                if mask_action:
                    nonexec.append(action)

            elif action == "reqmore":
                if belief['features']['lastInformedVenue'] == '':
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

            elif action == "restart":
                if not global_summary['GLOBAL_RESTART']:
                    mask_action = True
                mask_action = True  # ic340: this action is "deactivated" because simuser doesnt know how to react to it
                if mask_action:
                    nonexec.append(action)

            elif "request_" in action:
                pass
                if mask_action and self.request_mask:
                    nonexec.append(action)

            elif "select_" in action:
                slot_summary = array_slot_summary[action.split("_")[1]]
                top_prob = slot_summary['TOPHYPS'][0][1]
                sec_prob = slot_summary['TOPHYPS'][1][1]
                if top_prob == 0 or sec_prob == 0:
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

            elif "confirm_" in action:
                slot_summary = array_slot_summary[action.split("_")[1]]
                top_prob = slot_summary['TOPHYPS'][0][1]
                if top_prob == 0:
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

            elif "confreq_" in action:
                slot_summary = array_slot_summary[action.split("_")[1]]
                top_prob = slot_summary['TOPHYPS'][0][1]
                if top_prob == 0:
                    mask_action = True
                if mask_action and self.request_mask:
                    nonexec.append(action)

        logger.info('masked inform actions:' + str([act for act in nonexec if 'inform' in act]))
        return nonexec

    # added by phs26, 4 Nov 2016
    def getExecutableMask(self, belief, lastSystemAction):
        '''
        '''
        """
        # hack, make every action executable
        return [0.0] * len(self.action_names)
        """

        execMask = []
        nonExec = self.getNonExecutable(belief.getDomainState(belief.currentdomain), lastSystemAction)
        for action in self.action_names:
            if action in nonExec:
                # execMask.append(-sys.maxint)
                execMask.append(-25.0)
            else:
                execMask.append(0.0)

        return execMask


    # CONVERTING METHODS FOR EACH SPECIFIC ACT:
    #------------------------------------------------------------------------------------
    
    def getRequest(self, slot):
        return 'request({})'.format(slot)

    def getConfirm(self, slot):
        summary = self._array_slot_summary[slot]
        top_value = summary['TOPHYPS'][0][0]
        return 'confirm({}="{}")'.format(slot, top_value)
    
    def getConfReq(self, cslot, rslot):
        summary = self._array_slot_summary[cslot]
        top_value = summary['TOPHYPS'][0][0]
        return 'confreq({}="{}",{})'.format(cslot, top_value, rslot)

    def getSelect(self, slot):
        summary = self._array_slot_summary[slot]
        top_value = summary['TOPHYPS'][0][0]
        sec_value = summary['TOPHYPS'][1][0]
        return 'select({}="{}",{}="{}")'.format(slot, top_value, slot, sec_value)

    def getInformByConstraints(self, belief):
        accepted_values = SummaryUtils.getTopBeliefs(belief, domainString=self.domainString)
        constraints = SummaryUtils.get_constraints(accepted_values)
        return SummaryUtils.getInformByConstraints(constraints, self.domainString, belief['features']['lastInformedVenue'])

    def getInformByName(self, belief):
        requested_slots = SummaryUtils.getRequestedSlots(belief)
        name = SummaryUtils.getTopBelief(belief['beliefs']['name'])[0]
        if name == '**NONE**':
            name = belief['features']['lastInformedVenue']
        return SummaryUtils.getInformRequestedSlots(requested_slots, name, self.domainString)

    def getInformAlternatives(self, belief):
        self.alternatives_requested = False
        informedVenueSinceNone = set(belief['features']['informedVenueSinceNone'])
        accepted_values = SummaryUtils.getTopBeliefs(belief, domainString=self.domainString)
        return SummaryUtils.getInformAlternativeEntities(accepted_values, informedVenueSinceNone, self.domainString)

    def getBye(self):
        return 'bye()'

    def getReqMore(self):
        return 'reqmore()'

    def getInformRepeat(self):
        #TODO: implement the proper action, this was not implemented in PolicyUtils.py
        return 'null()'

    def getRestart(self):
        # TODO: implement the proper action, this was not implemented in PolicyUtils.py
        return 'null()'

#END OF FILE
