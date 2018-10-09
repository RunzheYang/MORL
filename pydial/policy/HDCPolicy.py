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
HDCPolicy.py - Handcrafted dialogue manager
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`policy.Policy` |.|
    import :mod:`policy.PolicyUtils` |.|
    import :mod:`policy.SummaryUtils` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"
import copy

import Policy
import PolicyUtils
import SummaryUtils
from utils import ContextLogger, Settings
from ontology import Ontology
logger = ContextLogger.getLogger('')

MAX_NUM_ACCEPTED = 10
ACCEPT_PROB = 0.8


class HDCPolicy(Policy.Policy):
    """
    Handcrafted policy derives from Policy base class. Based on the slots defined in the ontology and fix thresholds, defines a rule-based policy. 
    
    If no info is provided by the user, the system will always ask for the slot information in the same order based on the ontology. 
    """
    def __init__(self, domainString):
        """
        Handcrafted policy constructor.
        """
        super(HDCPolicy, self).__init__(domainString) # inherited from Policy.Policy() is self.domainString
        
        self.use_confreq = False
        
        if Settings.config.has_option('policy', 'useconfreq'):
            self.use_confreq = Settings.config.getboolean('policy', 'useconfreq')
        if Settings.config.has_option('policy_'+domainString, 'useconfreq'):
            self.use_confreq = Settings.config.getboolean('policy_'+domainString, 'useconfreq')
        
        inpolicyfile = ''
        if Settings.config.has_option('policy', 'inpolicyfile'):
            inpolicyfile = Settings.config.get('policy', 'inpolicyfile')
        if Settings.config.has_option('policy_'+domainString, 'inpolicyfile'):
            inpolicyfile = Settings.config.get('policy_'+domainString, 'inpolicyfile')
        if inpolicyfile == '':
            msg = 'Policy file is given: {}, but policy type is set to hdc.'.format(inpolicyfile)
            msg += ' Ignoring the given policy file and using hdc policy.'
            logger.warning(msg)

        self.disableLowProbAct = False  #TODO - can make this a config variable if actually used

        logger.debug("numActions = "+str(self.numActions))

        self.restart()
        

    def restart(self):
        super(HDCPolicy,self).restart()
        
    def nextAction(self, belief):
        """Primary response function of HDC policy - hands off control to entity-retrieval policy.
        """
        global_summary = SummaryUtils.globalSummary(belief, domainString=self.domainString)
        return self.work_entity_retrieval(belief, global_summary)
        
    def work_entity_retrieval(self, belief, global_summary):
        '''
        '''
        array_slot_summary = SummaryUtils.arraySlotSummary(belief, self.domainString)
        logger.debug(str(global_summary))
        logger.debug('HDC policy: getGlobal') 
        done, output = self._getGlobal(belief, global_summary)
        
        if not done:
            logger.debug('HDC policy: getConfirmSelect')
            done, output = self._getConfirmSelect(belief, array_slot_summary)
        if not done:
            logger.debug('HDC policy: getInform')
            inform_summary = []
            for num_accepted in range(1, MAX_NUM_ACCEPTED+1):
                temp = SummaryUtils.actionSpecificInformSummary(belief, num_accepted, self.domainString)
                inform_summary.append(temp)
                       
            done, output = self._getInform(belief, global_summary, inform_summary)
        if not done:
            logger.debug('HDC policy: getRequest')
            done, output = self._getRequest(belief, array_slot_summary)
        if not done:
            logger.warning("HDCPolicy couldn't find action: execute reqmore().")
            output = 'reqmore()'

        if output == 'badact()' or output == 'null()':
            logger.warning('HDCPolicy chose bad or null action')
            output = 'null()'

        if self.use_confreq:
            #TODO - known problem here if use_confreq is True (ie being used)  FIXME
            output = PolicyUtils.add_venue_count(output, belief)
        return output

    def _getGlobal(self, belief, global_summary):
        '''Note - this function seems a little odd - compares booleans to 0.5 - Not sure if Dongho had a different\
        format in mind for global_summary? djv27 
        '''
        act = 'null()'

        if global_summary['GLOBAL_BYCONSTRAINTS'] > 0.5 and global_summary['GLOBAL_COUNTACCEPTED'] > 3:
            act = PolicyUtils.getGlobalAction(belief, 'INFORM_BYNAME', domainString=self.domainString)
        elif global_summary['GLOBAL_BYALTERNATIVES'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'INFORM_ALTERNATIVES', domainString=self.domainString)
        elif global_summary['GLOBAL_BYNAME'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'INFORM_REQUESTED', domainString=self.domainString)
        elif global_summary['GLOBAL_FINISHED'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'BYE', domainString=self.domainString)
        elif global_summary['GLOBAL_REPEAT'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'INFORM_REPEAT', domainString=self.domainString)
        elif global_summary['GLOBAL_REQMORE'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'INFORM_BYNAME', domainString=self.domainString)
        elif global_summary['GLOBAL_THANKYOU'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'REQMORE', domainString=self.domainString)
        elif global_summary['GLOBAL_ACK'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'REQMORE', domainString=self.domainString)
        elif global_summary['GLOBAL_RESTART'] > 0.5:
            act = PolicyUtils.getGlobalAction(belief, 'RESTART', domainString=self.domainString)

        if act != 'null()':
            return True, act
        return False, act

    def _getConfirmSelect(self, belief, array_slot_summary):
        for slot in Ontology.global_ontology.get_sorted_system_requestable_slots(self.domainString):  
            summary = array_slot_summary[slot]
            (top_value, top_prob) = summary['TOPHYPS'][0]
            (sec_value, sec_prob) = summary['TOPHYPS'][1]
            if top_prob < 0.8:
                if top_prob > 0.6:
                    # Confirm
                    return True, 'confirm(%s="%s")' % (slot, top_value)
                elif top_prob > 0.3:
                    if top_prob - sec_prob < 0.2:
                        # Select
                        return True, 'select(%s="%s",%s="%s")' % (slot, top_value, slot, sec_value)
                    else:
                        # Confirm
                        return True, 'confirm(%s="%s")' % (slot, top_value)

        return False, 'null()'

    def _getInform(self, belief, global_summary, inform_summary):
        act = 'null()'

        count80 = global_summary['GLOBAL_COUNTACCEPTED']
        offer_happened = global_summary['GLOBAL_OFFERHAPPENED']

        if count80 >= MAX_NUM_ACCEPTED:
            count80 = MAX_NUM_ACCEPTED - 1

        arr = inform_summary[count80]
        first = arr[0]  # True if there is no matching entities
        second = arr[1] # True if there is one matching entities
        #third = arr[2]  # True if there is two~four matching entities
        discr = arr[4]  # True if we can discriminate more

        logger.debug('%d among %d slots are accepted (>=0.8 belief).' % 
                     (count80, Ontology.global_ontology.get_length_system_requestable_slots(self.domainString)))

        count80_logic = count80 >= Ontology.global_ontology.get_length_system_requestable_slots(self.domainString)
        if first or second or not discr or count80_logic:  
            # If this inform gives either 0 or 1 or we've found everything we can ask about
            logger.debug('Trying to get inform action, have enough accepted slots.')
            logger.debug('Is there no matching entity? %s.' % str(first))
            logger.debug('Is there only one matching entity? %s.' % str(second))
            logger.debug('Can we discriminate more? %s.' % str(discr))
            requested_slots = SummaryUtils.getRequestedSlots(belief)

            if len(requested_slots) > 0 and offer_happened:
                logger.debug('Getting inform requested action.')
                act = PolicyUtils.getGlobalAction(belief, 'INFORM_REQUESTED', domainString=self.domainString)
            else:
                logger.debug('Getting inform exact action with %d accepted slots.' % count80)
                act = PolicyUtils.getInformAction(count80, belief, domainString=self.domainString)

        if act != 'null()':
            return True, act
        return False, act

    def _getRequest(self, belief, array_slot_summary):
        '''
        '''

        # This is added for confreq.
        need_grounding = SummaryUtils.getTopBeliefs(belief, 0.8, domainString=self.domainString)

        for slot in Ontology.global_ontology.get_sorted_system_requestable_slots(self.domainString):
            summary = array_slot_summary[slot]
            (_, topprob) = summary['TOPHYPS'][0]
            #(_, secprob) = summary['TOPHYPS'][1]

            if topprob < 0.8:
                # Add implicit confirmation (for confreq.)
                grounding_slots = copy.deepcopy(need_grounding)
                if slot in grounding_slots:
                    del grounding_slots[slot]

                grounding_result = []
                for grounding_slot in grounding_slots:
                    if len(grounding_result) < 3:
                        (value, _) = grounding_slots[grounding_slot]
                        #(value, prob) = grounding_slots[grounding_slot]
                        grounding_result.append('%s="%s"' % (grounding_slot, value))

                if not grounding_result or not self.use_confreq:
                    return True, 'request(%s)' % slot
                else:
                    return True, 'confreq(' + ','.join(grounding_result) + ',%s)' % slot

        return False, 'null()'


#END OF FILE
