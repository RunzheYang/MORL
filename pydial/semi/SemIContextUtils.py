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
SemIContextUtils.py - Semantic input parser context utils
===================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies:

    import :mod:`ontology.Ontology` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

from utils import ContextLogger, DiaAct
from ontology import OntologyUtils, Ontology
logger = ContextLogger.getLogger('')


#START CONTEXTUAL SEMI PARSING TOOLS-------------------------------------------------------------
def _add_context_to_user_act(sys_act, hyps, active_domain):
    '''
    Context information is added to the detected user input, eg affirm() is transformed to inform(slot=value)
    
    Note: this does eventually depend on the current domain we assume we are operating in (self.active_domain)
    
    :param sys_act: system dialogue act
    :type sys_act: str
    :param hyps: hypothesis
    :type hyps: list
    :return:
    '''

    logger.info('Possibly adding context to user semi hyps: %s' % hyps)
    if not len(hyps) or sys_act is None:
        return hyps
    # if negated -- only deal with this for now if it pertains to a binary slot. dont have an act for
    # "i dont want indian food" for example
    new_hyps = []
    for hyp in hyps:
        if hyp[0] in ['affirm()','negate()']:
            user_act,prob = hyp
            user_act = _convert_yes_no(sys_act, user_act, active_domain)  
            new_hyps.append((user_act,prob))
        else:
            new_hyps.append(hyp)
    return new_hyps
       
def _give_context(slot,val):
    '''
    Added context to user act, ie, create the slot-value string.
    
    :param slot: slot
    :type slot: str
    :param val: value
    :type val: str
    :return: the string slot='value'
    '''
    
    contextual_slot_val_pair = slot+'="'+val+'"'
    logger.info("Added context to user act: "+contextual_slot_val_pair) 
    return contextual_slot_val_pair
    
def _apply_affirm_negate_to_value(affirm_or_negate,val):
    '''
    Returns the value implied by the affirm() or negate() act
    
    Returns the passed value val as a default
    
    :param affirm_or_negate: the affirm() or negate() act
    :type affirm_or_negate: 
    :param val: the value to be used
    :type val: str
    :return: value or 0/1
    '''

#         NB we have checked at this point that the slotX is binary if the sys_act was request(slotX).
#         For confirm(slotX=Y) we can only deal with this if it is an affirm() --> inform(slotX=Y)
#         if it is a negate we will need to have checked this outside this function.


    if val is None:
        val = '1'     # binary slot true value - should be only case where val is None here 
    if affirm_or_negate == 'affirm()':
        return val
    elif affirm_or_negate == 'negate()' and val in ['0','1']: #only know how to negate binary slots
        # Note that request(hasparking) > negate() > val would have been None on passing here (changed to 1 above)
        return '0' if val == '1' else '1'
    else: 
        return val  # set val = none ? so we ask again on this slot?

def _convert_yes_no(sys_act, user_act, active_domain):
    '''
    Converts yes/no only responses from user into affirm and negate.
    
    Necessary for binary slots in system utterance ie. request(hasparking) --> inform(slot=opposite)
    
    :param sys_act: the last system action
    :type sys_act: str
    :param user_act: the user input act to be processed
    :type user_act: str
    :return: the transformed user act if conditions apply else the untouched user act
    '''

    # TODO - should definitely be more scenarios to deal with here
    
    if OntologyUtils.BINARY_SLOTS[active_domain] is None:
        Ontology.global_ontology.updateBinarySlots(active_domain)
    
    dact = DiaAct.DiaAct(sys_act)
    slot_val_pairs = []
    if dact.act in ['request', 'confirm']: 
        for item in dact.items:         
            slot, op, val = item.slot, item.op, item.val
            if dact.act == 'request' and slot not in OntologyUtils.BINARY_SLOTS[active_domain]:
                logger.warning('Attempting to negate/affirm a non binary valued slot')  
                return user_act # default back to unchanged act
            if dact.act == 'confirm' and user_act == 'negate()':
                # cases 1: should have been parsed by semi into some deny(A=b,A=c) act
                # cases 2: user just replied 'no' which doesn't make sense -- just return original act
                # Dangerously close to writing policy rules here ... dont want to assume we can return 
                # inform(A=dontcare) for instance. Same for above if statement. Let the dialog have a bad turn... 
                return user_act  
            val = _apply_affirm_negate_to_value(user_act, val)
            slot_val_pairs.append(_give_context(slot,val))
        contextual_act = 'inform('+','.join(slot_val_pairs)+')'
        logger.info("New contexual act: "+contextual_act)
        return unicode(contextual_act)
    elif dact.act in ['reqmore']:
        if user_act == 'negate()':
            return user_act  # TODO - think it might cause problems in MTurk dialogs to return bye()
            #return unicode('bye()')  # no to reqmore() is an implicit goodbye. 
    else:
        logger.warning('affirm or negate in response to currently unhandled system_act:\n '+str(sys_act))
    return user_act  # otherwise leave it the same
#END ADD CONTEXT SEMI PARSE TOOLS-------------------------------------------------------------------------

