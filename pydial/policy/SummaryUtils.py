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
SummaryUtils.py - summarises dialog events for mapping from master to summary belief 
======================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Usage**: 
    >>> import SummaryUtils
   
.. Note::
        No classes; collection of utility methods

Local module variables::

    global_summary_features:    (list) global actions/methods
    REQUESTING_THRESHOLD:             (float) 0.5 min value to consider a slot requested

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|

************************

'''

__author__ = "cued_dialogue_systems_group"

import copy
from scipy.stats import entropy
from ontology import Ontology
from utils import ContextLogger, Settings
logger = ContextLogger.getLogger('')


global_summary_features = ['GLOBAL_BYCONSTRAINTS',
                           'GLOBAL_BYALTERNATIVES',
                           'GLOBAL_BYNAME',
                           'GLOBAL_FINISHED',
                           'GLOBAL_REPEAT',
                           'GLOBAL_REQMORE',
                           'GLOBAL_THANKYOU',
                           'GLOBAL_ACK',
                           'GLOBAL_RESTART',
                           'GLOBAL_COUNTACCEPTED',
                           'GLOBAL_NAMENONE',
                           'GLOBAL_OFFERHAPPENED']

REQUESTING_THRESHOLD = 0.5


'''
#####Belief state related methods.####
'''

def globalSummary(belief, domainString):
    '''
    summary of global actions such as offer happened etc.

    :param belief: dict representing the full belief state
    :param domainString: string representing the domain
    :return: (dict) summary. dict keys are given by :meth:`global_summary_features`
    '''
    method_prob_mass = sum(belief['beliefs']['method'].values())
    belief['beliefs']['method']['none'] = 1 - method_prob_mass # hack to fix the bug not assigning 'none' any prob mass
    topMethod, topMethodBelief = getTopBelief(belief['beliefs']['method'])
    topDiscourseAct, topDiscourseActBelief = getTopBelief(belief['beliefs']['discourseAct'])

    summaryArray = dict.fromkeys(global_summary_features, False)
    summaryArray['GLOBAL_COUNTACCEPTED'] = len(getTopBeliefs(belief, domainString=domainString))
    summaryArray['GLOBAL_NAMENONE'] = belief['features']['lastActionInformNone']
    summaryArray['GLOBAL_OFFERHAPPENED'] = belief['features']['offerHappened']

    if topMethod == 'byalternatives':
        summaryArray['GLOBAL_BYALTERNATIVES'] = True
    elif topMethod == 'byname':
        summaryArray['GLOBAL_BYNAME'] = True
    elif topMethod == 'finished' and topMethodBelief > 0.5:
        summaryArray['GLOBAL_FINISHED'] = True
    elif topMethod == 'restart' and topMethodBelief > 0.5:
        summaryArray['GLOBAL_RESTART'] = True
    else:
        summaryArray['GLOBAL_BYCONSTRAINTS'] = True

    if topDiscourseAct == 'repeat' and topDiscourseActBelief > 0.5:
        summaryArray['GLOBAL_REPEAT'] = True
    elif topDiscourseAct == 'reqmore' and topDiscourseActBelief > 0.5:
        summaryArray['GLOBAL_REQMORE'] = True
    elif topDiscourseAct == 'thankyou' and topDiscourseActBelief > 0.5:
        summaryArray['GLOBAL_THANKYOU'] = True
    elif topDiscourseAct == 'ack' and topDiscourseActBelief > 0.5:
        summaryArray['GLOBAL_ACK'] = True

    return summaryArray


def arraySlotSummary(belief, domainString):
    '''
    Gets the summary vector for goal slots, including the top probabilities, entropy, etc.

    :param belief: dict representing the full belief state
    :param domainString: string representing the domain
    :return: (dict) of slot goal summaries
    '''
    summary = {}
    slots = Ontology.global_ontology.get_sorted_system_requestable_slots(domainString)
        
    for slot in slots:
        summary[slot] = {}
        slot_belief = belief['beliefs'][slot]
        summary[slot]['TOPHYPS'], summary[slot]['ISTOPNONE'] = getTopBeliefsExcludingNone(belief['beliefs'][slot])
        belief_dist = slot_belief.values()
        summary[slot]['ENTROPY'] = entropy(belief_dist)
        summary[slot]['ISREQUESTTOP'] = belief['beliefs']['requested'][slot] > 0.5

    return summary


def getRequestedSlots(belief):
    '''
    Iterate get the list of mentioned requested slots

    :param belief: dict representing the full belief state
    :return: (list) of slot names with prob retrieved from belief > REQUESTING_THRESHOLD (an internal global)
    '''
    requested_slots = []
    for slot in belief['beliefs']['requested']:
        requestprob = belief['beliefs']['requested'][slot]
        if requestprob > REQUESTING_THRESHOLD:
            requested_slots.append(slot)
    return requested_slots


def getTopBelief(slot_belief):
    '''
    Return slot value with the largest belief

    :param slot_belief: dict of value-prob pairs for slot distribution
    :return: top_value (str), top_belief (float)
    '''

    top_value = max(slot_belief, key=slot_belief.get)
    return top_value, slot_belief[top_value]


def getTopBeliefs(belief, threshold='auto', domainString=None):
    '''
    Get slot values with belief larger than threshold

    :param belief: dict representing the full belief state
    :param threshold: threshold on slot value probabilities. Default value is 'auto', only allowable string
    :param domainString: string representing the domain
    :return: (dict) as {slot: (topvalue, topbelief), ...}
    '''
    top_beliefs = {}
    for slot in Ontology.global_ontology.get_system_requestable_slots(domainString):
        if threshold == 'auto':
            numvalues = Ontology.global_ontology.get_len_informable_slot(domainString, slot)
            thres = 1. / (float(numvalues) - 0.1)
        else:
            thres = threshold

        topvalue, topbelief = getTopBelief(belief['beliefs'][slot])

        if topvalue != '**NONE**' and topbelief > thres:
            top_beliefs[slot] = (topvalue, topbelief)

    return top_beliefs


def getTopBeliefsExcludingNone(slot_belief):
    '''
    get the ordered list of (value,belief) in slot

    :param slot_belief: dict of value-prob pairs for slot distribution
    :return: (list) of ordered value-beliefs, (bool) telling if the top value is **NONE**
    '''
    slot_belief_copy = copy.deepcopy(slot_belief)
    top_hyps = []
    is_top_none = False
    while len(slot_belief_copy) > 0:
        topvalue, topbelief = getTopBelief(slot_belief_copy)
        if len(top_hyps) == 0 and topvalue == '**NONE**':
            is_top_none = True
        if topvalue != '**NONE**':
            top_hyps.append((topvalue, topbelief))
        del slot_belief_copy[topvalue]

    return top_hyps, is_top_none

'''
####Methods for inform related actions.####
'''

def acceptanceListCanBeDiscriminated(accepted_values, domainString, num_accepted=None):
    '''
    Checks if the given acceptance list with the given number of values accepted
    returns a list of values which can be discriminated between -
    i.e. there is a question which we could ask which would give differences between
    the values.
    Note that only slots from the full acceptanceList (i.e. not just below
    maxAcceptedSlots are used for discrimination to exclude things like phone, addr, etc)

    :param accepted_values: dict of slot-value-beliefs whose beliefs are above **NONE**
    :param domainString: string representing the domain
    :return: (bool) answering discrimination question
    '''

    if num_accepted == None:
        num_accepted = len(accepted_values)

    ordered_accepted_values = []
    for slot, value in accepted_values.iteritems():
        ordered_accepted_values.append((slot, value[0], value[1]))
    ordered_accepted_values = sorted(ordered_accepted_values, key=lambda x: x[2], reverse=True)[:num_accepted]

    return Ontology.global_ontology.constraintsCanBeDiscriminated(domainString, constraints=ordered_accepted_values)


def getInformNoneVenue(constraints):
    '''
    creates inform(name=none,...) act

    :param constraints: dict of accepted slot-values
    :return: (str) inform(name=none,...) act
    '''
    feats = {}
    for slot in constraints:
        if constraints[slot] != 'dontcare':
            feats[slot] = constraints[slot]
    return 'inform(name=none, {})'.format(convertFeatsToStr(feats))


def getInformByConstraints(constraints, domainString, lastInformedVenue):
    '''
    Looks for a database match with constraints and converts this entity into a dialogue act

    :param constraints: dict of slot:values whose beliefs are above **NONE**
    :param domainString: string representing the domain
    :return: string representing the inform dialogue act
    '''
    entities = Ontology.global_ontology.entity_by_features(domainString, constraints)
    if len(entities) == 0:
        return getInformNoneVenue(constraints)
    else:
        ret_ent = entities[0]
        for ent in entities:
            if ent['name'] == lastInformedVenue:
                ret_ent = ent
                break
        return getInformEntity(constraints, ret_ent)


def getInformEntity(accepted_values, ent):
    '''
    Converts a database entity into a dialogue act

    :param accepted_values: dict of slot-values whose beliefs are above **NONE**
    :param ent: database entity to be converted to dialogue act
    :return: string representing the inform dialogue act
    '''
    feats = {'name': ent['name']}
    numFeats = len(accepted_values)
    acceptance_keys = accepted_values.keys()

    maxNumFeats = 5
    if Settings.config.has_option("summaryacts", "maxinformslots"):
        maxNumFeats = int(Settings.config.get('summaryacts', 'maxinformslots'))

    if numFeats > maxNumFeats:
        Settings.random.shuffle(acceptance_keys)
        acceptance_keys = acceptance_keys[:maxNumFeats]

    for slot in acceptance_keys:
        if slot != 'name':
            value = accepted_values[slot]
            if value == 'dontcare' and slot in ent and ent[slot] != "not available":
                feats[slot] = ent[slot]
            else:
                if slot in ent:
                    feats[slot] = ent[slot]
                else:
                    logger.warning('Slot {} is not found in data for entity {}'.format(slot, ent['name']))

    return 'inform({})'.format(convertFeatsToStr(feats))


def getInformRequestedSlots(requested_slots, name, domainString):
    '''
    Informs about the requested slots from the last informed venue of form the venue informed by name

    :param requested_slots: list of requested slots
    :param name: name of the last informed venue
    :param domainString: string representing the domain
    :return: string representing the inform dialogue act
    '''
    result = Ontology.global_ontology.entity_by_features(domainString, {'name': name})

    if len(result) > 0:
        ent = result[0]
        return _getInformRequestedSlotsForEntity(requested_slots, ent, domainString)
    else:
        if not name:
            # Return a random venue
            result = []
            while len(result) == 0:
                rand_name = Ontology.global_ontology.getRandomValueForSlot(domainString, 'name', nodontcare=True)
                result = Ontology.global_ontology.entity_by_features(domainString, {'name': rand_name})
            ent = result[0]
            return _getInformRequestedSlotsForEntity(requested_slots, ent, domainString)

        else:
            logger.warning('Couldn\'t find the provided name: ' + name)
            return getInformNoneVenue({'name': name})


def _getInformRequestedSlotsForEntity(requested_slots, ent, domainString):
    '''
    Converts the list of requested slots and the entity into a inform_requested dialogue act

    :param requested_slots: list of requested slots (obtained in getRequestedSlots())
    :param ent: dictionary with information about a database entity
    :return: string representing the dialogue act
    '''

    slotvaluepair = ['name="{}"'.format(ent['name'])]
    if len(requested_slots) == 0:
        if 'type' in ent:
            slotvaluepair.append('type="{}"'.format(ent['type']))
        else:
            # type is not part of some ontologies. in this case just add a random slot-value
            slots = Ontology.global_ontology.get_requestable_slots(domainString)
            if 'name' in slots:
                slots.remove('name')
            if Settings.config.has_option('summaryacts', 'DSTC2requestables'):
                if Settings.config.getboolean('summaryacts', 'DSTC2requestables'):
                    if 'description' in slots:
                        slots.remove('description')
                    if 'signature' in slots:
                        slots.remove('signature')
            slot = slots[Settings.random.randint(len(slots))]
            slotvaluepair.append('{}="{}"'.format(slot, ent[slot]))

    else:
        max_num_feats = 5
        if Settings.config.has_option("summaryacts", "maxinformslots"):
            max_num_feats = int(Settings.config.get('summaryacts', 'maxinformslots'))

        if len(requested_slots) > max_num_feats:
            Settings.random.shuffle(requested_slots)
            requested_slots = requested_slots[:max_num_feats]

        for slot in requested_slots:
            if slot != 'name' and slot != 'location':
                if slot in ent:
                    slotvaluepair.append('{}="{}"'.format(slot, ent[slot]))
                else:
                    slotvaluepair.append('{}=none'.format(slot))

    return 'inform({})'.format(','.join(slotvaluepair))


def getInformAlternativeEntities(accepted_values, prohibited_list, domainString):
    ''' returns an inform dialogue act informing about an entity that has not been informed before

    :param accepted_values: dict of slot-value-beliefs whose beliefs are above **NONE**
    :param prohibited_list: list of already mentioned entities
    :param domainString: string representing the domain
    :return: the dialogue act representing either
    1) there is not matching venue: inform(name=none, slot=value, ...)
    2) it offers a venue which is not on the prohibited list
    3) if all matching venues are on the prohibited list then it says
       there is no venue except x,y,z,... with such features:
       inform(name=none, name!=x, name!=y, name!=z, ..., slot=value, ...)
    '''

    constraints = get_constraints(accepted_values)
    result = Ontology.global_ontology.entity_by_features(domainString, constraints)
    if len(result) == 0:
        return getInformNoneVenue(constraints)
    else:
        for ent in result:
            name = ent['name']
            if name not in prohibited_list:
                return getInformEntity(accepted_values, ent)

        return getInformNoMoreVenues(accepted_values, result)


def getInformNoMoreVenues(accepted_values, entities):
    '''
    returns inform(name=none, other than x and y, with constraints w and z) act

    :param accepted_values: dict of slot-value-beliefs whose beliefs are above **NONE**
    :param entities: list of database entity dicts
    :return: (str) inform() action
    '''

    maxNumFeats = 5
    if Settings.config.has_option("summaryacts", "maxinformslots"):
        maxNumFeats = int(Settings.config.get('summaryacts', 'maxinformslots'))

    feats = {}
    for slot in accepted_values:
        value = accepted_values[slot][0]
        if slot != 'name' or value != 'dontcare':
            feats[slot] = value

    if len(feats) > maxNumFeats:
        feats_keys = feats.keys()
        truncated_feats = {}
        Settings.random.shuffle(feats_keys)
        for key in feats_keys[:maxNumFeats]:
            truncated_feats[key] = feats[key]
        feats = truncated_feats

    prohibited_list = ''
    for ent in entities:
        prohibited_list += 'name!="{}",'.format(ent['name'])

    return 'inform(name=none,{}{})'.format(prohibited_list, convertFeatsToStr(feats))


def get_constraints(accepted_values):
    constraints = {}
    for slot in accepted_values:
        constraints[slot] = accepted_values[slot][0]
    return constraints


def convertFeatsToStr(feats):
    result = []
    for slot in feats:
        value = feats[slot]
        if value is not None and value.lower() != 'not available' and value != '':
            result.append('{}="{}"'.format(slot, value))

    return ','.join(result)


def actionSpecificInformSummary(belief, numAccepted, domainString):
    '''count: # of entities matching with numAccepted slots in acceptance list.

        :param belief: full belief state
        :type belief: dict
        :param numAccepted: None
        :type numAccepted: int
        :param informable_slots: None
        :type informable_slots: list
        :returns: summary_array [count==0, count==1, 2<=count<=4, count>4, discriminatable] \  
                        discriminatable: matching entities can be further discriminated
    '''
    acceptanceList = getTopBeliefs(belief, domainString=domainString)
    count = _countEntitiesForAcceptanceListPart(acceptanceList, numAccepted, domainString)
    discriminatable = acceptanceListCanBeDiscriminated(acceptanceList, domainString, numAccepted)
    summary_array = [count == 0, count == 1, 2 <= count <= 4, count > 4, discriminatable]
    return summary_array

def _countEntitiesForAcceptanceListPart(accepted_values, num_accepted, domainString):
    '''
    Returns the number of entities matching the first self.maxAcceptedSlots (default=10)
    values in the acceptance list. Includes values with dontcare in the count

    :param acceptanceList: {slot: (topvalue, topbelief), ...}
    :param numAccepted: None
    :type numAccepted: int
    :returns: (int) number of entities
    '''

    ordered_accepted_values = []
    for slot, value in accepted_values.iteritems():
        ordered_accepted_values.append((slot, value[0], value[1]))
    ordered_accepted_values = sorted(ordered_accepted_values, key=lambda x: x[2], reverse=True)[:num_accepted]

    constraints = {}
    for slot, value, _ in ordered_accepted_values: # slot, value, belief
        if value != 'dontcare':
            constraints[slot] = value

#     return len(Ontology.global_ontology.entity_by_features(domainString, constraints=constraints))
    return Ontology.global_ontology.get_length_entity_by_features(domainString, constraints=constraints)




#END OF FILE
