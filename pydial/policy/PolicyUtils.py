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
PolicyUtils.py - Utility Methods for Policies 
================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. Note::
        PolicyUtils.py is a collection of utility functions only (No classes).

Local/file variables::
    
    ZERO_THRESHOLD:             unused
    REQUESTING_THRESHOLD:       affects getRequestedSlots() method

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`policy.SummaryUtils` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"
import copy
import os

from utils import DiaAct, Settings
from utils import ContextLogger
from ontology import Ontology
import SummaryUtils
logger = ContextLogger.getLogger('')

ZERO_THRESHOLD = 0.001
REQUESTING_THRESHOLD = 0.5

'''
Methods for global action.
'''

def getGlobalAction(belief, globalact, domainString):
    '''**Method for global action:** returns action 

    :param belief: full belief state
    :type belief: dict
    :param globalact: - str of globalActionName, e.g. 'INFORM_REQUESTED'
    :type globalact: int
    :param domainString: domain tag
    :type domainString: str
    :returns: (str) action
    '''

    # First get the name for the name goal.
    topvalue, topbelief = SummaryUtils.getTopBelief(belief['beliefs']['name'])
    toptwo, _ = SummaryUtils.getTopBeliefsExcludingNone(belief['beliefs']['name'])
    if topvalue == '**NONE**' or topvalue == 'dontcare' or topbelief < 0.8:
        topnamevalue = ''
    else:
        topnamevalue = toptwo[0][0]

    lastInformedVenue = belief['features']['lastInformedVenue']
    informedVenueSinceNone = belief['features']['informedVenueSinceNone']
    acceptanceList = SummaryUtils.getTopBeliefs(belief, domainString=domainString)
    inform_threshold = 0
    if Settings.config.has_option("policy", "informthreshold"):
        inform_threshold = float(Settings.config.get('policy','informthreshold'))
    if inform_threshold > 0:
        acceptanceList80 = SummaryUtils.getTopBeliefs(belief, inform_threshold, domainString=domainString)
    else:
        acceptanceList80 = acceptanceList
    requestedSlots = SummaryUtils.getRequestedSlots(belief)

    # logger.debug('topnamevalue = %s, lastInformedVenue = %s' % (topnamevalue, lastInformedVenue))
    if topnamevalue == '' and lastInformedVenue != '':
        # logger.debug('topnamevalue is None, but copy lastInformedVenue')
        topnamevalue = lastInformedVenue


    if globalact == 'INFORM_REQUESTED':
        if topnamevalue != '':
            return _getInformRequestedSlots(acceptanceList80, requestedSlots, topnamevalue, domainString)
        else:
            return _getInformRequestedSlots(acceptanceList80, requestedSlots, 'none', domainString)
    elif globalact == 'INFORM_ALTERNATIVES':
        #if lastInformedVenue == '':
        #    print 'Cant inform alternatives no last informed venue'
        #    return 'null()'
        #else:
        return _getInformAlternativeEntities(acceptanceList, acceptanceList80, informedVenueSinceNone, domainString)
    elif globalact == 'INFORM_MORE': #ic340: is this ever used?
        if len(informedVenueSinceNone) > 0 and topnamevalue != '':
            return _getInformMoreEntity(topnamevalue, domainString)
        else:
            return _getInformMoreEntity('none', domainString)
    elif globalact == 'INFORM_BYNAME':
        return _getInformAlternativeEntities(acceptanceList, acceptanceList80, [], domainString)
    elif globalact == 'INFORM_REPEAT':
        return 'null()'
    elif globalact == 'REQMORE':
        if lastInformedVenue != '':
            return 'reqmore()'
        else:
            return 'null()'
    elif globalact == 'BYE':
        return 'bye()'
    elif globalact == 'RESTART':
        return 'null()'
    else:
        logger.warning('Invalid global summary action name: ' + globalact)
        return 'null()'


'''
Methods for global inform action.
'''
def getInformAction(numAccepted, belief, domainString):
    '''**Method for global inform action:** returns inform act via getInformExactEntity() method \
            or null() if not enough accepted 
    
    :param belief: full belief state
    :type belief: dict
    :param numAccepted: number of slots with prob. mass > 80
    :type numAccepted: int
    :param domainString: domain tag
    :type domainString: str
    :returns: getInformExactEntity(acceptanceList,numAccepted)
    '''

    acceptanceList = SummaryUtils.getTopBeliefs(belief, domainString=domainString) # dict containing the slots with top values diferent to **NONE** and their top values
    if numAccepted > len(acceptanceList): # ic340: When would this happen?
        return 'null()'

    return getInformExactEntity(acceptanceList, numAccepted, domainString)


def getInformExactEntity(acceptanceList, numAccepted, domainString):
    '''**Method for global inform action:** creates inform act with none or an entity

    :param acceptanceList: of slots with value:prob mass pairs 
    :type acceptanceList: dict
    :param numAccepted: number of *accepted slots* (>80 prob mass)
    :type numAccepted: int
    :param domainString: domain tag
    :type domainString: str
    :returns: getInformNoneVenue() or getInformAcceptedSlotsAboutEntity() as appropriate
    '''
    acceptedValues = {}
    for i, slot in enumerate(acceptanceList):
        if i >= numAccepted:
            break
        #(topvalue, topbelief) = acceptanceList[slot]
        (topvalue, _) = acceptanceList[slot]
        if topvalue != 'dontcare':
            acceptedValues[slot] = topvalue

    result = Ontology.global_ontology.entity_by_features(domainString, acceptedValues)
    if len(result) == 0:
        return SummaryUtils.getInformNoneVenue(acceptedValues)
    else:
        ent = result[0]
        return getInformAcceptedSlotsAboutEntity(acceptanceList, ent, numAccepted)


def getInformAcceptedSlotsAboutEntity(acceptanceList, ent, numFeats):
    '''**Method for global inform action:** returns filled out inform() string
    need to be cleaned (Dongho)
    
    :param acceptanceList: of slots with value:prob mass pairs 
    :type acceptanceList: dict
    :param ent: slot:value properties for this entity
    :type ent: dict
    :param numFeats: result of globalOntology.entity_by_features(acceptedValues)
    :type numFeats: int
    :returns: (str) filled out inform() act
    '''

    ans = 'inform('
    feats = {'name': ent['name']}
    acceptanceKeys = acceptanceList.keys()

    maxNumFeats = 5
    if Settings.config.has_option("policy", "maxinformslots"):
        maxNumFeats = int(Settings.config.get('policy', 'maxinformslots'))

    if numFeats > maxNumFeats:
        Settings.random.shuffle(acceptanceKeys)
        acceptanceKeys = acceptanceKeys[:maxNumFeats]

    for i, slot in enumerate(acceptanceKeys):
        if i >= numFeats:
            break
        if slot == 'name':
            continue

        (value, belief) = acceptanceList[slot]
        if value == 'dontcare' and slot in ent and ent[slot] != "not available":
            feats[slot] = ent[slot]
        else:
            if slot in ent:
                feats[slot] = ent[slot]#value
            else:
                logger.debug('Slot %s is not found in data for entity %s' % (slot, ent['name']))

    ans += SummaryUtils.convertFeatsToStr(feats) + ')'
    return ans

def _getInformRequestedSlots(acceptanceList80, requestedSlots, name, domainString, bookinginfo=None):
    result = Ontology.global_ontology.entity_by_features(domainString, {'name':name})

    acceptedValues = {}
    for slot in acceptanceList80:
        (topvalue, topbelief) = acceptanceList80[slot]
        if topvalue != 'dontcare':
            acceptedValues[slot] = topvalue

    # add bookinginfo info to acceptedValues
    if bookinginfo is not None:
        for slot,constraint in bookinginfo.iteritems():
            if constraint is not None:
                acceptedValues[slot] = constraint
        

    if len(result) > 0 and name != 'none':
        # We found exactly one or more matching entities. Use the first one
        ent = result[0]
#         return SummaryUtils._getInformRequestedSlotsForEntity(acceptedValues, requestedSlots, ent) CHECK
        return SummaryUtils._getInformRequestedSlotsForEntity(requestedSlots, ent, domainString)
    else:
        logger.debug('Couldn\'t find the provided name ' + name)
        # We have not informed about an entity yet, or there are too many entities.
        return 'null()'


def _getInformAlternativeEntities(acceptanceList, acceptanceList80, prohibitedList, domainString):
    '''
    Returns the dialogue act representing either
    1) there is not matching venue: inform(name=none, slot=value, ...)
    2) it offers a venue which is not on the prohibited list
    3) if all matching venues are on the prohibited list then it says
       there is no venue except x,y,z,... with such features:
       inform(name=none, name!=x, name!=y, name!=z, ..., slot=value, ...)
    '''
    acceptedValues = {}
    numFeats = len(acceptanceList80)
    for slot in acceptanceList80:
        (topvalue, topbelief) = acceptanceList80[slot]
        if topvalue != 'dontcare':
            acceptedValues[slot] = topvalue

    if len(acceptedValues) == 0:
        logger.warning("User didn't specify any constraints or all are dontcare")
        #return 'null()'

    result = Ontology.global_ontology.entity_by_features(domainString, acceptedValues)
    if len(result) == 0:
        return SummaryUtils.getInformNoneVenue(acceptedValues)
    else:
        for ent in result:
            name = ent['name']
            if name not in prohibitedList:
                return getInformAcceptedSlotsAboutEntity(acceptanceList80, ent, numFeats)

        return SummaryUtils.getInformNoMoreVenues(acceptanceList, result)

    return 'null()'


def _getInformMoreEntity(name, domainString): #ic340 is this ever used? apparently not
    '''
    Finds the last informed entity and it informs about the non-accepted slots.
    @param name the last informed entity
    '''
    result = Ontology.global_ontology.entity_by_features(domainString, {'name':name})
    if name != 'none' and len(result) > 0:
        ent = result[0]
        return _getInformCommentSlotAboutEntity(ent)
    else:
        return 'null()'

def _getInformCommentSlotAboutEntity(ent):

    output = 'inform(name="%s"' % ent['name']
    if 'comment' in ent:
        output += ',comment="%s"' % ent['comment']
    if 'type' in ent:
        output += ',type="%s"' % ent['type']
    output += ')'
    return output


def add_venue_count(input, belief, domainString):
    '''Add venue count.

    :param input: String input act.
    :param belief: Belief state
    :param domainString: domain tag like 'SFHotels'
    :type domainString: str
    :returns: act with venue count.
    '''
    acceptanceList = SummaryUtils.getTopBeliefs(belief, domainString)
    accepted_slots = {}
    for i, slot in enumerate(acceptanceList):
        (topvalue, topbelief) = acceptanceList[slot]
        if topvalue != 'dontcare':
            accepted_slots[slot] = topvalue

    count = Ontology.global_ontology.get_length_entity_by_features(domainString, accepted_slots)
    input_act = DiaAct.DiaAct(input)
    if input_act.act == 'confreq':
        if count > 1:
            output = copy.deepcopy(input_act)
            for slot in accepted_slots:
                val = accepted_slots[slot]
                if not input_act.contains(slot, val):
                    output.append_front(slot, val)

            output.append_front('count', str(count))
            return str(output)
    #     else:
    #         logger.warning('accepted slots: ' + str(accepted_slots))
    #         logger.error('badact in add_venue_count: input=%s, count=%d' % (input, count))
    #         return 'badact()'
    # elif count <=1 and len(accepted_slots) > 0 and input_act.act in ['confirm', 'request', 'select']:
    #     logger.warning('accepted slots: ' + str(accepted_slots))
    #     logger.error('badact in add_venue_count: input=%s, count=%d' % (input, count))
    #     return 'badact()'
    return input

# Saving policy:
def checkDirExistsAndMake(fullpath):
    """Used when saving a policy -- if dir doesn't exisit --> is created
    """
    if '/' in fullpath:
        path = fullpath.split('/')
        path = '/'.join(path[:-1])
        try: 
            os.makedirs(path)
            logger.info("Created dir(s): " + path)
        except OSError:
            if not os.path.isdir(path):
                raise
    else:
        return  # nothing to do, saving to root
        

#END OF FILE