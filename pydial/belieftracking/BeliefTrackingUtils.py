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
BeliefTrackingUtils.py - Belief Tracking Utility Methods
=========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 
    
    none

************************

'''


__author__ = "cued_dialogue_systems_group"

import json
import copy


def _hasType(action, act):
    '''
    Check whether action list has act.

    :param action: list of items including dialogue acts  and slots
    :type action: list
          
    :param act: dialogue act
    :type act: string

    :return: None
    '''
    for item in action:
        if act == item['act']:
            return True
    
    return False


def _hasTypeSlot(action, act, slot):
    '''
    Check whether action list has act and slot.

    :param action: list of items including dialogue acts  and slots
    :type action: list
          
    :param act: dialogue act
    :type act: string

    :param slot: slot name
    :type slot: string

    :return: None
    '''
    for item in action:
        if act == item['act']:
            for slotvalue in item['slots']:
                if slot == slotvalue[0]:
                    return True
    
    return False


def _getTypeSlot(action, act, slot):
    '''
    Get the value name given act and slot.

    :param action: list of items including dialogue acts  and slots
    :type action: list
          
    :param act: dialogue act
    :type act: string

    :param slot: slot name
    :type slot: string

    :return: string -- value name given act and slot
    '''

    for item in action:
        if act == item['act']:
            for slotvalue in item['slots']:
                if slot == slotvalue[0]:
                    return slotvalue[1]
    
    return ''


def _getCurrentInformedVenue(sact):
    '''
    Get the current informed venue name.

    :param sact: list of items including dialogue acts  and slots
    :type sact: list
    
    :return: string -- value name given act and slot
    '''
    if _hasTypeSlot(sact, 'offer', 'name'):
        return _getTypeSlot(sact, 'offer', 'name')
    return ''


def simplify_belief(ontology, belief):
    '''
    Make the given belief printable by pruning slot values with less than 0.1 belief

    :param ontology: a dict includes the informable/requestable slots and info
    :type ontology: dict

    :param belief: current belief state
    :type belief: dict
    
    :return: dict -- simplified belief
    '''
    simple_belief = copy.deepcopy(belief)
    for slot in simple_belief['beliefs']:
        if slot in ontology['informable']:
            simple_belief['beliefs'][slot] = \
                dict((k, p) for k, p in simple_belief['beliefs'][slot].iteritems() if p > 0.1)
    del simple_belief['features']['inform_info']
    if 'states' in simple_belief:
        del simple_belief['states']
    return simple_belief


def _addprob(sluhyps, hyp, prob):
    '''
    Add prob to hyp in slu hypotheses

    :param sluhyps: slu hypotheses
    :type sluhyps: dict

    :param hyp: target hypothesis
    :type hyp: string

    :param prob: probability to be added
    :type prob: float
    
    :return: dict -- slu hypotheses
    '''
    score = min(1.0, float(prob))
    sluhyps[json.dumps(hyp)] += score
    return sluhyps

def _normaliseandsort(sluhyps):
    '''
    Normalise and sort the given slu hypotheses

    :param sluhyps: slu hypotheses
    :type sluhyps: dict

    :return: list -- list of normalised hypotheses
    '''
    result = []
    sortedhyps = sluhyps.items()
    sortedhyps.sort(key=lambda x:-x[1])
    total_score = sum(sluhyps.values())
    for hyp, score in sortedhyps:
        if total_score==0:
            result.append({"score":0, "slu-hyp":json.loads(hyp)})   #TODO check how to set score here if total_score==0
        else:
            result.append({"score":min(1.0,score/total_score), "slu-hyp":json.loads(hyp)})
    return result

def _transformAct(act, valueTrans,ontology=None, user=True):
    '''
    Normalise and sort the given slu hypotheses

    :return: dict -- transformed dialogue act
    '''
    if user:
        act_without_null = []
        for this_act in act:
            # another special case, to get around deny(name=none,name=blah):
            if this_act["act"] == "deny" and this_act["slots"][0][1] == "none" :
                continue
            # another special case, to get around deny(name=blah,name):
            if this_act["act"] == "inform" and this_act["slots"][0][1] == None :
                continue
            # another special case, to get around deny(name,name=blah):
            if this_act["act"] == "deny" and this_act["slots"][0][1] == None :
                continue
            act_without_null.append(this_act)
        act = act_without_null;

    # one special case, to get around confirm(type=restaurant) in Mar13 data:
    if not user and ontology!=None and "type" not in ontology["informable"] :
        for this_act in act:
            if this_act["act"] == "expl-conf" and this_act["slots"] == [["type","restaurant"]] :
                act = [{"act":"confirm-domain","slots":[]}]

    for i in range(0,len(act)):
        for j in range(0, len(act[i]["slots"])):
            act[i]["slots"][j][:] = [valueTrans[x] if x in valueTrans.keys() else x for x in act[i]["slots"][j]]

    # remove e.g. phone=dontcare and task=find
    if ontology != None:
        new_act = []
        for a in act:
            new_slots = []
            for slot,value in a["slots"]:
                keep = True
                if slot not in ["slot","this"] and (slot not in ontology["informable"]) :
                    if user or (slot not in ontology["requestable"]+["count"]) :
                        keep = False
                if keep :
                    new_slots.append((slot,value))
            if len(a["slots"]) == 0 or len(new_slots)>0 :
                a["slots"] = new_slots
                new_act.append(a)
    else :
        new_act = act

    return new_act

def print_obs(obs):
    '''
    Print observations
    '''
    hypstr = ''.join(obs.charArray)
    hyplist = hypstr.split('\t')
    for i in xrange(len(hyplist)):
        print hyplist[i], obs.doubleArray[i]

def order_using(l, lookup):
    '''
    Return the sorted list of l given the lookup table 

    :param l: given list l
    :type l: type

    :param lookup: lookup table
    :type lookup: list

    :return: list -- sorted list 
    '''
    def _index(item):
        try :
            return lookup.index(item)
        except ValueError :
            return len(lookup)
    return sorted(l, key = _index)


#END OF FILE
