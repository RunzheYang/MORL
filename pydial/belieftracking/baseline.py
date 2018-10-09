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
baseline.py - Baseline Belief Tracker from DST
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. note::
        This is (essentially) the baseline.py file from the DSTC challenge written by Matt Henderson. It implements the
        "Baseline" and "Focus" trackers from the DSTC challenge. Some unused things have been removed from the DST version
        here - just to keep PyDial simple. 


.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :class:`belieftracking.BeliefTracker.BeliefTracker` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"

import copy
from collections import defaultdict
from BeliefTracker import BeliefTracker
from ontology import Ontology
from utils import ContextLogger
logger = ContextLogger.getLogger('')


def labels(user_act, mact, lastInformedVenue):
    '''
    Convert inputs to compatible info to belief state

    :param user_act: user's dialogue acts
    :type user_act: dict

    :param mact: machine's dialogue acts
    :type mact: dict

    :param lastInformedVenue:
    :type lastInformedVenue: string

    :return: informed_goals, denied_goals, requested_slots, method, discourseAct, lastInformedVenue
    '''
    # get context for "this" in inform(=dontcare)
    # get context for affirm and negate
    this_slot = None
    
    confirm_slots = {"explicit":[], "implicit":[]}
    for act in mact :
        if act["act"] == "request" :
            this_slot = act["slots"][0][1]
        elif act["act"] == "select" :
            this_slot = act["slots"][0][0]
        elif act["act"] == "impl-conf":
            confirm_slots["implicit"] += act["slots"]
        elif act["act"] == "expl-conf" :
            confirm_slots["explicit"] += act["slots"]
            this_slot = act["slots"][0][0]

    # lastInformedVenue by dk449
    for act in mact:
        if act["act"] == "offer":
            lastInformedVenue = act["slots"][0][1]
            
            
    # goal_labels
    informed_goals = {}
    denied_goals = defaultdict(list)
    for act in user_act :
        act_slots = act["slots"]
        slot = None
        value = None
        if len(act_slots) > 0:
            assert len(act_slots) == 1
            
            if act_slots[0][0] == "this" :
                slot = this_slot
            else :
                slot = act_slots[0][0]
            value = act_slots[0][1]
                
        
        if act["act"] == "inform" and slot != None:
            informed_goals[slot]=(value)
            
        elif act["act"] == "deny" and slot != None:
            denied_goals[slot].append(value)
            
        elif act["act"] == "negate" :
            slot_values = confirm_slots["implicit"] + confirm_slots["explicit"]
            if len(slot_values) > 1:
                #print "Warning: negating multiple slots- it's not clear what to do."
                pass
            else :
                for slot, value in slot_values :
                    denied_goals[slot].append(value)
            
        elif act["act"] == "affirm" :
            slot_values = confirm_slots["explicit"]
            if len(slot_values) > 1:
                #print "Warning: affirming multiple slots- it's not clear what to do."
                pass
            else :
                for slot, value in confirm_slots["explicit"] :
                    informed_goals[slot]=(value)
                    
    # requested slots
    requested_slots = []
    for act in user_act :
        if act["act"] == "request": 
            for _, requested_slot in act["slots"]:
                requested_slots.append(requested_slot)
        if act["act"] == "confirm": # added by dk449
            for requested_slot, _ in act["slots"]:
                requested_slots.append(requested_slot)
    # method
    method="none"
    act_types = [act["act"] for act in user_act]
    mact_types = [act["act"] for act in mact]
    
    if "reqalts" in act_types :
        method = "byalternatives"
    elif "bye" in act_types :
        method = "finished"
    elif "inform" in act_types:
        method = "byconstraints"
        for act in [uact for uact in user_act if uact["act"] == "inform"] :
            slots = [slot for slot, _ in act["slots"]]
            if "name" in slots :
                method = "byname"
    # dk449
    elif "restart" in act_types:
        method = "restart"
    elif not "inform" in act_types and not "deny" in act_types and ("request" in act_types or "confirm" in act_types):
        if lastInformedVenue != "":
            method = "byname"

    # discourseAct
    discourseAct = "none"
    if "silence" in act_types:
        discourseAct = "silence"
    elif "repeat" in act_types:
        discourseAct = "repeat"
    elif "thankyou" in act_types:
        discourseAct = "thankyou"
    elif "ack" in act_types:
        discourseAct = "ack"
    elif "hello" in act_types:
        discourseAct = "hello"
    elif "bye" in act_types:
        discourseAct = "bye"
            
    return informed_goals, denied_goals, requested_slots, method, discourseAct, lastInformedVenue

  
def Uacts(turn):
    '''
    Convert turn info to hypotheses

    :param turn:
    :type turn: dict

    :return: list -- converted hypotheses
    '''
    # return merged slu-hyps, replacing "this" with the correct slot
    mact = []
    if "dialog-acts" in turn["output"] :
        mact = turn["output"]["dialog-acts"]
    this_slot = None
    for act in mact :
        if act["act"] == "request" :
            this_slot = act["slots"][0][1]
    this_output = []
    for slu_hyp in turn['input']["live"]['slu-hyps'] :
        score = slu_hyp['score']
        this_slu_hyp = slu_hyp['slu-hyp']
        these_hyps =  []
        for  hyp in this_slu_hyp :
            for i in range(len(hyp["slots"])) :
                slot,_ = hyp["slots"][i]
                if slot == "this" :
                    hyp["slots"][i][0] = this_slot
            these_hyps.append(hyp)
        this_output.append((score, these_hyps))
    this_output.sort(key=lambda x:x[0], reverse=True)
    return this_output

class RuleBasedTracker(BeliefTracker):
    
    def __init__(self,domainString):
        super(RuleBasedTracker, self).__init__(domainString)
        
    def _updateBelief(self, turn):
        '''
        Update belief state

        :param turn:
        :type turn: dict
        
        :return: None
        '''
        track = self._addTurn(turn)
        return self._tobelief(self.prevbelief, track)
    
    def _addTurn(self, turn):
        '''
        Add turn info

        :param turn:
        :type turn: dict
        
        :return: None
        '''
        pass
    
    def _tobelief(self, prev_belief, track):
        '''
        Add up previous belief and current tracking result to current belief

        :param prev_belief:
        :type prev_belief: dict

        :param track:
        :type track: dict
       
        :return: dict -- belief state
        '''

        belief = {}
        for slot in Ontology.global_ontology.get_informable_slots_and_values(self.domainString):
            if slot in track['goal-labels']:
                infom_slot_vals = Ontology.global_ontology.get_informable_slot_values(self.domainString,slot)
                # su259: user simulator may issue a dontcare for all informable slots, not only system requestable
#                 if slot not in Ontology.global_ontology.get_system_requestable_slots(self.domainString):
#                     belief[slot] = dict.fromkeys(infom_slot_vals, 0.0)
#                 else:
                belief[slot] = dict.fromkeys(infom_slot_vals+['dontcare'], 0.0)
                for v in track['goal-labels'][slot]:
                    belief[slot][v] = track['goal-labels'][slot][v]
                belief[slot]['**NONE**'] = 1.0 - sum(belief[slot].values())
            else:
                belief[slot] = prev_belief['beliefs'][slot]

        belief['method'] = dict.fromkeys(Ontology.global_ontology.get_method(self.domainString), 0.0)
        for v in track['method-label']:
            belief['method'][v] = track['method-label'][v]
        belief['discourseAct'] = dict.fromkeys(Ontology.global_ontology.get_discourseAct(self.domainString), 0.0)
        for v in track['discourseAct-labels']:
            belief['discourseAct'][v] = track['discourseAct-labels'][v]
        belief['requested'] = dict.fromkeys(Ontology.global_ontology.get_requestable_slots(self.domainString), 0.0)
        for v in track['requested-slots']:
            belief['requested'][v] = track['requested-slots'][v]

        return {'beliefs': belief}

class BaselineTracker(RuleBasedTracker):
    """
    Record single hypothesis for each slot at each turn, 
    whose value is the top scoring value for that slot so far. 
    Note that this tracker does not handle goal constraint changes well as it ignores past states 
    but only uses the current state.
    """
    def __init__(self, domainString):
        super(BaselineTracker, self).__init__(domainString)
        self.restart()
        self.lastInformedVenue = ""        
        
    def _addTurn(self, turn):
        '''
        Add turn info

        :param turn:
        :type turn: dict
        
        :return: None
        '''
        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        # clear requested-slots that have been informed
        for act in mact :
            if act["act"] == "inform" :
                for slot,value in act["slots"]:
                    if slot in hyps["requested-slots"] :
                        hyps["requested-slots"][slot] = 0.0
        slu_hyps = Uacts(turn)
        
        requested_slot_stats = defaultdict(float)
        method_stats = defaultdict(float)
        goal_stats = defaultdict(lambda : defaultdict(float))
        discourseAct_stats = defaultdict(float)
        prev_method = "none"
        
        if len(hyps["method-label"].keys())> 0 :
            prev_method = hyps["method-label"].keys()[0]
        for score, uact in slu_hyps :
            informed_goals, denied_goals, requested, method, discourseAct, self.lastInformedVenue = labels(uact, mact, self.lastInformedVenue)
            # requested
            for slot in requested:
                requested_slot_stats[slot] += score
            if method == "none" :
                method = prev_method
            if method != "none" :
                method_stats[method] += score
            # goal_labels
            for slot in informed_goals:
                value = informed_goals[slot]
                goal_stats[slot][value] += score
            # discourseAct
            discourseAct_stats[discourseAct] += score            

        # pick top values for each slot
        for slot in goal_stats:
            curr_score = 0.0
            if (slot in hyps["goal-labels"]) :
                curr_score = hyps["goal-labels"][slot].values()[0]
            for value in goal_stats[slot]:
                score = goal_stats[slot][value]
                if score >= curr_score :
                    hyps["goal-labels"][slot] = {
                            value:clip(score)
                        }
                    curr_score = score
                    
        # joint estimate is the above selection, with geometric mean score
        goal_joint_label = {"slots":{}, "scores":[]}
        for slot in  hyps["goal-labels"] :
            (value,score), = hyps["goal-labels"][slot].items()
            if score < 0.5 :
                # then None is more likely
                continue
            goal_joint_label["scores"].append(score)
            goal_joint_label["slots"][slot]= value
            
        if len(goal_joint_label["slots"]) > 0 :
            geom_mean = 1.0
            for score in goal_joint_label["scores"] :
                geom_mean *= score
            geom_mean = geom_mean**(1.0/len(goal_joint_label["scores"]))
            goal_joint_label["score"] = clip(geom_mean)
            del goal_joint_label["scores"]
            
            hyps["goal-labels-joint"] = [goal_joint_label]
        
        for slot in requested_slot_stats :
            hyps["requested-slots"][slot] = clip(requested_slot_stats[slot])
            
        # normalise method_stats    
        hyps["method-label"] = normalise_dict(method_stats)
        # normalise discourseAct_stats
        hyps["discourseAct-labels"] = normalise_dict(discourseAct_stats)
        self.hyps = hyps 
        return self.hyps
    
    def restart(self):
        '''
        Reset the hypotheses
        '''
        super(BaselineTracker, self).restart()
        self.hyps = {"goal-labels":{}, "goal-labels-joint":[], "requested-slots":{}, "method-label":{}, "discourseAct-labels":{}}
    
class FocusTracker(RuleBasedTracker):
    """
    It accumulates evidence and has a simple model of how the state changes throughout the dialogue.
    Only track goals but not requested slots and method.
    """
    def __init__(self,domainString):
        super(FocusTracker, self).__init__(domainString)
        self.restart()
        self.lastInformedVenue = ""

    def _addTurn(self, turn):
        '''
        Add turn info

        :param turn:
        :type turn: dict
        
        :return: None
        '''
        hyps = copy.deepcopy(self.hyps)
        if "dialog-acts" in turn["output"] :
            mact = turn["output"]["dialog-acts"]
        else :
            mact = []
        slu_hyps = Uacts(turn)
       
        this_u = defaultdict(lambda : defaultdict(float))
        method_stats = defaultdict(float)
        requested_slot_stats = defaultdict(float)
        discourseAct_stats = defaultdict(float)
        for score, uact in slu_hyps :
            informed_goals, denied_goals, requested, method, discourseAct, self.lastInformedVenue = labels(uact, mact, self.lastInformedVenue)
            method_stats[method] += score
            for slot in requested:
                requested_slot_stats[slot] += score
            # goal_labels
            for slot in informed_goals:
                this_u[slot][informed_goals[slot]] += score
            discourseAct_stats[discourseAct] += score


        for slot in set(this_u.keys() + hyps["goal-labels"].keys()) :
            q = max(0.0,1.0-sum([this_u[slot][value] for value in this_u[slot]])) # clipping at zero because rounding errors
            if slot not in hyps["goal-labels"] :
                hyps["goal-labels"][slot] = {}
                
            for value in hyps["goal-labels"][slot] :
                
                hyps["goal-labels"][slot][value] *= q
            prev_values = hyps["goal-labels"][slot].keys()
            for value in this_u[slot] :
                if value in prev_values :
                    hyps["goal-labels"][slot][value] += this_u[slot][value]
                else :
                    hyps["goal-labels"][slot][value]=this_u[slot][value]
        
            hyps["goal-labels"][slot] = normalise_dict(hyps["goal-labels"][slot])
        
        # method node, in 'focus' manner:
        q = min(1.0,max(0.0,method_stats["none"]))
        method_label = hyps["method-label"]
        for method in method_label:
            if method != "none" :
                method_label[method] *= q
        for method in method_stats:
            if method == "none" :
                continue
            if method not in method_label :
                method_label[method] = 0.0
            method_label[method] += method_stats[method]
        
        if "none" not in method_label :
            method_label["none"] = max(0.0, 1.0-sum(method_label.values()))
        
        hyps["method-label"] = normalise_dict(method_label)

        # discourseAct (is same to non-focus)
        hyps["discourseAct-labels"] = normalise_dict(discourseAct_stats)

        # requested slots
        informed_slots = []
        for act in mact :
            if act["act"] == "inform" :
                for slot,value in act["slots"]:
                    informed_slots.append(slot)
                    
        for slot in set(requested_slot_stats.keys() + hyps["requested-slots"].keys()):
            p = requested_slot_stats[slot]
            prev_p = 0.0
            if slot in hyps["requested-slots"] :
                prev_p = hyps["requested-slots"][slot]
            x = 1.0-float(slot in informed_slots)
            new_p = x*prev_p + p
            hyps["requested-slots"][slot] = clip(new_p)
            
        self.hyps = hyps 
        return self.hyps
    
    
    def restart(self):
        '''
        Reset the hypotheses
        '''
        super(FocusTracker, self).restart()
        self.hyps = {"goal-labels":{},"method-label":{}, "requested-slots":{}}
    

def clip(x) :
    if x > 1:
        return 1
    if x<0 :
        return 0
    return x


def normalise_dict(x) :
    x_items = x.items()
    total_p = sum([p for k,p in x_items])
    if total_p > 1.0 :
        x_items = [(k,p/total_p) for k,p in x_items]
    return dict(x_items)




#END OF FILE
