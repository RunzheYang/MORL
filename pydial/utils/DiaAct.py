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
DiaAct.py - dialogue act specification that extends dact.py
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Usage**: 
    >>> import DiaAct   
   
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.dact`

************************

'''

__author__ = "cued_dialogue_systems_group"
from collections import defaultdict


import dact, copy
from utils import ContextLogger
logger = ContextLogger.getLogger('')

actTypeToItemFormat = defaultdict(int)
known_format = {'hello':0,
                       'bye':0,
                       'inform':2,
                       'request':1,
                       'confreq':1,
                       'select':3,
                       'affirm':2,
                       'negate':2,
                       'deny':3,
                       'repeat':0,
                       'confirm':2,
                       'reqalts':2,
                       'null':0,
                       'badact':0,
                       'thankyou':0}
'''
**within class dictionary**

0 signals no dia item, eg hello()
1 signals slot, but no value, eg request(bar)
2 signals slot value pair, eg inform(food="Italian")
3 signals 2 slot value pairs, eg deny(food="Italian", food="Chinese")
'''
for act in known_format:
    actTypeToItemFormat[act] = known_format[act]

# actTypeToItemFormat = {'HelloAct':0,
#                        'ByeAct':0,
#                        'InformAct':2,
#                        'RequestAct':1,
#                        'ConfReqAct':1,
#                        'SelectAct':3,
#                        'AffirmAct':2,
#                        'NegateAct':2,
#                        'DenyAct':3,
#                        'RepeatAct':0,
#                        'ConfirmAct':2,
#                        'ReqAltsAct':2,
#                        'NullAct':0,
#                        'BadAct':0}


class DiaAct(object):
    '''
    Dialogue act class.

    self.dact = ``{'act': acttype,'slots': [(slot1, op1, value1), ..])}``

    :param act: dialogue act in string

    .. todo:: SummaryAction is not implemented.

    ''' 

    def __init__(self, act):
        # parsed is a dict of lists. 'slots' list contains dact items
        parsed = dact._InParseAct(act)
        self.act = parsed['act']
        self.items = parsed['slots']
        
        self.prompt = None


    def append(self, slot, value, negate=False):
        '''
        Add item to this act avoiding duplication

        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: semantic operation is negation or not
        :type negate: bool [Default=False]
        :return:
        '''
        op = '='
        if negate:
            op = '!='
        self.items.append(dact.DactItem(slot, op, value))

    def append_front(self, slot, value, negate=False):
        '''
        Add item to this act avoiding duplication

        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: operation is '=' or not?  False by default.
        :type negate: bool
        :return:
        '''
        op = '='
        if negate:
            op = '!='
        self.items = [dact.DactItem(slot, op, value)] + self.items

    def contains_slot(self, slot):
        '''
        :param slot: slot name
        :type slot: str
        :returns: (bool) answering whether self.items mentions slot
        '''
        for item in self.items:
            # There was an issue here with == using the dact.py __eq__ method, since each item is a dact.DactItem() 
            #if slot == item.slot:
            if slot == str(item.slot):
                return True
        return False

    def get_value(self, slot, negate=False):
        '''
        :param slot: slot name
        :type slot: str
        :param negate: relating to semantic operation, i.e slot = or slot !=.
        :type negate: bool - default False
        :returns: (str) value
        '''
        value = None
        for item in self.items:
            if slot == item.slot and negate == (item.op == '!='):
                if value is not None:
                    logger.warning('DiaAct contains multiple values for one slot: ' + str(self))
                else:
                    value = item.val
        return value

    def get_values(self, slot, negate=False):
        '''
        :param slot: slot name
        :type slot: str
        :param negate: - semantic operation
        :type negate: bool - default False
        :returns: (list) values in self.items
        '''
        values = []
        for item in self.items:
            if slot == item.slot and negate == (item.op == '!='):
                values.append(item.val)
        return values

    def contains(self, slot, value, negate=False):
        '''
        :param slot: None
        :type slot: str
        :param value: None
        :type value: str
        :param negate: None
        :type negate: bool - default False
        :returns: (bool) is full semantic act in self.items?
        '''
        op = '='
        if negate:
            op = '!='
        item = dact.DactItem(slot, op, value)
        return item in self.items

    def has_conflicting_value(self, constraints):
        '''
        :param constraints: as  [(slot, op, value), ...]
        :type constraints: list
        :return: (bool) True if this DiaAct has values which conflict with the given constraints. Note that consider
                 only non-name slots.
        '''
        for const in constraints:
            slot = const.slot
            op = const.op
            value = const.val
            if slot == 'name' or value == 'dontcare':
                continue

            this_value = self.get_value(slot, negate=False)
            if op == '!=':
                if this_value in [value]:
                    return True
            elif op == '=':
                if this_value not in [None, value]:
                    return True
            else:
                logger.error('unknown constraint operator exists: ' + str(const))
        return False

    def getDiaItemFormat(self):
        '''
        :param None: None
        :type None: based on self.items
        :returns: the number of arguments of this diaact type. e.g. act: 0, act(slot): 1, act(slot,value): 2
        '''
        if len(self.items) == 0:
            return 0
        dip = self.items[0]
        if not dip.slot:
            return 0
        if dip.val == '':
            return 1
        else:
            return 2

    def to_string(self):
        '''
        :param None:
        :returns: (str) semantic act
        '''
        s = ''
        s += self.act + '('
        for i, item in enumerate(self.items):
            if i != 0:
                s += ','
            if item.slot is not None:
                s += item.slot
            if item.val is not None:
                s = s+item.op+'"'+str(item.val)+'"'
        s += ')'
        return s

    def __eq__(self, other):
        '''
        :param other:
        :return: True if this DiaAct is equivalent to other. Items can be reordered.
        '''
        if self.act != other.act:
            return False
        if len(self.items) != len(other.items):
            return False
        for i in self.items:
            if i not in other.items:
                return False
        for i in other.items:
            if i not in self.items:
                return False
        return True

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()


class DiaActWithProb(DiaAct):
    '''
    self.dact = ``{'act': acttype,'slots': set([(slot1, value1), ..])}``

    :param act: dialogue act
    :type act: str

    .. todo:: Parser is not complete. Attached probability P_Au_O cannot be parsed.
           SummaryAction is not implemented.
           Emphasis and operator are not implemented.
    '''
    def __init__(self, act):
        if isinstance(act,DiaAct):
            if isinstance(act,DiaActWithProb):
                self.act = copy.deepcopy(act.act)
                self.items = copy.deepcopy(act.items)
                self.P_Au_O = act.P_Au_O
            else:
                self.act = copy.deepcopy(act.act)
                self.items = copy.deepcopy(act.items)
                self.P_Au_O = 1.0
        else:
            dia_act = dact._InParseAct(act)
            self.act = dia_act['act']
            self.items = dia_act['slots']
            self.P_Au_O = 1.0

    def __str__(self):
        s = self.to_string()
        
        if self.P_Au_O > 0:
            s += ' {'+str(self.P_Au_O)+'}'
        return s
    
    def __lt__(self, other):
        return self.P_Au_O > other.P_Au_O

# END OF FILE
