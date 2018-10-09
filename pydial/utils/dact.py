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
dact.py - dialogue act specification
========================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Usage**: 
    >>> import dact
   
.. Note::

    Copied from dstc-ii code

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger`

************************
'''

import re
import string

from utils import ContextLogger
logger = ContextLogger.getLogger('')


class DactItem:
    '''
    Dialogue act specification

    :param slot: slot name
    :type slot: str
    :param op: comparative operation, e.g. '=' or '!='
    :type op: str
    '''
    def __init__(self, slot, op, val):
        self.slot = slot
        self.op = op
        self.val = val

        # if slot == "count":
        #     try:
        #         self.val = int(val)
        #     except ValueError:
        #         pass
        if val is not None and type(val) in [str, unicode] and len(val) > 0 and val[0] not in string.punctuation:
            self.val = val.lower()

    def match(self, other):
        '''
        Commutative operation for comparing two items.
        Note that "self" is the goal constraint, and "other" is from the system action.
        The item in "other" must be more specific. For example, the system action confirm(food=dontcare) doesn't match
        the goal with food=chinese, but confirm(food=chinese) matches the goal food=dontcare.

        If slots are different, return True.

        If slots are the same, (possible values are x, y, dontcare, !x, !y, !dontcare)s
            x, x = True
            x, y = False
            dontcare, x = True
            x, dontcare = False
            dontcare, dontcare = True

            x, !x = False
            x, !y = True
            x, !dontcare = True
            dontcare, !x = False
            dontcare, !dontcare = False

            !x, !x = True
            !x, !y = True
            !x, !dontcare = True
            !dontcare, !dontcare = True

        :param other:
        :return:
        '''
        if self.slot != other.slot:
            return True

        if self.val is None or other.val is None:
            logger.error('None value is given in comparison between %s and %s' % str(self), str(other))

        if self.op == '=' and other.op == '=':
            if self.val == other.val:
                return True
            if self.val == 'dontcare':
                return True
            return False

        elif self.op == '=' and other.op == '!=':
            if self.val == 'dontcare':
                return False
            elif self.val == other.val:
                return False
            else:
                return True

        elif self.op == '!=' and other.op == '=':
            if other.val == 'dontcare':
                return False
            elif self.val == other.val:
                return False
            else:
                return True

        else: # self.op == !=' and other.op == '!=':
            return True

    def __eq__(self, other):
        if self.slot == other.slot and self.op == other.op and self.val == other.val:
            return True
        return False

    def __hash__(self):
        return hash(repr((self.slot, self.op, self.val)))

    def __str__(self):
        return repr((self.slot, self.op, self.val))

    def __repr__(self):
        return repr((self.slot, self.op, self.val))


def _InParseAct(t):

    r = {}
    r['slots'] = []

    if t == "BAD ACT!!":
        r['act'] = 'null'
        return r

    #m = re.search('^(.*)\((.*)\)$',t.strip())
    m = re.search('^([^\(\)]*)\((.*)\)$',t.strip())
    if not m:
        r['act'] = 'null'
        return r

    r['act'] = m.group(1).strip()
    content = m.group(2)
    while len(content) > 0:
        m = re.search('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', content)
        if m:
            slot = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip("' ")
            items = DactItem(slot, op, val)
            content = re.sub('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', '', content)
            r['slots'].append(items)
            continue
        m = re.search('^([^,=]*)(!?=)\s*([^,]*)\s*,?', content)
        if m:
            slot = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip("' ")
            items = DactItem(slot, op, val)
            content = re.sub('^([^,=]*)(!?=)\s*([^,]*)\s*,?', '', content)
            r['slots'].append(items)
            continue
        m = re.search('^([^,]*),?', content)
        if m:
            slot = m.group(1).strip()
            op = None
            val = None
            items = DactItem(slot, op, val)
            content = re.sub('^([^,]*),?', '', content)
            r['slots'].append(items)
            continue
        raise RuntimeError, 'Cant parse content fragment: %s' % content

    return r


def _InParseActSet(t):

    r = {}
    r['slots'] = set()

    if t == "BAD ACT!!":
        r['act'] = 'null'
        return r

    #m = re.search('^(.*)\((.*)\)$',t.strip())
    m = re.search('^([^\(\)]*)\((.*)\)$',t.strip())
    if not m:
        r['act'] = 'null'
        return r
      
    r['act'] = m.group(1)
    content = m.group(2)
    while len(content) > 0:
        m = re.search('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', content)
        if m:
            slot = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip("' ")
            items = DactItem(slot, op, val)
            content = re.sub('^([^,!=]*)(!?=)\s*\"([^\"]*)\"\s*,?', '', content)
            r['slots'].add(items)
            continue
        m = re.search('^([^,=]*)(!?=)\s*([^,]*)\s*,?', content)
        if m:
            slot = m.group(1).strip()
            op = m.group(2).strip()
            val = m.group(3).strip("' ")
            items = DactItem(slot, op, val)
            content = re.sub('^([^,=]*)(!?=)\s*([^,]*)\s*,?', '', content)
            r['slots'].add(items)
            continue
        m = re.search('^([^,]*),?', content)
        if m:
            slot = m.group(1).strip()
            op = None
            val = None
            items = DactItem(slot, op, val)
            content = re.sub('^([^,]*),?', '', content)
            r['slots'].add(items)
            continue
        raise RuntimeError, 'Cant parse content fragment: %s' % content

    return r


def __ParseAct(t):

    r = {}
    r['slots'] = []

    if t == "BAD ACT!!":
      r['act'] = 'null'
      return r

    #m = re.search('^(.*)\((.*)\)$',t.strip())
    m = re.search('^([^\(\)]*)\((.*)\)$',t.strip())
    if (not m):
        r['act'] = 'null'
        return r

    r['act'] = m.group(1)
    content = m.group(2)
    while (len(content) > 0):
        m = re.search('^([^,=]*)=\s*\"([^\"]*)\"\s*,?',content)
        if (m):
            slot = m.group(1).strip()
            val = m.group(2).strip("' ")
            content = re.sub('^([^,=]*)=\s*\"([^\"]*)\"\s*,?','',content)
            r['slots'].append( [slot,val] )
            continue
        m = re.search('^([^,=]*)=\s*([^,]*)\s*,?',content)
        if (m):
            slot = m.group(1).strip()
            val = m.group(2).strip("' ")
            content = re.sub('^([^,=]*)=\s*([^,]*)\s*,?','',content)
            r['slots'].append( [slot,val] )
            continue
        m = re.search('^([^,]*),?',content)
        if (m):
            slot = m.group(1).strip()
            val = None
            content = re.sub('^([^,]*),?','',content)
            r['slots'].append( [slot,val] )
            continue
        raise RuntimeError,'Cant parse content fragment: %s' % (content)

    for slot_pair in r['slots']:
        if (slot_pair[1] == None):
            continue
        slot_pair[1] = slot_pair[1].lower()
        if slot_pair[0] == "count" :
            try :
                int_value = int(slot_pair[1])
                slot_pair[1] = int_value
            except ValueError:
                pass
    return r

def _ParseAct(raw_act_text, user=True):

    raw_act = __ParseAct(raw_act_text)
    final_dialog_act = []
    
      
    if raw_act['act'] == "select" and user :
      raw_act['act'] = "inform"
      
    main_act_type = raw_act['act']

    if raw_act['act'] == "request" or raw_act['act'] == "confreq":
        for requested_slot in [slot for slot, value in raw_act['slots'] if value == None] :
            final_dialog_act.append( {
                'act': 'request',
                'slots': [['slot',requested_slot]],
                })
            
        if raw_act['act'] == "confreq" :
            main_act_type = "impl-conf"

        else :
            main_act_type = "inform"
        
    elif (raw_act['act'] in ['negate','repeat','affirm','bye','restart','reqalts','hello','silence','thankyou','ack','help','canthear','reqmore']):
        if raw_act['act'] == "hello" and not user:
            raw_act['act'] = "welcomemsg"
        final_dialog_act.append( {
                'act': raw_act['act'],
                'slots': [],
                })
        main_act_type = 'inform'
    elif (raw_act['act'] not in ['inform','deny','confirm','select','null', 'badact']):
        print raw_act_text
        print raw_act
        raise RuntimeError,'Dont know how to convert raw act type %s' % (raw_act['act'])

    if raw_act['act'] == "confirm" and not user :
        main_act_type = "expl-conf"
        
    
    
    if raw_act['act'] == "select" and not user and "other" in [v for s,v in raw_act['slots']] :
      main_act_type = "expl-conf"
    
    if raw_act['act'] == "deny" and len(raw_act["slots"]) ==0 :
        final_dialog_act.append( {
                'act': "negate",
                'slots': [],
                })
    # Remove task=none
    # canthelps:
    if ["name","none"] in raw_act["slots"] and not user:
        other_slots = []
        only_names = [] # collect the names that are the only such venues
        for  slot,value in raw_act["slots"] :
          if value == "none" or slot=="other":
            continue
          if slot == "name!" :
            only_names.append(value)
            continue
          other_slots.append([slot,value])

        return [{
                'act': "canthelp",
                'slots': other_slots,
                }] + [{'act':"canthelp.exception", "slots":[["name", name]]} for name in only_names]
      
    elif not user and "none" in [v for _,v in raw_act["slots"]] :
      if raw_act["act"]!="inform" :
        return [{"act":"repeat","slots":[]}]
      none_slots = [s for s,v in raw_act["slots"] if v=="none"]
      name_value, = [v for s,v in raw_act["slots"] if s=="name"]
      other_slots = [[slot,value] for slot,value in raw_act["slots"] if value != "none"]
      final_dialog_act.append({
        'act':'canthelp.missing_slot_value',
        'slots':[['slot', none_slot] for none_slot in none_slots]+[['name',name_value]]
      })
      if other_slots:
        raw_act = ({'act':'inform', 'slots':other_slots})
      else :
        raw_act = {"slots":[],"act":"inform"}
      
    # offers
    if "name" in [slot for slot, value in raw_act["slots"]] and not user:
        name_value = [value for slot,value in raw_act["slots"] if slot=="name"]
        other_slots = [[slot,value] for slot,value in raw_act["slots"] if slot!="name"]

        final_dialog_act.append( {
                'act': "offer",
                'slots': [["name",name_value]]
                })
        raw_act['slots'] = other_slots

    
    
    # group slot values by type
    # try to group date and time into inform acts
    # put location fields in their own inform acts
    main_act_slots_dict = {}
    for (raw_slot_name,raw_slot_val) in raw_act['slots']:
        slot_name = raw_slot_name
        slot_val = raw_slot_val
        slot_group = slot_name
        if (slot_group not in main_act_slots_dict):
            main_act_slots_dict[slot_group] = {}
        if (slot_name not in main_act_slots_dict[slot_group]):
            main_act_slots_dict[slot_group][slot_name] = []
        if (slot_val not in main_act_slots_dict[slot_group][slot_name]):
            main_act_slots_dict[slot_group][slot_name].append(slot_val)
            
    for slot_group_name,slot_group_items in main_act_slots_dict.items():
        for slot,vals in slot_group_items.items():
            # if slot in ["task", "type"] :
            #     continue
            # we shouldn't skip this
            if slot == "" :
                slot = "this"
            if main_act_type == "deny" and len(vals) == 2 and "dontcare" not in vals :
                # deal with deny(a=x, a=y)
                false_value = vals[0]
                true_value = vals[1]
                final_dialog_act.append({
                                'act': "deny",
                                'slots': [[slot,false_value]],
                            })
                final_dialog_act.append({
                                'act': "inform",
                                'slots': [[slot,true_value]],
                            })
            else :
                for val in vals:
                    
                    
                    if val == None or val == "other":
                        
                        continue
                    
                    if len(slot)>0 and slot[-1] == "!" :
                        slot = slot[:-1]
                        slots = [ [slot,val] ]
                        final_dialog_act.append({
                                'act': "deny",
                                'slots': slots,
                                })
                    elif slot == "option" :
                        final_dialog_act.append({
                            'act':'giveoption',
                            'slots':[[requested_slot, val]]

                        })
                    else :
                        slots = [ [slot,val] ]
                        if ((slot,val) == ("this","dontcare")) and (main_act_type != "inform") :
                            continue
                        
                        final_dialog_act.append({
                                'act': ("inform" if slot=="count" else main_act_type),
                                'slots': slots,
                            })
                        


   
    if not user and len(final_dialog_act)==0 :
        final_dialog_act.append({"act":"repeat","slots":[]})
    return final_dialog_act


def ParseAct(raw_act_text, user=True):
    final = []
    for act_text in raw_act_text.split("|") :
        try:
            final += _ParseAct(act_text, user=user)
        except RuntimeError:
            pass # add nothing to final if junk act recieved
    return final


def inferSlotsForAct(uacts, ontology=None):
    '''
    Works out the slot from the ontology and value
    
    :param uacts:
    :param ontology:
    :return: user's dialogue acts
    '''
    for uact in uacts:
        for index in range(0, len(uact["slots"])):
            (slot, value) = uact["slots"][index]
            if slot == "this" and value != "dontcare":
                skipThis = False
                if ontology :
                    for s, vals in ontology["informable"].iteritems():
                        if value in vals:
                            if slot != "this":  # Already changed!
                                print "Warning: " + value + " could be for " + slot + " or " + s
                                skipThis = True
                            slot = s
                else :
                    slot = "type"  # default
                if not skipThis:
                    if slot == "this":
                        print "Warning: unable to find slot for value " + value
                        uact["slots"][index] = ("", "")
                    else:
                        uact["slots"][index] = (slot, value)
            uact["slots"][:] = list((s, v) for (s, v) in uact["slots"] if s != "" or v != "")
    return uacts

# if __name__ == '__main__':
#     import json
#     ontology = json.load(open("/home/dial/mh521/DSTC/GM/scripts/config/ontology_dstc2_da.json"))
#     act='inform(name="yu garden",phone="01223 248882")'
#     test = boostutil.transformAct(ParseAct(act, user=False), {}, ontology, False)
#     print json.dumps(test,indent=4)
    
   # act = ParseAct("inform(=restaurant)")
  #  ontology = json.load(open("scripts/config/ontology_Apr11.json","r"))
  #  act = inferSlotsForAct(act, ontology)
  #  print json.dumps(act)
    #print json.dumps(ParseAct("confreq(count=\"9\",type=restaurant,area=south,pricerange,option=cheap,option=expensive)", user=False), indent=4)



