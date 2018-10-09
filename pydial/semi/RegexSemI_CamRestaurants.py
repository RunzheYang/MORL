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

"""
RegexSemI_CamRestaurants.py - regular expression based CamRestaurants SemI decoder
===============================================================


HELPFUL: http://regexr.com

"""

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Jul 21 2016 lmr46   Refactoring, creating abstract class SemI
'''

import RegexSemI
import re,os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
from utils import ContextLogger
logger = ContextLogger.getLogger('')


class RegexSemI_CamRestaurants(RegexSemI.RegexSemI):
    """
    """
    def __init__(self, repoIn=None):
        RegexSemI.RegexSemI.__init__(self)  #better than super() here - wont need to be changed for other domains
        self.domainTag = "CamRestaurants"  #FIXME
        self.create_domain_dependent_regex() 

    def create_domain_dependent_regex(self):
        """Can overwrite any of the regular expressions set in RegexParser.RegexParser.init_regular_expressions(). 
        This doesn't deal with slot,value (ie domain dependent) semantics. For those you need to handcraft 
        the _decode_[inform,request,confirm] etc.
        """
        # REDEFINES OF BASIC SEMANTIC ACTS (ie those other than inform, request): (likely nothing needs to be done here)
        #eg: self.rHELLO = "anion"

        self._domain_init(dstring=self.domainTag)
            
        # DOMAIN DEPENDENT SEMANTICS:
        self.slot_vocab= dict.fromkeys(self.USER_REQUESTABLE,'')
        # FIXME: define slot specific language -  for requests
        #---------------------------------------------------------------------------------------------------
        self.slot_vocab["addr"] = "(address)" 
        self.slot_vocab["pricerange"] = "(price|cost)(\ ?range)*" 
        self.slot_vocab["area"] = "(area|location)" 
        self.slot_vocab["food"] = "(food)"
        self.slot_vocab["phone"] = "(phone(\ number)*|number|phone)"
        self.slot_vocab["postcode"] = "(postcode|post\ code)"
        self.slot_vocab["signature"] = "(signature)"
        self.slot_vocab["description"] = "(description)"
        self.slot_vocab["name"] = "(name)"
        #---------------------------------------------------------------------------------------------------
        # Generate regular expressions for requests:
        self._set_request_regex()
            
        # FIXME:  many value have synonyms -- deal with this here:
        self._set_value_synonyms()  # At end of file - this can grow very long
        self._set_inform_regex()


    def _set_request_regex(self):
        """
        """
        self.request_regex = dict.fromkeys(self.USER_REQUESTABLE)
        for slot in self.request_regex.keys():
            # FIXME: write domain dependent expressions to detext request acts
            self.request_regex[slot] = self.rREQUEST+"\ "+self.slot_vocab[slot]
            self.request_regex[slot] += "|(?<!"+self.DONTCAREWHAT+")(?<!want\ )"+self.IT+"\ "+self.slot_vocab[slot]
            self.request_regex[slot] += "|(?<!"+self.DONTCARE+")"+self.WHAT+"\ "+self.slot_vocab[slot]

        # FIXME:  Handcrafted extra rules as required on a slot to slot basis:
        self.request_regex["pricerange"] += "|(how\ much\ is\ it)"
        self.request_regex["food"] += "|(what\ (type\ of\ )*food)"
        self.request_regex["phone"] += "|(phone(\ num(ber)*)*)"
        self.request_regex["postcode"] += "|(postcode)|(post\ code)|(zip\ code)"
        self.request_regex["addr"] += "|(address)"
        self.request_regex["signature"] += "|(signature)|(best\ dish)|(specialty)|(recipe)"
        self.request_regex["description"] += "|(description)|(more\ information)|(more\ details)|(describe)"

    def _set_inform_regex(self):
        """
        """
        self.inform_regex = dict.fromkeys(self.USER_INFORMABLE)
        for slot in self.inform_regex.keys():
            self.inform_regex[slot] = {}
            for value in self.slot_values[slot].keys():
                self.inform_regex[slot][value] = self.rINFORM+"\ "+self.slot_values[slot][value]
                self.inform_regex[slot][value] += "|"+self.slot_values[slot][value] + self.WBG
                #self.inform_regex[slot][value] += "|((what|about|which)(\ (it\'*s*|the))*)\ "+slot+"(?!\ (is\ it))" 
                self.inform_regex[slot][value] += "|(\ |^)"+self.slot_values[slot][value] + "(\ (please|and))*"
                
                
                # FIXME:  Handcrafted extra rules as required on a slot to slot basis:
                if slot == "food":
                    self.inform_regex[slot][value] += "|(would\ like\ to\ eat\ (some|some\ *thing)*)\ "+\
                                                                                self.slot_values[slot][value]

            # FIXME: value independent rules: 
            nomatter = r"\ doesn\'?t matter"
            dontcare = r"((any(\ (kind|type)(\ of)?)?)|((i\ )?(don\'?t|do\ not)\ care\ (what|which|a?bout|for))(\ (kind|type)(\ of)?)?)\ "
            if slot == "pricerange":
                slot_term = r"(the\ )*(price|price(\ |-)*range)"
                self.inform_regex[slot]['dontcare'] = dontcare+slot_term
                self.inform_regex[slot]['dontcare'] += r"|" + slot_term + nomatter

            if slot == "area":
                LOCATION = r"(area|location|place|part\ of\ town)"
                slot_term = r"(the\ )*"+LOCATION
                self.inform_regex[slot]['dontcare'] = dontcare+slot_term
                self.inform_regex[slot]['dontcare'] += r"|" + slot_term + nomatter
                self.inform_regex[slot]['dontcare'] += r"|any(\ ?where)(\ is\ (fine|ok\b|good|okay))?"
                
            if slot == "food":
                slot_term = r"(the\ )*"+slot
                self.inform_regex[slot]['dontcare'] = dontcare+slot_term
                self.inform_regex[slot]['dontcare'] += r"|" + slot_term + nomatter



    def _generic_request(self,obs,slot):
        """
        """
        if self._check(re.search(self.request_regex[slot],obs, re.I)):
            self.semanticActs.append('request('+slot+')')

    def _generic_inform(self,obs,slot):
        """
        """
        
        
        
        DETECTED_SLOT_INTENT = False
        for value in self.inform_regex[slot]:
            if self._check(re.search(self.inform_regex[slot][value],obs, re.I)):
                #FIXME:  Think easier to parse here for "dont want" and "dont care" - else we're playing "WACK A MOLE!"
                ADD_SLOTeqVALUE = True
#                 # Deal with -- DONTWANT --:
#                 if self._check(re.search(self.rINFORM_DONTWANT+"\ "+self.slot_values[slot][value], obs, re.I)): 
#                     self.semanticActs.append('inform('+slot+'!='+value+')')  #TODO - is this valid?
#                     ADD_SLOTeqVALUE = False
#                 # Deal with -- DONTCARE --:
#                 if self._check(re.search(self.rINFORM_DONTCARE+"\ "+slot, obs, re.I)) and not DETECTED_SLOT_INTENT:
#                     self.semanticActs.append('inform('+slot+'=dontcare)')
#                     ADD_SLOTeqVALUE = False
#                     DETECTED_SLOT_INTENT = True
                # Deal with -- REQUESTS --: (may not be required...)
                #TODO? - maybe just filter at end, so that inform(X) and request(X) can not both be there?
                if ADD_SLOTeqVALUE and not DETECTED_SLOT_INTENT:
                    self.semanticActs.append('inform('+slot+'='+value+')')

    def _decode_request(self, obs):
        """
        """
        # if a slot needs its own code, then add it to this list and write code to deal with it differently
        DO_DIFFERENTLY= [] #FIXME 
        for slot in self.USER_REQUESTABLE:
            if slot not in DO_DIFFERENTLY:
                self._generic_request(obs,slot)
        # Domain independent requests:
        self._domain_independent_requests(obs)

        
    def _decode_inform(self, obs):
        """
        """
        # if a slot needs its own code, then add it to this list and write code to deal with it differently
        DO_DIFFERENTLY= [] #FIXME 
        for slot in self.USER_INFORMABLE:
            if slot not in DO_DIFFERENTLY:
                self._generic_inform(obs,slot)
        # Check other statements that use context
        self._contextual_inform(obs)

    def _decode_type(self,obs):
        """
        """
        # This is pretty ordinary - will just keyword spot for now since type really serves no point at all in our system
        if self._check(re.search(self.inform_type_regex,obs, re.I)):
            self.semanticActs.append('inform(type='+self.domains_type+')')


    def _decode_confirm(self, obs):
        """
        """
        #TODO?
        pass


    def _set_value_synonyms(self):
        """Starts like: 
            self.slot_values[slot] = {value:"("+str(value)+")" for value in DOMAINS_ontology["informable"][slot]}
            # Can add regular expressions/terms to be recognised manually:
        """
        #FIXME: 
        #---------------------------------------------------------------------------------------------------
        # TYPE:
        self.inform_type_regex = r"(restaurant|cafe|(want|looking for) food|(place|some(thing|where)) to eat)"
        # SLOT: area 
        slot = 'area'
        # {u'west': '(west)', u'east': '(east)', u'north': '(north)', u'south': '(south)', u'centre': '(centre)'}
        self.slot_values[slot]['north'] = "((the)\ )*(north|kings\ hedges|arbury|chesterton)"
        self.slot_values[slot]['east'] = "((the)\ )*(east|castle|newnham)"
        self.slot_values[slot]['west'] = "((the)\ )*(west|abbey|romsey|cherry hinton)"
        self.slot_values[slot]['south'] = "((the)\ )*(south|trumpington|queen ediths|coleridge)"
        self.slot_values[slot]['centre'] = "((the)\ )*(centre|center|downtown|central|market)"  # lmr46, added rule for detecting the center
#         self.slot_values[slot]['dontcare'] = "any(\ )*(area|location|place|where)"    
        # SLOT: pricerange
        slot = 'pricerange'
        # {u'moderate': '(moderate)', u'budget': '(budget)', u'expensive': '(expensive)'}
        self.slot_values[slot]['moderate'] = "(to\ be\ |any\ )*(moderat|moderate|moderately\ priced|mid|middle|average)"
        self.slot_values[slot]['moderate']+="(?!(\ )*weight)"
        self.slot_values[slot]['cheap'] = "(to\ be\ |any\ )*(budget|cheap|bargin|bargain|inexpensive|cheapest|low\ cost)"
        self.slot_values[slot]['expensive'] = "(to\ be\ |any\ )*(expensive|expensively|dear|costly|pricey)"
#         self.slot_values[slot]['dontcare'] = "any\ (price|price(\ |-)*range)"
        # SLOT: food
        slot = "food"
        # rely only on ontology values for now
        self.slot_values[slot]["asian oriental"] = "(oriental|asian)"
        self.slot_values[slot]["gastropub"] = "(gastropub|gastro pub)"
        self.slot_values[slot]["italian"] = "(italian|pizza|pasta)"
        self.slot_values[slot]["north american"] = "(american|USA)"
        
        #---------------------------------------------------------------------------------------------------

#END OF FILE
