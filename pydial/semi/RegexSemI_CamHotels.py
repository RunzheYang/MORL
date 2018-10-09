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
RegexSemI_CamHotels.py - regular expression based CamHotels SemI decoder
=========================================================================


HELPFUL: http://regexr.com

"""

import RegexSemI
import re,os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
from utils import ContextLogger
logger = ContextLogger.getLogger('')


class RegexSemI_CamHotels(RegexSemI.RegexSemI):
    """
    """
    def __init__(self, repoIn=None):
        RegexSemI.RegexSemI.__init__(self)  #better than super() here - wont need to be changed for other domains
        self.domainTag = "CamHotels"  #FIXME
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
        self.slot_vocab["pricerange"] = "(price)(\ ?range)"  
        self.slot_vocab["price"] = "(price|cost|expense)(?!(\ ?range))" 
        self.slot_vocab["area"] = "(area|location)" 
        self.slot_vocab["near"] = "(near)" 
        self.slot_vocab["kind"] = "(kind)" 
        self.slot_vocab["stars"] = "(stars|rating)"
        self.slot_vocab["phone"] = "((tele)?phone(\ number)*)"
        self.slot_vocab["postcode"] = "(postcode|post\ code)"
        self.slot_vocab["hasinternet"] = "(internet)"
        self.slot_vocab["hasparking"] = "(parking|car(\ ?park)|space\ available\ for\ parking)"
        self.slot_vocab["name"] = "(name)"
        #---------------------------------------------------------------------------------------------------
        
	    # FIXME:  many value have synonyms -- deal with this here:
        self._set_value_synonyms()  # At end of file - this can grow very long
        
	    # Generate regular expressions for requests:
        self._set_request_regex()
               
    	# Generate regular expressions for informs:
        self._set_inform_regex()

        # THis is causing an issue with binary slots: redefine from base class
        self.rINFORM_DONTWANT = r"i\ dont\ want\ "


    def _set_request_regex(self):
        """
        """
        self.request_regex = dict.fromkeys(self.USER_REQUESTABLE)
        for slot in self.request_regex.keys():
            # FIXME: write domain dependent expressions to detext request acts
            self.request_regex[slot] = self.rREQUEST+"\ "+self.slot_vocab[slot]
            self.request_regex[slot] += "|(?<!"+self.DONTCAREWHAT+")(?<!want\ )"+self.IT+"\ "+self.slot_vocab[slot]
            self.request_regex[slot] += "|(?<!"+self.DONTCARE+")"+self.WHAT+"\ "+self.slot_vocab[slot]
            self.request_regex[slot] += "|(may\ i|will\ it)\ have\ (the\ )?"+self.slot_vocab[slot]
            self.request_regex[slot] += "|i\ need\ the?\ "+self.slot_vocab[slot]

        # FIXME:  Handcrafted extra rules as required on a slot to slot basis:
        self.request_regex["pricerange"] += "|(how\ much\ is\ it)"
        self.request_regex["stars"] += "|(how\ many\ stars\ (does\ (it|the\ (hotel|place))\ have|has\ it\ got|is\ it))"
        self.request_regex["area"] += "|(where\ is\ (it|the\ hotel)(\ located)?)"
        self.request_regex["hasinternet"] += "|((does\ (the\ hotel|it)\ have\ good?)|(is\ there)|(has\ it\ got?))\ (wifi|internet(\ connection)?)"
        self.request_regex["hasinternet"] += "|is\ (it|the\ hotel)\ connected\ to\ the\ internet"
        self.request_regex["postcode"] += "|(postcode)|(post\ code)"
        

        newExpression = ""
        for value in self.slot_values["pricerange"].keys():
            if value != "dontcare":
                newExpression += self.slot_values["pricerange"][value] + "|"
        self.request_regex["pricerange"] += "|is\ (it|(the|that)\ hotel)\ ("+newExpression[:-1]+")"
#        print self.request_regex["pricerange"]
        
        newExpression = ""
        for value in self.slot_values["stars"].keys():
            if value != "dontcare":
                newExpression += self.slot_values["stars"][value] + "|"
        self.request_regex["stars"] += "|is\ (it|(the|that)\ hotel)\ a\ ("+newExpression[:-1]+")\ (one|hotel)"
        self.request_regex["stars"] += "|does\ (it|the\ hotel)\ have\ ("+newExpression[:-1]+")"
#        print self.request_regex["stars"]

        

    def _set_inform_regex(self):
        """
        """
        self.inform_regex = dict.fromkeys(self.USER_INFORMABLE)
        for slot in self.inform_regex.keys():
            self.inform_regex[slot] = {}
            for value in self.slot_values[slot].keys():
                self.inform_regex[slot][value] = self.rINFORM+"\ "+self.slot_values[slot][value]
                self.inform_regex[slot][value] += "|"+self.slot_values[slot][value] + self.WBG
                self.inform_regex[slot][value] += "|a\ (hotel\ with(\ a)*\ )*" +self.slot_values[slot][value]
                self.inform_regex[slot][value] += "|((what|about|which)(\ (it\'*s*|the))*)\ "+slot+"(?!\ (is\ (it|the\ hotel)))" 
                self.inform_regex[slot][value] += "|(\ |^)"+self.slot_values[slot][value] + "(\ (please|and))*"
#                print self.inform_regex[slot][value]

                # FIXME:  Handcrafted extra rules as required on a slot to slot basis:
                

            # FIXME: value independent rules: 
            if slot == "pricerange":
                self.inform_regex[slot]['dontcare'] = "(any|whatever)\ (costs|price|price(\ |-)*range)" 
                self.inform_regex[slot]['dontcare'] +=\
                        "|(don\'*t|do\ not)\ care\ (what|which|about|for)\ (the\ )*(price|price(\ |-)*range)"
#                self.inform_regex[slot]['dontcare'] += "(any|whatever)\ (price|price(\ |-)*range)"                         

            if slot == "hasparking":
                self.inform_regex[slot]['1'] += "|(i\ (need|want)\ (one|a\ hotel)\ with\ (car\ )*parking)"
                self.inform_regex[slot]['1'] += "|(i\ (need|want)\ (one|a\ hotel)\ (one\ )*where\ i\ can\ park)"
                self.inform_regex[slot]['1'] += "|(should\ have|with)((\ available)?\ space\ for)?\ parking(\ available)?"

                


    def _generic_request(self,obs,slot):
        """
        """
        if self._check(re.search(self.request_regex[slot],obs, re.I)):
            self.semanticActs.append('request('+slot+')')

    def _generic_inform(self,obs,slot):
        """
        """
        DETECTED_SLOT_INTENT = False
        for value in self.slot_values[slot].keys():
            if self._check(re.search(self.inform_regex[slot][value],obs, re.I)):
                #FIXME:  Think easier to parse here for "dont want" and "dont care" - else we're playing "WACK A MOLE!"
                ADD_SLOTeqVALUE = True
                # Deal with -- DONTWANT --:
                if self._check(re.search(self.rINFORM_DONTWANT+"\ "+self.slot_values[slot][value], obs, re.I)): 
                    raw_input(self.rINFORM_DONTWANT+"\ "+self.slot_values[slot][value])
                    self.semanticActs.append('inform('+slot+'!='+value+')')  #TODO - is this valid?
                    ADD_SLOTeqVALUE = False
                # Deal with -- DONTCARE --:
                if self._check(re.search(self.rINFORM_DONTCARE+"\ "+slot, obs, re.I)) and not DETECTED_SLOT_INTENT:
                    self.semanticActs.append('inform('+slot+'=dontcare)')
                    ADD_SLOTeqVALUE = False
                    DETECTED_SLOT_INTENT = True
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
            self.slot_values[slot] = {value:"("+str(value)+")" for value in domain_ontology["informable"][slot]}
            # Can add regular expressions/terms to be recognised manually:
        """
        #FIXME: 
        #---------------------------------------------------------------------------------------------------
        # TYPE:
        self.inform_type_regex = r"(hotel|motel|place to stay)"
        # SLOT: area 
        slot = 'area'
        # {u'west': '(west)', u'east': '(east)', u'north': '(north)', u'south': '(south)', u'centre': '(centre)'}
        self.slot_values[slot]['north'] = "((the)\ )*(north)"
        self.slot_values[slot]['east'] = "((the)\ )*(east)"
        self.slot_values[slot]['west'] = "((the)\ )*(west)"
        self.slot_values[slot]['south'] = "((the)\ )*(south)"
        self.slot_values[slot]['centre'] = "((the)\ )*(center|centre)"
        self.slot_values[slot]['dontcare'] = "((any(\ )*(area|location|place|where))|wherever)"
        # SLOT: pricerange
        slot = 'pricerange'
        # {u'moderate': '(moderate)', u'budget': '(budget)', u'expensive': '(expensive)'}
        self.slot_values[slot]['moderate'] = "(to\ be\ |any\ )*(moderate|moderately\ priced|mid|middle|average)"
        self.slot_values[slot]['moderate']+="(?!(\ )*weight)"
        self.slot_values[slot]['cheap'] = "(to\ be\ |any\ )*(budget|cheap|bargin|cheapest|low\ cost)"
        self.slot_values[slot]['expensive'] = "(to\ be\ |any\ )*(expensive|expensively|luxurious|dear|costly|pricey)"
        self.slot_values[slot]['dontcare'] = "((any|whatever)\ (costs|price|(price(\ |-)*range)))"
        # SLOT: near
        # rely on ontology for now
        # SLOT: hasparking 
        slot = 'hasparking'
        self.slot_values[slot]['1'] = "((should|must|needs\ to)\ have\ (car\ )*parking)"
        self.slot_values[slot]['1'] += "|(with\ (car\ )* parking)"
        self.slot_values[slot]['0'] = "should\ not\ have\ parking"
        self.slot_values[slot]['dontcare'] = "((don\'*t\ care)|((doesn\'*t|does\ not)\ matter))\ about\ parking"
        # SLOT: stars 
        slot = 'stars'
        self.slot_values[slot]['0'] = "(zero(\ (star|stars))?)|(no\ (star|stars))?"
        self.slot_values[slot]['4'] = "(four)(\ (star|stars))?"
        self.slot_values[slot]['3'] = "(three)(\ (star|stars))?"
        self.slot_values[slot]['2'] = "(two)(\ (star|stars))?"
        self.slot_values[slot]['dontcare'] = "((don\'*t\ care)|((doesn\'*t|does\ not)\ matter))\ about\ the\ (stars|rating|quality)"
        # SLOT: kind  
        slot = 'kind'
        self.slot_values[slot]['guesthouse'] = "guest(\ )*house"
        #---------------------------------------------------------------------------------------------------


#END OF FILE
