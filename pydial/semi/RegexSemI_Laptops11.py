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
RegexSemI_Laptops11.py - regular expression based Laptops11 SemI decoder
=================================================================


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


class RegexSemI_Laptops11(RegexSemI.RegexSemI):
    """
    """
    def __init__(self, repoIn=None):
        RegexSemI.RegexSemI.__init__(self)  #better than super() here - wont need to be changed for other domains
        self.domainTag = "Laptops11"  #FIXME
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
        #-------------------------------------------------------------------------------------------    
        self.slot_vocab["warranty"] = "(warranty)"
        self.slot_vocab["design"] = "(design)"
        self.slot_vocab["display"] = "(display)"
        self.slot_vocab["graphadaptor"] = "(graphadaptor)"
        self.slot_vocab["sysmemory"] = "(sysmemory)"
        self.slot_vocab["processor"] = "(processor)"
        self.slot_vocab["processorclass"] = "(processorclass)"
        self.slot_vocab["platform"] = "(platform)"
        self.slot_vocab["utility"] = "(utility)"
        self.slot_vocab["price"] = "(price|cost|expense)(?!(\ ?range))" 
        self.slot_vocab["pricerange"] = "(price\ ?range)" 
        self.slot_vocab["dimension"] = "(size|dimension)" 
        self.slot_vocab["weight"] = "(weight)(?!(\ ?range))"
        self.slot_vocab["weightrange"] = "(weight|weight\ ?range)"
        self.slot_vocab["batteryrating"] = "(battery|battery\ ?rating|rating\ of\ the\ (laptops\ )*battery)"
        self.slot_vocab["isforbusinesscomputing"] = "(business\ computing|gaming|fun|games|work|play)"
        self.slot_vocab["drive"] = "(drive|hard\ *drive)(?!(\ ?range))"
        self.slot_vocab["driverange"] = "(hard\ *)*drive(\ ?range)"
        self.slot_vocab["name"] = "(name)"
        self.slot_vocab["family"] = "(family|class)"
        #-------------------------------------------------------------------------------------------    
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
        self.request_regex["price"] += "|(how\ much\ is\ it)"
        self.request_regex["dimension"] += "|(how\ big\ is\ it)"
        self.request_regex["weightrange"] += "|(how\ heavy\ is\ it)"

    def _set_inform_regex(self):
        """
        """
        self.inform_regex = dict.fromkeys(self.USER_INFORMABLE)
        for slot in self.inform_regex.keys():
            self.inform_regex[slot] = {}
            for value in self.slot_values[slot].keys():
                self.inform_regex[slot][value] = self.rINFORM+"\ "+self.slot_values[slot][value]
                self.inform_regex[slot][value] += "|"+self.slot_values[slot][value] + self.WBG
                self.inform_regex[slot][value] += "|a\ (laptop\ with(\ a)*\ )*" +self.slot_values[slot][value]
                self.inform_regex[slot][value] += "|((what|about|which)(\ (it\'*s*|the))*)\ "+slot+"(?!\ (is\ it))" 
                self.inform_regex[slot][value] += "|(\ |^)"+self.slot_values[slot][value] + "(\ (please|and))*"


                # FIXME:  Handcrafted extra rules as required on a slot to slot basis:

            # FIXME: value independent rules: 
            if slot == "pricerange":
                self.inform_regex[slot]['dontcare'] = "any\ (price|price(\ |-)*range)" 
                self.inform_regex[slot]['dontcare'] +=\
                        "|(don\'*t|do\ not)\ care\ (what|which|about|for)\ (the\ )*(price|price(\ |-)*range)"
            if slot == "weightrange":
                self.inform_regex[slot]['dontcare'] = "any\ (weight|weight(\ |-)*range)"
                self.inform_regex[slot]['dontcare'] +=\
                        "|(don\'*t|do\ not)\ care\ (what|which|about|for)\ (the\ )*(weight|weight(\ |-)*range)"
                self.inform_regex[slot]['dontcare'] += "|"+r"((dont\ care\ how\ heavy){1,1}\ it\ is)"
            if slot == "batteryrating":
                self.inform_regex[slot]['dontcare'] = "any\ (battery|battery(\ |-)*(range|rating))"
                self.inform_regex[slot]['dontcare'] +=\
                        "|(don\'*t|do\ not)\ care\ (what|which|about|for)\ (the\ )*(battery|battery(\ |-)*(range|rating))"
            if slot == "driverange":
                self.inform_regex[slot]['dontcare'] = "any\ (hard\ )*(drive|drive(\ |-)*(range|rating))"
                self.inform_regex[slot]['dontcare'] +=\
                    "|(don\'*t|do\ not)\ care\ (what|which|about|for)\ (the\ )*(hard\ )*(drive|drive(\ |-)*(range|rating))"


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
        #-------------------------------------------------------------------------------------------    
        # TYPE:
        self.inform_type_regex = r"(laptop|computer)"
        # SLOT: family
        slot = 'family'
        # {u'satellite': '(satellite)', u'satellite pro': '(satellite pro)', u'tecra': '(tecra)', u'portege': '(portege)'}
        self.slot_values[slot]['satellite'] = "((to\ be\ |in|any|of)\ )*(satellite)(?!((\ |-)pro))"
        self.slot_values[slot]['satellite pro'] = "((to\ be\ |in|any|of)\ )*((sat|satellite)(\ |-)pro)"
        self.slot_values[slot]['tecra'] = "((to\ be\ |in|any|of)\ )*(tecra)"
        self.slot_values[slot]['portege'] = "((to\ be\ |in|any|of)\ )*(portege)"
        self.slot_values[slot]['dontcare'] = "any\ family"
        # SLOT: pricerange
        slot = 'pricerange'
        # {u'moderate': '(moderate)', u'budget': '(budget)', u'expensive': '(expensive)'}
        self.slot_values[slot]['moderate'] = "(to\ be\ |any\ )*(moderate|moderately\ priced|mid|middle|average)"
        self.slot_values[slot]['moderate']+="(?!(\ )*weight)"
        self.slot_values[slot]['budget'] = "(to\ be\ |any\ )*(budget|cheap|bargin|cheapest|low\ cost)"
        self.slot_values[slot]['expensive'] = "(to\ be\ |any\ )*(expensive|expensively|dear|costly|pricey)"
        self.slot_values[slot]['dontcare'] = "any\ (price|price(\ |-)*range)"
        # SLOT: batteryrating
        slot = 'batteryrating'
        # {u'exceptional': '(exceptional)', u'good': '(good)', u'standard': '(standard)'}
        self.slot_values[slot]['exceptional'] = "(to\ be\ |any\ )*(exceptional|best)"
        self.slot_values[slot]['good'] = "(to\ be\ |any\ )*(good\ battery|battery\ (rating\ )* good|good)(?! (\ *bye))"
        self.slot_values[slot]['standard'] = "(to\ be\ |any\ )*(standard)"
        self.slot_values[slot]['dontcare'] = "any\ (battery|battery(\ |-)*(range|rating))"
        # SLOT: weightrange
        slot = 'weightrange'
        # {u'light weight': '(light weight)', u'mid weight': '(mid weight)', u'heavy': '(heavy)'}
        self.slot_values[slot]['light weight'] = "(to\ be\ |any\ )*(light(-weight|\ weight)*|light\ (would\ be|is)\ good)"
        self.slot_values[slot]['mid weight'] = "(to\ be\ |any\ )*(mid(\ ||-)|middle|medium|average|moderate)(weight)"
        self.slot_values[slot]['heavy'] = "(?<!(how\ ))(to\ be\ |any\ )*(heavy)"
        self.slot_values[slot]['dontcare'] = "any\ (weight|weight(\ |-)*range)|"
        # SLOT: isforbusinesscomputing
        slot = 'isforbusinesscomputing'
        # {u'1': '(1)', u'0': '(0)'}
        WORK ="(work|business|(my\ )*job)"
        self.slot_values[slot]['1'] = "((?<!(is\ it\ ))((used\ )*for\ "+WORK+"))"
        # Next 2 create problems - copied from Phoenix basically. --TODO - DELETE THEM
        #self.slot_values[slot]['1'] += "|(?<!(is\ it\ ))(i\ want\ it\ for\ "+WORK+"(\ computing)*)"
        #self.slot_values[slot]['1'] += "|((it\'*s*|it\ is)\ for\ "+WORK+"(\computing)*))"
        self.slot_values[slot]['0'] = "(no\ business\ computing|not\ (used\ )*(for\ )*business(\ computing)*"
        self.slot_values[slot]['0']+="|i\ want\ a\ gaming(\ laptop)*|(i\ want\ it\ )*for\ (fun|games|play|gaming|leisure))"
        # SLOT: driverange
        slot = 'driverange'
        # {u'small': '(small)', u'large': '(large)', u'medium': '(medium)'}
        self.slot_values[slot]['small'] = "(small|little)"
        self.slot_values[slot]['large'] = "(large|big|lots)(?!(\ is\ it))"
        self.slot_values[slot]['medium'] = "(medium|average)"
        self.slot_values[slot]['dontcare'] = "any\ (drive|drive(\ |-)*(range|rating))"
        # SLOT: processorclass 
        slot = 'processorclass'
        # SLOT: platform 
        slot = 'platform'
        # SLOT: sysmemory
        slot = 'sysmemory'
        # SLOT: warranty
        slot = 'warranty'
        #-------------------------------------------------------------------------------------------    



#END OF FILE
