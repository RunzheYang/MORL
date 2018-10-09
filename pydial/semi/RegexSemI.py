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
RegexSemI.py - Regular expressions SemI parser base class
==========================================================

.. note::

    This implementation is based on the following assumptions:
    
    - obs can be a ASR n-best list - potential sentence inputs each with a probability (Currently - no probabilities - will have to slightly amend code to deal with these.)
    - will only output text semantic acts (plus probs maybe) -- wont be instances of DiaAct for example


.. warning::

    Remember that this is the base class for all of the regex parsers. Making changes here could possibly fix a parser
    in your domain, but (at worst!) break/weaken parsers in all other domains! i.e. -- 1st, 2nd and 3rd approaches
    should be to tweak the derived class for the domain of interest. You can redefine anything in the derived class.
    
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :class:`semi.SemI.SemI` |.|
    import :class:`semi.SemIContextUtils` |.|
    import :mod:`ontology.Ontology`


"""

__author__ = "cued_dialogue_systems_group"

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Jul 21 2016 lmr46   Refactoring, creating inheritance and grouping functionalities together
'''

import re,os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
from utils import ContextLogger
from ontology import Ontology
logger = ContextLogger.getLogger('')
from SemI import SemI
import SemIContextUtils as contextUtils

class RegexSemI(SemI):
    """Is a  base class for each domains Regular Expressions based semantic parser. Primary focus is on **users intent**.
        The derived semantic parsers of each domain can deal with the constraints (slots,values).
    """
    def __init__(self):
        # test if regex describing the entity at hand has been set in child class. used for request alternatives
        try:
            self.rTYPE
        except AttributeError:
            self.rTYPE = "(thereisnovalueyet)"
            logger.warning("No rTYPE regex has been set.")
            
        self.domainTag = None
            
        self.init_regular_expressions()

    def _domain_init(self, dstring):
        '''Will be used by all classes that extend this
        '''
        self.USER_REQUESTABLE = Ontology.global_ontology.get_requestable_slots(dstring)
        self.USER_INFORMABLE = Ontology.global_ontology.get_informable_slots(dstring)
        self.domains_type = Ontology.global_ontology.get_type(dstring)
        # Generate regular expressions for informs:
        self.slot_values = dict.fromkeys(self.USER_INFORMABLE)
        for slot in self.slot_values.keys():
            slot_values = Ontology.global_ontology.get_informable_slot_values(dstring, slot)
            self.slot_values[slot] = {value:"("+str(value)+")" for value in slot_values}

    # Optional arguments sys_act and turn here, needed only by DLSemI
    def decode(self, obs, sys_act=None, turn=None):
        """Assumes input is either a single string or a list of tuples (string, prob) as in a nbest list.
        """ 
        # bundle up into a list if a single sentence (ie string) is input
        if isinstance(obs,str):
            obs = [obs]
        if not isinstance(obs,list):
            logger.error("Expecting a list or string as input") 
       
        all_hyps = []
        for ob in obs:
            if isinstance(ob,tuple):
                sentence,sentence_prob = ob[0],ob[1]
            elif isinstance(ob,str):
                sentence,sentence_prob = ob, None
            else:
                logger.error("For decoding, expected either a str or (str,probability) tuple") 
            assert(isinstance(sentence,str) or isinstance(sentence,unicode))
            all_hyps.append((self.decode_single_hypothesis(sentence, sys_act),sentence_prob)) 
        return self.combine_parses(all_hyps)
        #"""

    def combine_parses(self, nbest_parses):
        """TODO - should return a list of tuples (semantic act, prob) - in order
        """
        #TODO  JUST A HACK FOR NOW - needs to combine probs and order to get single most likely semantic hypothesis
        # Should probably also be ammending self.clean() or doing what it does here or ...
        # will work for now on assumption that there was only 1 sentence (not an nbest list) to parse anyway
        return nbest_parses 

    def decode_single_hypothesis(self,obs, sys_act = None):
        """
        :param: (str) obs - sentence (an ASR hypothesis)
        """ 
        self.semanticActs = []
        # run the obs thru all possible semantic acts:
        self._decode_request(obs)
        self._decode_affirm(obs)
        self._decode_inform(obs)
        self._decode_confirm(obs)
        self._decode_hello(obs)
        self._decode_negate(obs)
        self._decode_repeat(obs)
        self._decode_reqalts(obs)
        self._decode_bye(obs)
        self._decode_type(obs)
        # probably need to then do some cleaning on acts in semanticActs
        self.clean(sys_act)
        return self.semanticActs

    def init_regular_expressions(self):
        """
        """
        self.rHELLO = "(\b|^|\ )(hi|hello)\s"
        self.rNEG =  "(\b|^|\ )(no\b|wrong|incorrect|error)|not\ (true|correct|right)\s" 
        self.rAFFIRM = "(yes|yeah|(\b|^)ok\b|(\b|^)OK\b|okay|sure|(that('?s| is) )?(?<!not\ )(?<!no\ )(right|correct|confirm))" 
        self.rBYE = "(\b|^|\ )(bye|goodbye|that'?s?\ (is\ )*all)(\s|$|\ |\.)"
        self.GREAT = "(great|good|awesome)"
        self.HELPFUL = "(that((\')?s|\ (is|was))\ (very\ )?helpful)"
        self.THANK = "(thank(s|\ you)(\ (very|so)\ much)?)"
        self.rTHANKS = r"(^(\ )*)((" + self.GREAT + "\ )?(" + self.HELPFUL + "\ )?"+self.THANK+"(\ " + self.HELPFUL + ")?|(" + self.GREAT + "\ )?" + self.HELPFUL + "|" + self.GREAT + ")((\ )*$)"
        self.rREQALTS = "(\b|^|\ )((something|anything)\ else)|(different(\ one)*)|(another\ one)|(alternatives*)"
        self.rREQALTS += "|(other options*)|((don\'*t|do not) (want|like)\ (that|this)(\ one)*)" 
        self.rREQALTS += "|(others|other\ "+self.rTYPE+"(s)?)"
        self.rREPEAT = "(\b|^|\ )(repeat\ that)|(say\ that\ again)" 
        # The remaining regex are for the slot,value dependent acts - and so here in the base class are \
        # just aiming to catch intent.
        # REQUESTS:
        self.WHAT = "(what\'*s*|which|does|where)(\ (its|the))*"
        self.IT = "(it\'*s*|it\ have|is\ it\'*s*|is\ (the|their))(\ for)*"
        self.CYTM = "(can\ you\ tell\ me\ (the|it\'*s|their))"
        self.CIG = "(can\ I\ get\ (the|it\'*s|their))"
        self.NEGATE ="((i\ )*(don\'?t|do\ not|does\ not|does\'?nt)\ (care|mind|matter)(\ (about|what))*(\ (the|it\'?s*))*)"
        self.DONTCARE = "(i\ dont\ care)"#Cant create variable lengths with negative lookback... else merge following:
        self.DONTCAREWHAT = "(i\ dont\ care\ what\ )"
        self.DONTCAREABOUT = "(i\ dont\ care\ about\ )"
        self.rREQUEST = r"(\b|^|\ )(?<!"+self.DONTCARE+")("+self.WHAT+"\ "+self.IT+"|"+self.CYTM+"|"+self.CIG+")"
        # INFORMS:
        self.WANT = "(what\ about|want|have|need|looking\ for|used\ for)(\ a(n)?)*"
        self.WBG = "(\ ((would|seems\ to)\ be\ (good|nice)($|[^\?]$)|seems\ (good|nice)($|[^\?]$)))"
        self.rINFORM = "(\b|^|\ )"+self.WANT
        self.rINFORM_DONTCARE = self.DONTCARE+r"((what|which|about)(\ (it\'*s*|the))*)+" 
        self.rINFORM_DONTWANT = r"(((i\ )*(don\'*t\ want))|it\ (shouldn\'*t|should\ not)\ (have|be))+" 
        # Contextual dontcares: i.e things that should be labelled inform(=dontcare)
        self.rCONTEXTUAL_DONTCARE = r"(anything(?!\ else)|((any$|any\ kind)|(i\ )*(don\'?t|do\ not)\ (care|know))($|(?!\ (a?bout|of|what))|(\ (a?bout|of|what)\ (type|kind)(?!\ of))|\ a?bout\ (that|this))|(any(thing)?\ (is\ )*(fine|ok\b|okay|will\ do))($|\ and|\ but)|(it )?(doesn\'?t|does not) matter)+" 
        # The following are NOT regular expresions, but EXACT string matching:
        self.COMMON_CONTEXTUAL_DONTCARES = ["i dont care","any","anything", "i dont mind"]
        self.COMMON_CONTEXTUAL_DONTCARES += ["it doesn\'t matter", "dont care"]



    def _decode_request(self,obs):
        """TO BE DEFINED IN DOMAIN SPECIFIC WAY IN DERIVED CLASS"""
        pass

    def _decode_inform(self,obs):
        """TO BE DEFINED IN DOMAIN SPECIFIC WAY IN DERIVED CLASS"""
        pass

    def _decode_confirm(self, obs):
        """TO BE DEFINED IN DOMAIN SPECIFIC WAY IN DERIVED CLASS"""
        pass

    def _decode_type(self,obs):
        """TO BE DEFINED IN DOMAIN SPECIFIC WAY IN DERIVED CLASS"""
        pass

    def _decode_hello(self,obs):
        """
        """  
        if self._check(re.search(self.rHELLO,obs, re.I)): #DEL is not None:
            self.semanticActs.append('hello()')

    def _decode_negate(self, obs):
        """
        """
        if self._check(re.search(self.rNEG,obs, re.I)): #DEL  is not None:
            self.semanticActs.append('negate()')  #TODO - is this a used act? syntax: neg()?
        if obs in ['no','wrong']:  # deal with negate a little differently
            self.semanticActs.append('negate()')

    def _decode_affirm(self,obs):
        """
        """ 
        if self._check(re.search(self.rAFFIRM,obs, re.I)): #DEL is not None:
            self.semanticActs.append('affirm()')

    def _decode_bye(self, obs):
        """
        """
        if self._check(re.search(self.rBYE,obs, re.I)): #DEL is not None:
            self.semanticActs.append('bye()')
#         elif self._check(re.search(self.rTHANKS,obs,re.I)):
#             self.semanticActs.append('bye()')

    def _decode_reqalts(self,obs):
        """
        """
        if self._check(re.search(self.rREQALTS,obs, re.I)): #DEL is not None:
            self.semanticActs.append('reqalts()')

    def _decode_repeat(self, obs):
        """
        """
        if self._check(re.search(self.rREPEAT,obs, re.I)): #DEL is not None:
            self.semanticActs.append('repeat()')

    def clean(self, sys_act = None):
        """
        """
        #TODO - deal with duplicates, probabilities, others?
        tempActs = {}
        
        #first add context to user acts
        for act in self.semanticActs:
            [act] = contextUtils._add_context_to_user_act(sys_act,[[act,1.0]],self.domainTag)
            act = act[0]
            intent, slotValues = self._parseDialogueAct(act)
            if intent not in tempActs:
                tempActs[intent] = slotValues
            else:
                for slot in slotValues:
                    if slot in tempActs[intent]:
                        if slotValues[slot] != tempActs[intent][slot]:
                            logger.warning('Sematic decoding of input lead to different interpretations within one hypothesis. Slot {} has values {} and {}.'.format(slot,slotValues[slot],tempActs[intent][slot]))
                    else:
                        tempActs[intent][slot] = slotValues[slot]
        #x
        hypos = []              
        for intent in tempActs:
            if len(tempActs[intent]):
                (key, value) = tempActs[intent].items()[0]
                if value is not None:
                    hypos.append('{}({})'.format(intent,','.join('%s=%s' % (key, value) for (key, value) in tempActs[intent].items())))
                else:
                    hypos.append('{}({})'.format(intent,','.join('%s' % (key) for (key, value) in tempActs[intent].items())))
            else:
                logger.warning("intent {} found in input without arguments")
        self.semanticActs =  "|".join(hypos)

    def _contextual_inform(self,obs):
        """
        """
        # Statements that are contextual/implicit (ie dont mention slot or value explicitly):
        if self._check(re.search(self.rCONTEXTUAL_DONTCARE, obs, re.I)): #DEL is not None: 
            self.semanticActs.append('inform(=dontcare)')
        if self._exact_match(self.COMMON_CONTEXTUAL_DONTCARES, obs):
            self.semanticActs.append('inform(=dontcare)')

    def _exact_match(self, strings, obs):
        """
        """
        if obs.lstrip().lower().replace("'","") in strings:
            return True
        return False

    def _domain_independent_requests(self,obs):
        """  
        """
        rDOM_IN_REQ_NAME = r"((what(s*)|what\ is)\ it called)"
        if self._check(re.search(rDOM_IN_REQ_NAME, obs, re.I)): #DEL is not None: 
            self.semanticActs.append('request(name)')

    def _check(self,re_object):
        """
        """
        if re_object is None:
            return False
        for o in re_object.groups():
            if o is not None:
                return True
        return False
    
    def _parseDialogueAct(self, act):
        slotValues = {}
        intent = None
        if act is not None:
            match = re.match("([^\(]+)\(([^\)]*)\)", act)
            if match is not None:
                intent = match.group(1)
                slotValueString = match.group(2)
                slots = slotValueString.split(',')
                slotValues = {slot.split('=')[0] : slot.split('=')[1] for slot in slots if '=' in slot}
                slotValues.update({slot : None for slot in slots if '=' not in slot})
                    
        return intent, slotValues


class FileParser(object):
    """
    """
    def __init__(self,filename, domainTag="CamRestaurants"):
        self.domain_tag = domainTag
        self.filename=filename
        self.JOINER = " <=> "
        # TODO - note that can import parser into Hubs via below 2 lines:
        parser_module = __import__("RegexSemI_"+self.domain_tag, fromlist=["RegexSemI_"+self.domain_tag]) 
        self.parser = getattr(parser_module, "RegexSemI_"+self.domain_tag)()

    def decode_file(self, DOPRINT=True):
        """
        """
        self.inputs = []
        self.results = []
        with open(self.filename,"r") as f:
            for line in f:
                parse = (line.strip('\n'), self.parser.decode(line)[0][0])
                self.results.append(parse[1]) #list order is persistent, as required here
                self.inputs.append(parse[0])
                if DOPRINT:
                    print parse[0] + self.JOINER + parse[1]

    def test_file(self, referenceFile=None):
        """
        Note this just has some **very basic checking** that the ref and parsed file match up appropriately. 
         
        A guide to using this function for developing Regex SemI parsers:
        0. create a list of example sentences for parsing
        1. get a parser working a little
        2. Dump the output of parsing the example sentences file 
        >> python RegexSemI.py _resources/EXAMPLE_INPUT_SENTENCES_FOR_DOMAIN DOMAINTAG PATH_TO_REPO_ROOT > OUTfile
        3. Fix the semantic parsers in the OUTfile so that it can be used as a reference 
        4. Improve the parser
        5. Check the improvements against the reference OUTfile
        >> python RegexSemI.py _resources/EXAMPLE_INPUT_SENTENCES_FOR_DOMAIN DOMAINTAG PATH_TO_REPO_ROOT OUTfile
        6. go back to 4, add more sentences to examples file etc etc
        """
        if referenceFile is None:
            return
        lineNum = -1
        with open(referenceFile,"r") as f:
            for line in f:
                lineNum += 1
                line = line.strip('\n')
                bits = line.split(self.JOINER)
                assert(len(bits)==2)
                userinput,reference = bits[0],bits[1]
                if userinput != self.inputs[lineNum]:
                    print "MISMATCH ERROR: " + userinput + " != "  + self.inputs[lineNum]
                elif self.results[lineNum] != reference:
                    print "INCORRECT PARSE: " + userinput
                    print '\t\t'+self.results[lineNum] + " != "  + reference
                else:
                    pass
                    #print "CORRECT: " + self.results[lineNum] + self.JOINER + reference
                


#-------------------------------------------------------------------------------------------
#  Main
#-------------------------------------------------------------------------------------------
if __name__=="__main__":
    import sys
    reload(sys)  
#     sys.setdefaultencoding('utf8')  # Sometime the encoding in the example files may cause trouble otherwise
    if len(sys.argv) < 4:
        exit("Usage: python RegexSemi.py EXAMPLE_SENTENCES_FILEPATH DOMAIN_TAG REPOSITORY_ROOT [optional: REFERENCE_FILE]")
    if len(sys.argv) == 5:
        refFileIn = sys.argv[4]
    else:
        refFileIn = None
        
    from utils import Settings
    Settings.load_root(rootIn=sys.argv[3])  #when runing with FileParser() -- since without a config, root may be unavailable.
    Settings.load_config(None)
    Settings.config.add_section("GENERAL")
    Settings.config.set("GENERAL",'domains', sys.argv[2])
    

    Ontology.init_global_ontology()
    fp = FileParser(filename=sys.argv[1], domainTag=sys.argv[2])
    fp.decode_file(DOPRINT=refFileIn is None)
    fp.test_file(referenceFile=refFileIn)


#END OF FILE
