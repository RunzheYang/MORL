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
RuleTopicTrackers.py - Rule based topic trackers 
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils` |.|

************************

'''
__author__ = "cued_dialogue_systems_group"

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Jul 20 2016 lmr46   Inferring only the domains configured in the config-file
                        Note that keywords for domains are set in the dictionary here (handcoded)
                        TODO: What happen when the same keyword apply for different domains?
'''
from utils import Settings, ContextLogger
from ontology import OntologyUtils
logger = ContextLogger.getLogger('')

class TopicTrackerInterface(object):
    """Template for any Topic Tracker for the cued-python system
    
    .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
    """
    def infer_domain(self,userActHyps=None):
        pass  # Define in actual class. Must set and also return self.current_tracking_result
    
    def restart(self):
        pass  # Define in actual class. May be some notion of state etc to be reset in more advanced topic trackers


class TextBasedSwitchTopicTracker(TopicTrackerInterface):
    """When using texthub, you can enter: switch("CamRestaurants")  which will change domains to CamRestaurants for example.
    -- if switch("XX") not entered, assumes you want to stay in domain of previous turn
    """
    def __init__(self):
        self.restart()
        
    def restart(self):
        self.current_tracking_result = None
        self.FOUND_DOMAIN = False

    def infer_domain(self, userActHyps=None):
        """userActHyps : [(text, prob)]
        """
        if 'switch("' in userActHyps[0][0]:
            candidateDomain = userActHyps[0][0].split('"')[1]  # a little fragile -  
            if candidateDomain in OntologyUtils.available_domains:
                self.current_tracking_result = candidateDomain
                self.FOUND_DOMAIN = True
            else:
                logger.warning("Not a valid domain tag in your switch('X') command - remain with previous domain")
        elif not self.FOUND_DOMAIN:
            msg = '\nSWITCH TOPIC TRACKER USAGE: When using the texthub switch topic tracker '
            msg += '-You should start by saying which domain to switch to.\n' 
            msg += 'Enter exactly (where DOMAINTAG is CamRestaurants,Laptops6 etc): switch("DOMAINTAG")\n'
            msg += 'You can continue on directly by entering for example: switch("DOMAINTAG")i want a cheap one\n'
            msg += 'Alternatively, use a different topic tracker.'
            exit(msg)
        else:
            logger.info('Switch("DOMAINTAG") not detected - staying with previous domain')
        return self.current_tracking_result 


class KeywordSpottingTopicTracker(TopicTrackerInterface):
    """ Just a hacky topic tracker to develop voicehub with. 
    :: Assumptions/Notes
    -- To resolve resturants and hotels will also have to spot location
    -- Assume we will stick with last domain unless we detect one of our keywords
    """
    def __init__(self):
        self.current_tracking_result = None
        self.keywords = dict.fromkeys(OntologyUtils.available_domains, None)
        #lmr46: added some keywords or lexical units ('food')
        #consider to have a Lexicon that groups words per concepts, there are available lexica for English
        #lmr46: Adapting only the domains available in the config file
        domains = Settings.config.get("GENERAL",'domains') # a Hub has checked this exists
        possible_domains = domains.split(',')
        for dom in possible_domains:
            kwds=[]
            if dom=="CamRestaurants":
                kwds=["cambridge","restaurant",'food','eat']
            elif dom=="CamHotels":
                kwds=["cambridge","hotel", "guest house", "guesthouse"]
            elif dom=="SFRestaurants":
                kwds=["san francisco","restaurant", "food","place to eat"]
            elif dom=="SFHotels":
                kwds=["san francisco","hotel", "guest house", "guesthouse", "hostel", "motel", "place to stay"]
            elif dom=="wikipedia":
                kwds=["wiki"]

            self.keywords[dom]=kwds
        # self.keywords["CamRestaurants"] = ["cambridge","restaurant",'food']
        # self.keywords["CamHotels"] = ["cambridge","hotel", "guest house", "guesthouse"]
        # self.keywords["SFRestaurants"] = ["san francisco","restaurant", "food","book"]   # ASR cant recognise much at present -- will develop
        #             # system using CamRestaurants and CamHotels
        # self.keywords["SFHotels"] = ["san francisco","hotel", "guest house", "guesthouse", "hostel", "motel", "book"]
        # self.keywords["wikipedia"] = ["wiki"] # this could be used like "OK Google" or "Alexa"

    def restart(self):
        self.current_tracking_result = None

    def infer_domain(self,userActHyps=None):
        """
        -- Assumptions: Only working with the top hypothesis from ASR
        --              Stick to last domain if nothing spotted in this turn
        --              ORDER IS IMPORTANT -- ie it will hand off to FIRST domain a keyword is spotted in
        """
        # TODO - could require all keywords to be present - e.g to disambiguate cam hotels from SFHotels
        #lmr46: allowing overlapping keywords between domains
        #su259: making current_tracking_result a local variable. method returns none if no new domain has been identified.
        
        current_tracking_result = None
        
        overlappindomains=[]
        for dstring in self.keywords.keys():
            if self._is_a_keyword_in_sentence(self.keywords[dstring],userActHyps[0][0]):
                logger.info(dstring + " keyword found in: " + userActHyps[0][0])
                if "i(=" in userActHyps[0][0] or "inform(query=" in userActHyps[0][0]:
                    current_tracking_result = "wikipedia"  # this is just a hack so i can wiki things like hotels!
                else:
                    overlappindomains.append(dstring)
                    current_tracking_result = dstring                    
                #break  #TODO: Not handling overlapping of keywords between domains - it has to disambiguate!!!

        if len(overlappindomains) > 1:
            current_tracking_result = None

        return current_tracking_result

    def _is_a_keyword_in_sentence(self,keywords, sentence):
        """Note keywords just use the first spotted one ... this needs to be a little more sophisticated to resolve
        SF hotel versus Cambridge hotel
        """
        #TODO - will need changing if/when ASR is good enough to decode LOCATIONS - so that a match will require e.g
        # "CAMBRIDGE" + "RESTAURANT" to count for TT domain
        if keywords is not None:
            for keyword in keywords:
                if keyword in sentence.lower():
                    return True
        return False





#END OF FILE
