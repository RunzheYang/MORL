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
WikipediaTools.py - basic tools to access wikipedia
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017
 
.. seealso:: CUED Imports/Dependencies:
    
    import :mod:`policy.Policy` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************
"""

__author__ = "cued_dialogue_systems_group"
import wikipedia # pip install wikipedia
from utils import Settings
from utils import ContextLogger
import Policy
logger = ContextLogger.getLogger('wiki')

import sys
reload(sys)
# sys.setdefaultencoding("utf8")


class WikipediaDM(Policy.Policy):
    """
    Dialogue Manager interface to Wikipedia -- developement state.
    """
    def __init__(self):
        super(WikipediaDM,self).__init__("wikipedia")
        
        self.startwithhello = False
        
        self.wiki = Wikipedia_API() 
        self.wm = WikiAPI_Messages()


    def nextAction(self, beliefstate, hyps):
        """
        In case system takes first turn - Topic manager will just greet the user 
        """
        user_act = hyps[0][0]
        if "inform(query=" in user_act or 'i(=' in user_act:  # last one is due to lazy typing
            query = user_act.split("=")[1][0:-1]   # assume passthroughsemi for now and inform(query=elliot smith) 
            result = self.wiki.summary(query)
            if self.wm.msgs[self.wiki.status]:
                return unicode('inform(query="'+str(result)+'")')   
            elif self.wiki.status == "DISAMBIGUATE":
                a = result[0].replace('(','').replace(')','').rstrip()
                b = result[1].replace('(','').replace(')','').rstrip()
                return unicode('select("name='+str(a)+',name='+str(b)+'")') #can only choose between top 2
            else:
                return unicode('inform(failed)')
        elif "bye(" in user_act:
            return 'bye()'
        else:
            return 'hello()'

class WikiAPI_Messages:
    """
    """
    def __init__(self):
        self.msgs = {
        'SUCCESS' : True,
        'NOPAGE' : False,
        'DISAMBIGUATE' : False,
        'INVALIDINPUT' : False,
        'OTHERERROR' : False,
        }

class Wikipedia_API():
    """
    """
    def __init__(self):
        self.page = None
        self.status = None

    def summary(self, query=None, NUMSENTENCES=1):
        """
        """
        self.page = None
        if query is None:
            query = raw_input("Query what: ")

        try:
            self.pagesummary =  wikipedia.summary(query, sentences=NUMSENTENCES)
            # this is a little round about. summary may throw an error - else we get a single pageid 
            page = wikipedia.page(query) 
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning("wiki: DISAMBIGUATE ERROR")
            self.status = "DISAMBIGUATE"
            return e.options 
        except wikipedia.exceptions.WikipediaException as e:
            if "Page id" in str(e)[0:len("Page id")]: #e should not have a length problem with this... bit fragile though
                logger.warning("wiki: NOPAGE " + str(e))
                self.status = "NOPAGE"
            elif "An unknown error occured:" in str(e): # phrase is unique enough that dont need to check certain area of e
                logger.warning("wiki: INVALIDINPUT " + str(e))
                self.status = "INVALIDINPUT"
            else:
                logger.warning("WikipediaException: " + str(e))
                self.status = "OTHERERROR"
        else:
            return self.get_page_details(pageID=page.pageid)

    def get_page_details(self, pageID=None):
        try:
            self.page = wikipedia.WikipediaPage(pageid=pageID)
        except wikipedia.exceptions.PageError as e:
            logger.warning("wiki: NOPAGE " + str(e))
            self.status = "NOPAGE"
        self.pagetitle = str(self.page.title)
        self.pagesummary = str(self.pagesummary)
        self.pagesections = self.page.sections
        self.where()
        self.status = "SUCCESS"
        return self.pagesummary

    def where(self):
        """
        """
        if self.page is not None:
            try:
                result = self.page.coordinates
            except KeyError:
                logger.warning("No Coordinates associated with page: "+self.page.title)
                self.lat,self.lon = None,None
                result = None
            if result is not None:
                self.lat,self.lon = result
        return
                
    def randomPages(self, NUMPAGES=1):
        result = wikipedia.random(pages=NUMPAGES)
        if NUMPAGES > 1:
            result = result[0]  # TODO - only deal with single page for now
        return self.summary(result)
         
    def get_section(self, sectionNum=None, sectionName=None):
        if self.page is not None:
            if not len(self.pagesections):
                logger.warning("Page -"+self.pagetitle+"- has no sections -- according to wikipedia module")
                return self._get_page_sections(sectionNum, sectionName)
            elif sectionNum is not None:
                pass 
            elif sectionName is not None:
                self.pagesection = self.page.section(sectionName)
            else:
                logger.warning("No section info requested")

    def _get_page_sections(self, sectionNum=None, sectionName=None):
        """Shouldn't have to do this - but wikipedia.sections doesnt work ... 
        """
        self.section = {}
        self.sections = []  # list maintains order
        content = self.page.content
        lines = content.split("\n")
        currentSection = None
        for line in lines:
            if "==" in line:
                line = line.replace("Edit =","")
                line = line.replace("=","").lstrip().rstrip()
                self.section[line] = []
                currentSection = line
                self.sections.append(currentSection)
            elif currentSection is not None:
                line = line.lstrip().rstrip()
                self.section[currentSection].append(line)
            else:
                pass
        logger.info("Sections in page: "+str(self.sections))
        # and return some section:
        if sectionNum is not None:
            if sectionNum > len(self.sections) or sectionNum < 0:
                sectionNum = 0
            return self.section[self.sections[sectionNum]]
        elif sectionName is not None:
            pass 


#----------------------------------------------------
if __name__=="__main__":
    Settings.load_config(config_file="./config/texthub.cfg")
    ContextLogger.createLoggingHandlers(Settings.config)

    w = Wikipedia_API() 
    print w.summary(query=None)
    #print w.page.content
    print w.where()
    print w.get_section(sectionNum=2)   #sectionName="Phases")
    #print w.randomPages(2)


#END OF FILE
