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
Texthub.py - text level dialog system.
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

**Basic Execution**:
    >>> python texthub.py [-h] -C CONFIG [-l -r]

Optional arguments/flags [default values]::

    -r semantic error rate [0]
    -l set the system to use the given policy file
    -h help


**Relevant config variables**  (values are defaults)::

    [semi_DOMAIN]
    semitype = PassthroughSemI
    
    [semo_DOMAIN]
    semotype = PassthroughSemO


.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`utils.Settings` |.|
    import :class:`Agent.DialogueAgent` |.|
    import :mod:`ontology.Ontology`


************************

'''
import argparse, re

from Agent import DialogueAgent
from utils import ContextLogger
from utils import Settings
from ontology import Ontology
logger = ContextLogger.getLogger('')

__author__ = "cued_dialogue_systems_group"
__version__ = Settings.__version__

class ConsoleHub(object):
    '''
    text based dialog system
    '''
    def __init__(self):
 
        # Dialogue Agent:
        #-----------------------------------------
        self.agent = DialogueAgent(hub_id='texthub')

    def run(self):
        '''
        Runs one episode through Hub

        :returns: None
        '''

        logger.warning("NOTE: texthub is not using any error simulation at present.")
        sys_act = self.agent.start_call(session_id='texthub_dialog')
        print 'Prompt > ' + sys_act.prompt
        while not self.agent.ENDING_DIALOG:
            # USER ACT:
            obs = raw_input('User   > ')
        

            '''
                # Confused user act.
                # lastHyps = self.errorSimulator.confuseAct(lastUserAct)
                # print 'lastHyps >', lastHyps
                # nullProb = 0.0
                # for (act, prob) in lastHyps:
                #     if act == 'null()':
                #         nullProb += prob
                #     print 'Semi >', act, '['+str(prob)+']'
        
                # if self.forceNullPositive and nullProb < 0.001:
                #     lastHyps.append(('null()',0.001))
                #     print 'Semi > null() [0.001]'
                #--------------------------------
            '''
            domain = None
            if "domain(" in obs:
                match = re.search("(.*)(domain\()([^\)]+)(\))(.*)",obs)
                if match is not None:
                    domain = match.group(3)
                    obs = match.group(1) + match.group(5)
                
            # SYSTEM ACT:
            sys_act = self.agent.continue_call(asr_info = [(obs,1.0)], domainString = domain)
            print 'Prompt > ' + sys_act.prompt

        # Process ends. -----------------------------------------------------
        
        # NB: Can add evaluation things here - possibly useful to check things by hand with texthub ... 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TextHub')
    parser.add_argument('-C','-c', '--config', help='set config file', required=True, type=argparse.FileType('r'))
    parser.add_argument('-s', '--seed', help='set random seed', type=int)
    parser.set_defaults(use_color=True)
    parser.add_argument('--nocolor', dest='use_color',action='store_false', help='no color in logging. best to\
                        turn off if dumping to file. Will be overriden by [logging] config setting of "usecolor=".')
    args = parser.parse_args()

    seed = Settings.init(config_file=args.config.name,seed=args.seed)
    ContextLogger.createLoggingHandlers(config=Settings.config, use_color=args.use_color)
    logger.info("Random Seed is {}".format(seed))
    Ontology.init_global_ontology()
    
    hub = ConsoleHub()
    hub.run()


#END OF FILE
