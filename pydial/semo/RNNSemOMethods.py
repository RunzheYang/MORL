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
RNNSemOMethods.py - Interface for all RNN based Generators
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`semo.SemOManager` |.|
    import :mod:`semo.RuleSemOMethods` |.|
    import :mod:`semo.RNNLG.generator.net` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger`

************************

'''

__author__ = "cued_dialogue_systems_group"

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Oct 13 2016 thw28   Integrate RNNSemO into the system (CamRestaurants domain)
'''
import SemOManager
from RuleSemOMethods import BasicSemO
from utils import Settings, ContextLogger
import json
logger = ContextLogger.getLogger('')

class RNNSemO(SemOManager.SemO):
    ''' 
    An interface for RNN-based output generator.

    :parameter [rnnsemo] configuration: The config file to use for initialising RNNLG
    :parameter [rnnsemo] template file: The template file for a default rule-based generator. This rule based generator is used when the RNN generator cannot generate sensible sentences.
    '''
    def __init__(self, domainTag=None):
        # load DA mapping file 
        f = file('semo/mapping/DAMapping.json')
        self.pairs = json.load(f)
        f.close()
        # load both template generator and RNN generator
        self.template = BasicSemO(domainTag)
        config_filename = None
        if Settings.config.has_option('semo_'+domainTag, 'configfile'):            
            config_filename = str(Settings.config.get('semo_'+domainTag, 'configfile'))
        # init RNN generator from config file
        from semo.RNNLG.generator.net import Model
        self.generator = Model(config_filename)
        self.generator.loadConverseParams()

    def generate(self, act):
        '''
        Generate system response based on dialogue act. Use the RNN to generate first, if it fails, fall back to rule-based generator.
        '''
        logger.warning('Emphasis is not implemented.')
        # try to generate sentence from RNNLG, if fail, fall back to template generator
        sent = ''
        try:
            sent = self.generator.generate(self.mapDA(act))
        except:
            sent = self.template.generate(act)
        return sent

    def mapDA(self,act):
        '''
        Map pydial DA format to RNNLG DA format.
        '''
        # map system DA to generator DA 
        for k,v in sorted(self.pairs.iteritems(),key=lambda x:len(x[0]),reverse=True):
            act = act.replace(k,v)
        return act



#END OF FILE
