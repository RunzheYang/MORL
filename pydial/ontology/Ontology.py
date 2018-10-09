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
Ontology.py - Provides system wide access to ontology
======================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

 
.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.FlatOntologyManager` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''
import FlatOntologyManager
import OntologyUtils

__author__ = "cued_dialogue_systems_group"

global_ontology = None
def init_global_ontology():
    '''Should be called ONCE by hubs [texthub, simulate, dialogueserver] (and Tasks when creating) --
    Then holds the ontology that is used and accessible system wide. Note that the class FlatOntologyManager(object) is a singleton.
    '''
    OntologyUtils.initUtils()
    
    global global_ontology
    global_ontology = FlatOntologyManager.FlatOntologyManager()
       

#END OF FILE
