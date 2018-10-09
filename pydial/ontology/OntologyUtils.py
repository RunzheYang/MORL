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
OntologyUtils.py - paths and rules for ontology
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. warning:: 

    content partly hard-coded (paths, dicts, etc.)

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    

************************

'''
__author__ = "cued_dialogue_systems_group"
from utils import ContextLogger
import utils.Settings
import os
logger = ContextLogger.getLogger('')


# ont_db_pairs = {
#                     'Laptops6':{'ontology':'ontology/ontologies/Laptops6-rules.json',
#                             'database':'ontology/ontologies/Laptops6-dbase.'},
#                     'Laptops11':{'ontology':'ontology/ontologies/Laptops11-rules.json',
#                              'database':'ontology/ontologies/Laptops11-dbase.'},
#                     'TV':{'ontology':'ontology/ontologies/TV-rules.json',
#                                'database':'ontology/ontologies/TV-dbase.'},
#                     'TSBextHD':{'ontology':'ontology/ontologies/TSBextHD-rules.json',
#                                 'database':'ontology/ontologies/TSBextHD-dbase.'},
#                     'TSBplayer':{'ontology':'ontology/ontologies/TSBplayer-rules.json',
#                                  'database':'ontology/ontologies/TSBplayer-dbase.'},
#                     'SFRestaurants':{'ontology':'ontology/ontologies/SFRestaurants-rules.json',
#                            'database':'ontology/ontologies/SFRestaurants-dbase.'},
#                     'SFHotels':{'ontology':'ontology/ontologies/SFHotels-rules.json',
#                            'database':'ontology/ontologies/SFHotels-dbase.'},
#                     'CamRestaurants':{'ontology':'ontology/ontologies/CamRestaurants-rules.json',
#                           'database':'ontology/ontologies/CamRestaurants-dbase.'},
#                     'CamShops':{'ontology':'ontology/ontologies/CamShops-rules.json',
#                                'database':'ontology/ontologies/CamShops-dbase.'},
#                     'CamAttractions':{'ontology':'ontology/ontologies/CamAttractions-rules.json',
#                                  'database':'ontology/ontologies/CamAttractions-dbase.'},
#                     'CamTransport':{'ontology':'ontology/ontologies/CamTransport-rules.json',
#                                 'database':'ontology/ontologies/CamTransport-dbase.'},
#                     'CamHotels':{'ontology':'ontology/ontologies/CamHotels-rules.json',
#                                  'database':'ontology/ontologies/CamHotels-dbase.'},
#             }

ont_db_pairs = {}
available_domains = []
ALLOWED_CODOMAIN_RULES = {}
MULTIDOMAIN_GROUPS = {}
BINARY_SLOTS = {}

def initUtils():
    ont_db_pairs.clear()
    del available_domains[:]
    ALLOWED_CODOMAIN_RULES.clear()
    MULTIDOMAIN_GROUPS.clear()
    BINARY_SLOTS.clear()
    
    
    ontopath = os.path.join('ontology','ontologies')
    onlyfiles = [f for f in os.listdir(os.path.join(utils.Settings.root,ontopath)) if os.path.isfile(os.path.join(os.path.join(utils.Settings.root,ontopath), f))]
    domains = set([s.split('-')[0] for s in onlyfiles])
    for domain in domains:
        if domain not in ont_db_pairs:
            dbpath = os.path.join(ontopath,domain + '-dbase.db')
            rulespath = os.path.join(ontopath,domain + '-rules.json')
            if os.path.exists(os.path.join(utils.Settings.root,dbpath)) and os.path.exists(os.path.join(utils.Settings.root,rulespath)):
        #                 'Laptops6':{'ontology':'ontology/ontologies/Laptops6-rules.json',
        #                             'database':'ontology/ontologies/Laptops6-dbase.'},
                ont_db_pairs[domain] = {'ontology':rulespath,
                            'database':os.path.join(ontopath,domain + '-dbase.')}




    available_domains.extend(ont_db_pairs.keys())
#     available_domains = ont_db_pairs.keys()
    available_domains.append('topicmanager')   #add the topicmanager key to available domains
    available_domains.append('wikipedia')   #add the wikipedia key to available domains
    available_domains.append('ood')   #add the ood key to available domains


    # TODO - fix this
    # For Multi-domain dialog - determining which of the allowed (config specified) domains can be paired together:
    # Dont want to have a combinatorial explosion here - so make it linear and set allowed partners for each domain:
    # -- NB: these group restrictions only apply to simulate
    # TODO - just specifying GROUPS may be a simpler approach here ... 
    #-------- Hand Coded codomain rules:
    ALLOWED_CODOMAIN_RULES.update(dict.fromkeys(ont_db_pairs.keys()))
    ALLOWED_CODOMAIN_RULES["Laptops6"] = ["TV"]
    ALLOWED_CODOMAIN_RULES["Laptops11"] = ["TV"]
    ALLOWED_CODOMAIN_RULES["TV"] = [["Laptops6"], ["Laptops11"]]
    
    ALLOWED_CODOMAIN_RULES["SFRestaurants"] = ["SFHotels"]
    ALLOWED_CODOMAIN_RULES["SFHotels"] = ["SFRestaurants"]
    ALLOWED_CODOMAIN_RULES["CamRestaurants"] = ["CamHotels","CamShops", "CamAttractions","CamTransport"]
    ALLOWED_CODOMAIN_RULES["CamTransport"] = ["CamHotels","CamShops", "CamAttractions","CamRestaurants"]
    ALLOWED_CODOMAIN_RULES["CamAttractions"] = ["CamHotels","CamShops", "CamRestaurants","CamTransport"]
    ALLOWED_CODOMAIN_RULES["CamShops"] = ["CamHotels","CamRestaurants", "CamAttractions","CamTransport"]
    ALLOWED_CODOMAIN_RULES["CamHotels"] = ["CamRestaurants","CamShops", "CamAttractions","CamTransport"]
    
#     ALLOWED_CODOMAIN_RULES["Laptops6"] = ["TV","TSBextHD","TSBplayer"]
#     ALLOWED_CODOMAIN_RULES["Laptops11"] = ["TV","TSBextHD","TSBplayer"]
#     ALLOWED_CODOMAIN_RULES["TV"] = [["Laptops6","TSBextHD","TSBplayer"], ["Laptops11","TSBextHD","TSBplayer"]]
#     ALLOWED_CODOMAIN_RULES["TSBextHD"] = [["TV","Laptops11","TSBplayer"], ["Laptops6","TV","TSBplayer"]]
#     ALLOWED_CODOMAIN_RULES["TSBplayer"] = [["TV","TSBextHD","Laptops6"],["TV","TSBextHD","Laptops11"]]
#-----------------------------------------


    MULTIDOMAIN_GROUPS.update(dict.fromkeys(['camtourist','sftourist','electronics','all']))
    MULTIDOMAIN_GROUPS['camtourist'] = ["CamHotels","CamShops", "CamAttractions","CamRestaurants", "CamTransport"]
    MULTIDOMAIN_GROUPS['sftourist'] = ['SFRestaurants','SFHotels']
    MULTIDOMAIN_GROUPS['electronics'] = ["TV","Laptops11"]          # Laptops6 or Laptops11 here
#     MULTIDOMAIN_GROUPS['electronics'] = ["TV","Laptops11","TSBplayer","TSBextHD"]          # Laptops6 or Laptops11 here
    MULTIDOMAIN_GROUPS['all'] = list(available_domains)     # list copies - needed since we remove some elements 
    MULTIDOMAIN_GROUPS['all'].remove('topicmanager')        # remove special domains
    MULTIDOMAIN_GROUPS['all'].remove('wikipedia')
    MULTIDOMAIN_GROUPS['all'].remove('ood')

    # TODO - fix this
    #TODO add these for each domain  - or write something better like a tool to determine this from ontology
    # Note that ALL ONTOLOGIES should be representing binary values as 0,1  (Not true,false for example)
    # These are used by SEMI to check whether we can process a yes/no response as e.g. an implicit inform(true)
    
    BINARY_SLOTS.update(dict.fromkeys(ont_db_pairs.keys()))
    BINARY_SLOTS['CamHotels'] = ['hasparking']
    BINARY_SLOTS['SFHotels'] = ['dogsallowed','hasinternet','acceptscreditcards']
    BINARY_SLOTS['SFRestaurants'] = ['allowedforkids']
    BINARY_SLOTS['Laptops6'] = ['isforbusinesscomputing']
    BINARY_SLOTS['Laptops11'] = ['isforbusinesscomputing']
    BINARY_SLOTS['TV'] = ['usb']
    BINARY_SLOTS['CamRestaurants'] = []


#==============================================================================================================
# Methods 
#==============================================================================================================
def get_ontology_path(domainString):
    '''Required function just to handle repository root location varying if running on grid machines etc
    :rtype: object
    '''
    return os.path.join(utils.Settings.root, ont_db_pairs[domainString]['ontology'])

def get_database_path(domainString):
    '''Required function just to handle repository root location varying if running on grid machines etc
    '''
    return os.path.join(utils.Settings.root, ont_db_pairs[domainString]['database'])

def get_domains_group(domains):
    '''domains has (needs to have) been checked to be in ['camtourist','sftourist','electronics','all']:
    '''
    if domains == 'camtourist':
        return MULTIDOMAIN_GROUPS['camtourist']
    elif domains == 'sftourist':
        return MULTIDOMAIN_GROUPS['sftourist']
    elif domains == 'electronics':
        return MULTIDOMAIN_GROUPS['electronics']
    elif domains == 'all':
        return MULTIDOMAIN_GROUPS['all']
    else:
        logger.error('Invalid domain group: ' + domains) 


#END OF FILE