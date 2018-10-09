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
FlatOntologyManager.py - Domain class and Multidomain API
==========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Controls Access to the ontology files.

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''

__author__ = "cued_dialogue_systems_group"
import copy, math, json
import numpy as np

import os
import DataBaseSQLite
from ontology import OntologyUtils
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')



#------------------------------------------------------------------------------------------------------------
# ONTOLOGY FOR A SINGLE DOMAIN
#------------------------------------------------------------------------------------------------------------    
class FlatDomainOntology(object):
    '''Utilities for ontology queries 
    
         
    '''
    def __init__(self, domainString, rootIn=None):
        '''Should also be a singleton class - otherwise you can actually circumvent the point of FlatOntologyManager() being a 
            singleton, which is that if ontology is to be dynamic at all - then everything should refer to the one single source.
            
            :param domainString:  tag such as 'SFHotels' or 'CamHotels'
            :type domainString: str
            :param rootIn: path of repository - default None -
            :type rootIn: str
        '''
        # For conditional goal generation:
        self.PROB_MASS_OVER_CONDITIONALS = 0.9   # must be less than 1.0

        if rootIn is not None:
            Settings.load_root(rootIn)  # for use with semi file parser - where no config is given to set repository root by

        self.domainString = domainString
        self._set_ontology()     # sets self.ontology
        self._set_db()   # sets self.db
        self._set_domains_informable_and_requestable_slots()

    def _set_domains_informable_and_requestable_slots(self):
        '''
        '''
        self.sorted_system_requestable_slots = self.ontology["system_requestable"] 
        self.sorted_system_requestable_slots.sort()
        self.informable_slots = self.ontology["informable"].keys()
        return

    def _set_ontology(self):
        """Just loads json file -- No class for ontology representation at present. 
        """
        ontology_fname = OntologyUtils.get_ontology_path(self.domainString)
        logger.info('Loading ontology: '+ontology_fname)
        try:
            self.ontology = json.load(open(ontology_fname))
        except IOError:
            #print IOError
            logger.error("No such file or directory: "+ontology_fname+". Probably <Settings.root> is not set/set wrong by config.")
        return

    def _set_db(self):
        """Sets self.db to instance of choosen Data base accessing class. 
        
        .. note:: It is currently hardcoded to use the sqlite method. But this can be config based - data base classes share interface
        so only need to change class here, nothing else in code will need adjusting. 
        """
        db_fname = OntologyUtils.get_database_path(self.domainString)
        logger.info('Loading database: '+db_fname+'db')
        try:
            #self.db = DataBase.DataBase(db_fname+'txt')
            dbprefix = None
            if Settings.config.has_option("exec_config", "dbprefix"):
                dbprefix = Settings.config.get("exec_config", "dbprefix")
                if dbprefix.lower() == 'none':
                    dbprefix = None
            if dbprefix:
                db_fname = os.path.join(dbprefix, db_fname.split('/')[-1])
            self.db = DataBaseSQLite.DataBase_SQLite(dbfile=db_fname+'db', dstring=self.domainString)
        except IOError:
            print IOError
            logger.error("No such file or directory: "+db_fname+". Probably <Settings.root> is not set/set wrong by config.")
        return

    # methods:
    def getRandomValueForSlot(self, slot, nodontcare=False, notthese=[], conditional_values=[]):
        '''
    
        :param slot: None
        :type slot: str
        :param nodontcare: None
        :type nodontcare: bool
        :param notthese: None
        :type notthese: list
        '''
        if slot == 'type':
            #TODO - still think need to think about how "type" slot is used - rather pointless just now.
            return self.ontology['type']

        if slot not in self.ontology['informable']:
            return None

        candidate = copy.deepcopy(self.ontology['informable'][slot])
        if len(candidate) == 0:
            logger.warning("candidates for slot "+slot+" should not be empty")
        if not nodontcare:
            candidate += ['dontcare']
        candidate = list(set(candidate) - set(notthese))
        # TODO - think should end up doing something like if candidate is empty - return 'dontcare' 
        if len(candidate) == 0:
            return 'dontcare'
        
        # Conditionally sample a goal based on already generated goals in other domains  
        conditional_sample_prob = self.get_sample_prob(candidate,conditional_values)
        return Settings.random.choice(candidate, p=conditional_sample_prob)

    def get_sample_prob(self, candidate, conditional_values): 
        """Sets a prob distribution over the values in *candidate* (which could be slots, or values with a slot)
        - assigns larger probabilities to things within the *conditional_values* list

        :param candidate: of strings
        :type candidate: list
        :param conditional_values: of strings
        :type conditional_values: list
        :returns: numpy vector with prob distribution
        """
        conditional_sample_prob = None
        if len(conditional_values):
            prob_mass_per_cond = self.PROB_MASS_OVER_CONDITIONALS/float(len(conditional_values))
            conditional_sample_prob = np.zeros(len(candidate))
            for cond in conditional_values:
                conditional_sample_prob[candidate.index(cond)] += prob_mass_per_cond
            # and normalise (ie fix zero elements)
            prob_mass_per_non_cond = (1.0-self.PROB_MASS_OVER_CONDITIONALS)/\
                    float(len(conditional_sample_prob)-len(conditional_sample_prob[np.nonzero(conditional_sample_prob)]))
            conditional_sample_prob[conditional_sample_prob==0] = prob_mass_per_non_cond 
            if not np.isclose(1.0, math.fsum(conditional_sample_prob)): 
                logger.warning("Sampling prob distrib not normalised.sums to: "+str(math.fsum(conditional_sample_prob)))
                return None
        return conditional_sample_prob

    def getValidSlotsForTask(self):
        '''
        :param None:
        :returns: (list) with goal slot strings 
         
        '''
        goalslots = self.ontology['system_requestable']
        if len(goalslots) < 1:
            logger.error('Number of goal constraints == 0')
        return goalslots

    def getValidRequestSlotsForTask(self):
        '''
        :param None:
        :returns: (list) with user requestable goal slot strings 
        
        .. todo::
            should be extended to cover arbitrary domains and ontologies
        '''
        A = self.ontology['requestable']
        B = self.ontology['system_requestable']
        request_slots = list(set(A)-set(B))
        return request_slots


    def getSlotsToExpress(self, slot, value):
        '''
        :param slot:
        :param value:
        :returns: List of slot names that should be conveyed for
                 the given abstract slot-value pair.
        '''
        #
        #
        #        NOTE THAT THIS FUNCTION IS NOT IMPLEMENTED ... see below
        #
        #
        
        logger.debug('DomainUtils/FlatDomainOntology: not completely implemented')
        return [slot]
#         result = []
#         if value == '':
#             result.append(slot)
# 
#         rules = ruletable.findClassByTerm(slot)
# 
#         if not rules:
#             return result
# 
#         keytype = getKeyTypeForSlot(slot, rules[0].subclass)
#         if keytype == 'structKey':
#             argrules = ruletable.findClassBInst(slot)
#             if argrules and argrules[0].args and value != 'dontcare':
#                 result = getSlotsForSubclass(value)
# 
#         if not result:
#             result.append(slot)
# 
#         return result

    def is_valid_request(self, request_type, slot):
        # TODO
        #logger.warning('Currently not implemented: always return True.')
        return True

    def is_implied(self, slot, value):
        # TODO
        #logger.warning('Currently not implemented: always return False.')
        return False
    
    def constraintsCanBeDiscriminated(self, constraints):
        '''
        Checks if the given constraints list returns a list of values which can be 
        discriminated between - i.e. there is a question which we could ask which 
        would give differences between the values.
        '''
        real_constraints = {}
        dontcare_slots = []
        for slot, value, belief in constraints:
            if value != 'dontcare':
                real_constraints[slot] = value
            else:
                dontcare_slots.append(slot)
        
        entries = self.db.entity_by_features(constraints=real_constraints)
        
        discriminable = False
        if len(entries) < 2:
            return discriminable
        else:
            discriminating_slots = list(self.informable_slots)
            discriminating_slots.remove('name')
            if 'price' in discriminating_slots: #TODO: ic340 why is price in informable slots (SFR)?
                discriminating_slots.remove('price')
            for slot in discriminating_slots:
                if slot not in dontcare_slots:
                    values = []
                    for ent in entries:
                        values.append(ent[slot])
                    if len(set(values)) > 1:
                        discriminable = True
            return discriminable

    def get_length_entity_by_features(self, constraints): 
        return self.db.get_length_entity_by_features(constraints=constraints)
        


class FlatOntologyManager(object):
    """
    A singleton class that is used system wide (single instance created in Ontology.py-->globalOntology)
    Provides access to all available domains ontologies and databases.
    """
    instances = 0
    
    # TODO - think about Cambridge Tourist System (other multi domain systems) -- can mention this under domains -->
     
    def __init__(self):
        self._ensure_singleton()
        self.ontologyManagers = dict.fromkeys(OntologyUtils.available_domains)
        self._config_bootup()
        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia','ood']
    
    def _ensure_singleton(self):
        FlatOntologyManager.instances += 1
        if FlatOntologyManager.instances != 1:
            msg = "Should not be trying to instantiate FlatOntologyManager(). This class is to be used as a singleton."
            msg += " Only 1 global instance across system, accessed via ontology.Ontology module."
            logger.error(msg)
        return
    
    def _config_bootup(self):
        '''Boot up all domains given under [GENERAL] in config as domains = A,B,C,D
        Settings.config must have first been set.
        '''
        if not Settings.config.has_option("GENERAL","domains"):
            logger.error("You must specify the domains (a domain) under the GENERAL section of the config")
        domains = Settings.config.get("GENERAL",'domains')
        if domains in ['camtourist','sftourist','electronics','all']:
            self.possible_domains = OntologyUtils.get_domains_group(domains)
        else:
            self.possible_domains = domains.split(',') 
        # self.possible_domains is used by simulated user --> won't act outside these domains
        for dstring in self.possible_domains:
            self._checkDomainString(dstring)
            self._bootup(dstring)
    
    def _bootup(self, dstring):
        self.ontologyManagers[dstring] = self._load_domains_ontology(dstring)
    
    def _checkDomainString(self, dstring):        
        if dstring not in self.ontologyManagers:
            logger.error("Sorry, "+dstring+" is not an available domain string. See OntologyUtils.available_domains")
    
    def ensure_domain_ontology_loaded(self, domainString):
        '''
        '''
        if domainString is None or domainString in self.SPECIAL_DOMAINS:
            return
        else:
            try:
                if self.ontologyManagers[domainString] is None:
                    self._bootup(domainString)
                else:
                    return # already loaded
            except AttributeError as e:
                print e
                logger.error("Domain string {} is not valid".format(domainString))
            except KeyError as e:
                print e
                logger.error("Domain string {} is not valid".format(domainString))
            except Exception as e:
                print e # some other error
                
    def updateBinarySlots(self, dstring):
        if 'binary' in self.ontologyManagers[dstring].ontology:
            OntologyUtils.BINARY_SLOTS[dstring] = self.ontologyManagers[dstring].ontology['binary']
        else:
            OntologyUtils.BINARY_SLOTS[dstring] = []
                
            
    
    #------------------------------------------------------------------------------------------------------------
    # Wrappers for domain access to ontologies/database methods and info.  NB: No checks on valid domain strings
    #------------------------------------------------------------------------------------------------------------
    def entity_by_features(self, dstring, constraints): 
        if self.ontologyManagers[dstring] is not None:
            return self.ontologyManagers[dstring].db.entity_by_features(constraints=constraints)
        return {}
    
    def get_length_entity_by_features(self, dstring, constraints): 
        if self.ontologyManagers[dstring] is not None:
            return self.ontologyManagers[dstring].get_length_entity_by_features(constraints=constraints)
        return 0
    
    def getSlotsToExpress(self, dstring, slot, value):
        return self.ontologyManagers[dstring].getSlotsToExpress(slot=slot, value=value)
    
    def getValidSlotsForTask(self, dstring):
        return self.ontologyManagers[dstring].getValidSlotsForTask()
    
    def getRandomValueForSlot(self, dstring, slot, nodontcare=False, notthese=[], conditional_values=[]):
        '''
        Randomly select a slot value for the given slot slot.
        '''
        return self.ontologyManagers[dstring].getRandomValueForSlot(slot=slot, 
                                                                    nodontcare=nodontcare, 
                                                                    notthese=notthese, 
                                                                    conditional_values=conditional_values)
    
    def getValidRequestSlotsForTask(self, dstring):
        return self.ontologyManagers[dstring].getValidRequestSlotsForTask()
    
    def is_value_in_slot(self, dstring, value, slot):
        '''
        '''
        try:
            if value in self.ontologyManagers[dstring].ontology['informable'][slot]: 
                return True
            else:
                return False
        except:
            return False
    
    def get_sample_prob(self, dstring, candidate, conditional_values):
        return self.ontologyManagers[dstring].get_sample_prob(candidate=candidate, conditional_values=conditional_values)
    
    def is_only_user_requestable(self, dstring, slot):
        try:
            logic1 = slot in self.ontologyManagers[dstring].ontology['requestable'] 
            logic2 = slot not in self.ontologyManagers[dstring].ontology['system_requestable']
            if logic1 and logic2:
                return True
            else:
                return False
        except:
            return False
    
    def is_system_requestable(self, dstring, slot):
        try:
            if slot in self.ontologyManagers[dstring].ontology['system_requestable']:
                return True
            else:
                return False
        except:
            return False  
    
    def is_valid_request(self, dstring, request_type, slot):
        return self.ontologyManagers[dstring].is_valid_request(request_type=request_type, slot=slot)
    
    def is_implied(self, dstring, slot, value):
        return self.ontologyManagers[dstring].is_implied(slot, value)
    
    # GET LENGTHS:
    #------------------------------------------------------------------------------------
    def get_len_informable_slot(self, dstring, slot):
        return len(self.ontologyManagers[dstring].ontology['informable'][slot])
    
    def get_length_system_requestable_slots(self, dstring):
        return len(self.ontologyManagers[dstring].ontology['system_requestable'])
    
    # for things subsequently manipulated - use copy.copy() with gets 
    #------------------------------------------------------------------------------------
    def get_requestable_slots(self, dstring):
        requestable = []
        if self.ontologyManagers[dstring] is not None:
            requestable =  copy.copy(self.ontologyManagers[dstring].ontology['requestable'])
        return requestable
    
    def get_system_requestable_slots(self, dstring):
        requestable = []
        if self.ontologyManagers[dstring] is not None:
            requestable = copy.copy(self.ontologyManagers[dstring].ontology['system_requestable'])
        return requestable
    
    def get_type(self, dstring):
        return self.ontologyManagers[dstring].ontology["type"]  #can return a string - no problem 
    
    def get_informable_slot_values(self, dstring, slot):
        try:
            return copy.copy(self.ontologyManagers[dstring].ontology["informable"][slot])
        except:
            logger.error("Likely due to slot being invalid")
    
    def get_informable_slots_and_values(self, dstring):
        '''NOTE: not using copy.copy() since when used, it is only looped over, not modified
        '''
        slotsValues = {}
        if self.ontologyManagers[dstring] is not None:
            slotsValues = self.ontologyManagers[dstring].ontology["informable"] 
        return slotsValues
        
    def get_informable_slots(self, dstring):
        '''NB no copy
        '''
        informable = []
        if self.ontologyManagers[dstring] is not None:
            informable = self.ontologyManagers[dstring].informable_slots
        return informable 
        
    def get_random_slot_name(self, dstring):
        return Settings.random.choice(self.ontologyManagers[dstring].ontology['requestable'])

    def get_ontology(self, dstring):
        '''Note: not using copy.copy() -- object assumed not to change
        '''
        if self.ontologyManagers[dstring] is not None:
            return self.ontologyManagers[dstring].ontology
        return None
    
    def get_method(self, dstring):
        '''NB no copy
        '''
        method = []
        if self.ontologyManagers[dstring] is not None:
            method = self.ontologyManagers[dstring].ontology['method']
        return method
    
    def get_discourseAct(self, dstring):
        '''NB no copy
        '''
        acts = []
        if self.ontologyManagers[dstring] is not None:
            acts = self.ontologyManagers[dstring].ontology['discourseAct']
        if 'none' not in acts:
            acts.append('none')
        if 'bye' not in acts:
            acts.append('bye')
        return acts
    
    def get_sorted_system_requestable_slots(self, dstring, mode='entity'):
        '''NB no copy
        '''
        if mode not in ['entity']:  # SAFETY CHECK
            logger.warning('Mode %s is not valid ' % mode)
            mode = 'entity'
        if mode == 'entity':
            return self.ontologyManagers[dstring].sorted_system_requestable_slots
        else:
            logger.error('Mode %s is not valid' % mode)
            
    def constraintsCanBeDiscriminated(self, domainString, constraints):
        '''
        Checks if the given constraints list returns a list of values which can be 
        discriminated between - i.e. there is a question which we could ask which 
        would give differences between the values.
        '''
        if self.ontologyManagers[domainString] is not None:
            return self.ontologyManagers[domainString].constraintsCanBeDiscriminated(constraints=constraints)
        return False
    
    def _load_domains_ontology(self, domainString):
        '''
        Loads and instantiates the respective ontology object as configured in config file. The new object is added to the internal
        dictionary. 
        
        Default is FlatDomainOntology.
        
        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString.
        
        :param domainString: the domain the ontology will be loaded for.
        :type domainString: str
        :returns: None
        '''
        
        ontologyClass = None
        
        if Settings.config.has_option('ontology_' + domainString, 'handler'):
            ontologyClass = Settings.config.get('ontology_' + domainString, 'handler')
        
        if ontologyClass is None:
            return FlatDomainOntology(domainString)
        else:
            try:
                # try to view the config string as a complete module path to the class to be instantiated
                components = ontologyClass.split('.')
                packageString = '.'.join(components[:-1]) 
                classString = components[-1]
                mod = __import__(packageString, fromlist=[classString])
                klass = getattr(mod, classString)
                return klass(domainString)
            except ImportError:
                logger.error('Unknown domain ontology class "{}" for domain "{}"'.format(ontologyClass, domainString))
    

#END OF FILE
