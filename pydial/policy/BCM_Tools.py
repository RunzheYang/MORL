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
BCM_Tools.py - Script for creating slot abstraction mapping files
==========================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. Note::
        Collection of utility classes and methods

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`ontology.Ontology` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` 

**************************************************

This script is used to create a mapping from slot names to abstract slot (like slot0, slot1 etc), highest entropy to lowest. Writes mapping to JSON file  
'''

__author__ = "cued_dialogue_systems_group"
import numpy as np
import os,sys
curdir = os.path.dirname(os.path.realpath(__file__))
curdir = curdir.split('/')
curdir = '/'.join(curdir[:-1]) +'/'
sys.path.append(curdir) 
from ontology import Ontology
from utils import Settings, ContextLogger
logger = ContextLogger.getLogger('')


class CreateSlotAbstractionMappings(object):
    def __init__(self):
        pass
    
    def create(self, dstring):
        '''Creates a mapping from slot names to abstract slot (like slot0, slot1 etc). dstring should be a valid domain tag.  
        Highest entropy to lowest   
        Writes mapping to JSON file   
        '''
        import scipy.stats
        import operator
        Ontology.global_ontology.ensure_domain_ontology_loaded(dstring) 
        system_requestable =  Ontology.global_ontology.ontologyManagers[dstring].ontology['system_requestable']
        requestable = Ontology.global_ontology.ontologyManagers[dstring].ontology['requestable']
        
        # loop over all entities in database and record all values (have to do this for requestable slots at least -> doing it for both)
        ents = Ontology.global_ontology.ontologyManagers[dstring].db.get_all_entities()
        slots = {}
        for ent in ents:
            for slot,value in ent.iteritems():
                if slot not in slots:
                    slots[slot] = {}
                if value == 'not available':         # -- some SFHotels entities for example dont have all slots given. dogsallowed eg
                    continue
                if value not in slots[slot]:
                    slots[slot][value] = 0
                slots[slot][value] += 1
        
        sr_entropies = dict.fromkeys(system_requestable)
        r_entropies = dict.fromkeys(requestable)
        num_ents = len(ents)
        for slot in slots.keys():                             
            value_counts = np.asarray(slots[slot].values(), dtype=np.float)
            num_values = np.sum(value_counts)
            
            try:
                assert(num_ents==num_values)        # silly way of writing - was a check initially - now passing over
            except AssertionError:
                if num_ents > num_values:
                    print 'Values not given on all ents in domain {}'.format(dstring)
                else:
                    print 'Some entity has value given more than once in domain {}'.format(dstring) #dont think this can happen actually
                    
        
            unique_vals = float(len(value_counts))
            if slot in sr_entropies:
                sr_entropies[slot] = scipy.stats.entropy(value_counts)/ unique_vals
            elif slot in r_entropies: 
                r_entropies[slot] = scipy.stats.entropy(value_counts)/ unique_vals
            elif slot in ['id']:
                pass # know about these - fine to ignore
            else:
                raw_input('{} is not in either'.format(slot))       # id
        
        
        sorted_sr = sorted(sr_entropies.items(), key=operator.itemgetter(1), reverse=True)
        sorted_r = sorted(r_entropies.items(), key=operator.itemgetter(1), reverse=True)
        real2abstract = {}
        abstract2real = {}
        
        # first add name and type  -- remain unabstracted
        real2abstract['name'] = 'name'
        real2abstract['type'] = 'type'
        abstract2real['name'] = 'name'
        abstract2real['type'] = 'type'
        
        
        i = 0
        d = [sorted_sr,sorted_r]
        for i in range(2):
            count = 0
            for slot,_ in d[i]:      # slot,slot_ent ; there is a persistent order over the list d
                if slot == 'name':
                    continue
                if i ==0:       # system requestable slots
                    abs_slot = 'slot' + str(count).zfill(2)     # print 01, 02 etc
                else:           # system informable slots
                    if slot in system_requestable:
                        continue # only want to collect up the pure info slots
                    abs_slot = 'infoslot' + str(count).zfill(2)     
                real2abstract[slot] = abs_slot
                abstract2real[abs_slot] = slot
                count += 1
            
        bcm_mapping = {}
        bcm_mapping['real2abstract'] = real2abstract
        bcm_mapping['abstract2real'] = abstract2real
        with open('slot_abstractions/'+dstring+'.json','w+') as f:
            import json
            json.dump(bcm_mapping,f, indent=4)
        
        
        
        
    
#---------------------------------------------------------------------------------
class SlotAbstractor(object):
    def __init__(self):
        pass
    

#---------------------------------------------------------------------------------
if __name__ == '__main__':
    
    Settings.init(config_file='../config/simulate_BCM.cfg')
    Ontology.init_global_ontology()
    domain = sys.argv[1]
    c = CreateSlotAbstractionMappings()
    if domain == 'all':
        from ontology import OntologyUtils
        for dstring in OntologyUtils.MULTIDOMAIN_GROUPS['all']:
            c.create(dstring)            
    else:
        c.create(domain)        

# END OF FILE