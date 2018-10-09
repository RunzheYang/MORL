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
DataBase.py - defines the data base interface
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. Note::
    Called by :mod:`utils.Settings` to load database into global variable db

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|

************************

'''

__author__ = "cued_dialogue_systems_group"
import math

from utils import ContextLogger
logger = ContextLogger.getLogger('')


def get_dist(c1, c2):
    '''Utility function for calculating the distance between 2 points on Earth.

    :param c1: latitude,longitude
    :type c1: 2-tuple floats
    :param c2: latitude,longitude
    :type c2: 2-tuple floats
    :returns: (float) distance
    '''
    lat1 = c1[0]
    lon1 = c1[1]
    lat2 = c2[0]
    lon2 = c2[1]
    dlat = (lat1-lat2)*math.pi/180
    dlon = (lon1-lon2)*math.pi/180
    x = math.sin(dlat/2)
    y = math.sin(dlon/2)
    a = x ** 2 + math.cos(lat1*math.pi/180) * math.cos(lat2*math.pi/180) * y ** 2
    return 2 * 6371 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class DataBaseINTERFACE(object):
    '''DATA BASE HOLDERS must implement these methods
    '''
    def _loaddb(self, dbfile):
        pass
    
    def entity_by_features(self, constraints):
        '''Retrieves from database all entities matching the given constraints. 
       
        :param constraints: features. Dict {slot:value, ...} or List [(slot, op, value), ...] \
        (NB. the tuples in the list are actually a :class:`dact` instances)
        :returns: (list) all entities (each a dict)  matching the given features.
        '''
        pass
    
    def get_length_entity_by_features(self, constraints):
        pass
    
    def query_entity_property(self, entity, slot, value):
        '''This method is needed just because we only pass entity name around at present -- would be better to pass entity id around
        and retireive name when needed - as could wuickly access properties rather than searching ... TODO 
        '''
        pass
    
    



# END OF FILE
