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
DataBaseSQLite.py - loads sqlite db  
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Uses SQL style queries to access database entities. Note with sqlite3 approach, database is just a file (ie no other program
needs to be running, like would be required for mongodb for example). 

.. Note::
    Called by :mod:`utils.Settings` to load database into global variable db

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.DataBase` |.|
    import :mod:`utils.Settings` |.|

************************

'''

__author__ = "cued_dialogue_systems_group"
import sqlite3
from utils import Settings
from DataBase import DataBaseINTERFACE
from utils import ContextLogger
logger = ContextLogger.getLogger('')


# Note utility function get_dist(c1, c2): in DataBase.py if new ontologies are added and (lattitude, longitude) pairs need
# translating into area bins.

class DataBase_SQLite(DataBaseINTERFACE):
    '''SQLite3 access to entities. No explicit schema info here-- See scripts file script_txt2JSON_or_SQLITE.py which was used
    to create databases (domainTag-dbase.db in ontology/ontologies).
    -- basically: name is the primary key, all requestable slots are columns. 
    -- No other rules are added in SQL database regarding other lookups for commonly searched columns. Can use sql for this to optimise
    further if desired. 
    '''
    def __init__(self, dbfile, dstring):
        self._loaddb(dbfile)
        self.domain = dstring
        self.no_constraints_sql_query = '''select  * 
                from {}'''.format(self.domain) 
                
        self.limit = 10 # number of randomly returend entities
    
    def _loaddb(self, dbfile):
        '''Sets self.db
        '''
        try:
            self.db_connection = sqlite3.connect(dbfile)
            self.db_connection.row_factory = self._dict_factory   # for getting entities back as python dict's
            self.cursor = self.db_connection.cursor()         # we will just run 1 query - so only need a single cursor object
        except Exception as e:
            print e
            logger.error('Could not load database file: %s' % dbfile)
        return
    
    def _dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d
    
    def entity_by_features(self, constraints):
        '''Retrieves from database all entities matching the given constraints. 
       
        :param constraints: features. Dict {slot:value, ...} or List [(slot, op, value), ...] \
        (NB. the tuples in the list are actually a :class:`dact` instances)
        :returns: (list) all entities (each a dict)  matching the given features.
        '''
        
        # 1. Format constraints into sql_query 
        # NO safety checking - constraints should be a list or a dict 
        # Also no checking of values regarding none:   if const.val == [None, '**NONE**']: --> ERROR
        doRand = False
        
        
        if len(constraints):
            bits = []
            values = []
            if isinstance(constraints, list):
                for const in constraints:
                    if const.op == '=' and const.val == 'dontcare':
                        continue       # NB assume no != 'dontcare' case occurs - so not handling
                    if const.op == '!=' and const.val != 'dontcare':
                        bits.append(const.slot +'!= ?')
                    else:
                        bits.append(const.slot +'= ?  COLLATE NOCASE')
                    values.append(const.val)
            elif isinstance(constraints, dict):
                for slot,value in constraints.iteritems():
                    if value != 'dontcare':
                        bits.append(slot +'= ? COLLATE NOCASE')
                        values.append(value)
                        
            # 2. Finalise and Execute sql_query
            try:
                if len(bits):
                    sql_query = '''select  * 
                    from {} 
                    where '''.format(self.domain)
                    sql_query += ' and '.join(bits)
                    self.cursor.execute(sql_query, tuple(values))
                else:
                    sql_query =  self.no_constraints_sql_query
                    self.cursor.execute(sql_query)
                    doRand = True
            except Exception as e:
                print e     # hold to debug here
                logger.error('sql error ' + str(e))
                
                
            
        else:
            # NO CONSTRAINTS --> get all entities in database?  
            #TODO check when this occurs ... is it better to return a single, random entity? --> returning random 10
            
            # 2. Finalise and Execute sql_query
            sql_query =  self.no_constraints_sql_query
            self.cursor.execute(sql_query)
            doRand = True
        
        results = self.cursor.fetchall()        # can return directly
        
        if doRand:
            Settings.random.shuffle(results)
        return results
    
    def get_length_entity_by_features(self, constraints):
        return len(self.entity_by_features(constraints))
    
    def query_entity_property(self, name, slot, value):
        '''
        '''
        sql_query = '''select  * 
                from {} 
                where name="{}" COLLATE NOCASE'''.format(self.domain,name)
        # 2. Execute SQL query
        self.cursor.execute(sql_query)
        results = self.cursor.fetchall()        # TODO delete -- just for debug
        assert(len(results)==1)  # name should be a unique identifier
        try:
            if results[0][slot] == value:
                return True
        except:
            pass        # some entities haven't had all values for domains slots filled in - although I think this is fixed
        return False
    
    def get_all_entities(self):
        sql_query = ''' select * from {}'''.format(self.domain)
        self.cursor.execute(sql_query)
        results = self.cursor.fetchall()
        return results
    
    def get_num_unique_entities(self, cols = None):
        colStatement = '*'
        if cols is not None:
            colStatement = ','.join(cols)
        sql_query = ''' SELECT count(*) from (select distinct {} from {})'''.format(colStatement,self.domain)
        self.cursor.execute(sql_query)
        results = self.cursor.fetchall()
        return results[0]['count(*)']
    
    def _customQuery(self, query):
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        return results
    
    
    
# END OF FILE
