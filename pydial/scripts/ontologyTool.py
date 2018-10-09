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
Wizard.py - script to check or create an ontology file based on a sqlite data base
======================================================================================

Copyright CUED Dialogue Systems Group 2017

.. seealso:: CUED Imports/Dependencies: 

    none

************************

'''
import argparse, sys, os, math, json
import sqlite3
from shutil import copyfile, move


root = os.getcwd()

try:
    from utils import Settings
    from ontology import Ontology
except ImportError:
    root = os.path.dirname(os.path.realpath(__file__))
    root = root.split('/')
    root = '/'.join(root[:-1]) +'/'
    sys.path.append(root)
    from utils import Settings
    from ontology import Ontology


def checkOntology(domain):
    Settings.load_root(root)
    Settings.load_config(None)
    Settings.config.add_section("GENERAL")
    Settings.config.set("GENERAL",'domains', domain)
    
    Ontology.init_global_ontology()
    
    gOnt = Ontology.global_ontology
    for dstring in gOnt.possible_domains:
        informable = gOnt.get_informable_slots(dstring)
        
        for slot in informable:
            valuesJson = gOnt.get_informable_slot_values(dstring, slot)
            results = gOnt.ontologyManagers[dstring].db.customQuery('SELECT DISTINCT {} FROM {} ORDER BY {} ASC'.format(slot,dstring,slot))
            valuesDB = [entry[slot] for entry in results]
            # valuesDB = map(lambda a : a.lower(),valuesDB)
            valuesJson.append("not available")
            difference = list(set(valuesDB) - set(valuesJson))
            if len(difference):
                print dstring + " " + slot + " " + str(difference)
                

def _query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

                
def _collectColumns(columns, prompt, state = None):
    selection = set()
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        numLines = int(math.ceil(float(len(columns))/2.0))
#         numLinesLow = math.floor(float(len(columns))/2.0)
        
        for i in range(numLines):
#         for i in range(0,len(executable)):
            index1 = i+1
            index2 = index1 + numLines
            star1 = '*' if columns[i] in selection else ''
            star2 = '*' if len(columns) > i+numLines and columns[i+numLines] in selection else ''
                        
            if index2 <= len(columns):
                print "{:<2}{:2d}: {:<35} {:<2}{:2d}: {:<35}".format(star1,index1, columns[i], star2,index2, columns[i+numLines])
            else:
                print "{:<2}{:2d}: {:<35}".format(star1,index1, columns[i])
        print ""
        
        
#         print prompt
        choice = raw_input(prompt + " ").lower()
        try:
            choice = int(choice)
            isNumber = True
        except ValueError:
            isNumber = False
            
        if isNumber:
            if choice > 0 and choice <= len(columns):
                if columns[int(choice) - 1] in selection:
                    selection.remove(columns[int(choice) - 1])
                else:
                    selection.add(columns[int(choice) - 1])
        else:
            if _query_yes_no('Did you select all slots correctly (marked with a *)?'):
                return selection
        

def _setValuesToLower(columns, domainString, cursor):
    for column in columns:
        query = "UPDATE {} SET {} = LOWER({})".format(domainString,column,column)
        _customQuery(cursor, query)

def _dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def _customQuery(cursor, query):
    cursor.execute(query)
    results = cursor.fetchall()
    return results
                
                
def createNewOntology(dbFile, domainName, t, forceOverwrite):
    
    # first check if ontology with this name already exists in pydial
    ontologyFile = '{}-rules.json'.format(domainName)
    pathToOntologies = os.path.join(root,"ontology",'ontologies')
    
    if os.path.exists(os.path.join(pathToOntologies,ontologyFile)):
        if not forceOverwrite:
            print "Domain already exists."
            return False
        else:
            print "Domain already exists. Overwriting."
    
    try:
        db_connection = sqlite3.connect(dbFile)
        db_connection.row_factory = _dict_factory   # for getting entities back as python dict's
    except Exception as e:
        print e
        print 'Could not load database file: %s' % dbFile
        return False
    
    
    cursor = db_connection.cursor()         # we will just run 1 query - so only need a single cursor object
    
    query_getColumns = "PRAGMA table_info({});".format(domainName)
    result = _customQuery(cursor,query_getColumns)
    columns = sorted([r['name'].lower() for r in result])

    informable = None
    sys_requestable = None
    requestable = None
    state = (informable, sys_requestable,requestable) # currently not used, may be used to indicate which slots have already been selected for which type
    
    informable = _collectColumns(columns,"Please select all informable slots.",state)
    sys_requestable = _collectColumns(columns,"Please select all system requestable slots.",state)
    requestable = _collectColumns(columns,"Please select all requestable slots.",state)
    binary_slots = _collectColumns(columns,"Please select all binary (yes/no) slots.",state)
    
    
    onto = {}
    onto['discourseAct'] = [
      "ack",
      "hello",
      "none",
      "repeat",
      "silence",
      "thankyou"
   ]
    onto['method'] = [
      "none",
      "byconstraints",
      "byname",
      "finished",
      "byalternatives",
      "restart"
   ]
    onto['binary'] = list(binary_slots)
    onto['informable'] = {}
    for slot in informable:
        query_getValues = "SELECT DISTINCT {} FROM {};".format(slot, domainName)
        result = _customQuery(cursor,query_getValues)
        onto['informable'][slot] = sorted([r[slot].lower() for r in result if r[slot] != 'not available'])
    
    onto['requestable'] = list(requestable)
    onto['system_requestable'] = list(sys_requestable)
    
    if t is not None:
        onto['type'] = t

    db_connection.close()
    
    with open(ontologyFile,"w") as f:
        f.write(json.dumps(onto, indent=4, sort_keys=True))
        
    move(ontologyFile, os.path.join(pathToOntologies,ontologyFile))
    
    newDbFileName = '{}-dbase.db'.format(domainName)
    copyfile(dbFile,os.path.join(pathToOntologies,newDbFileName))

    # change values of new db file to lower
    try:
        db_connection2 = sqlite3.connect(os.path.join(pathToOntologies,newDbFileName))
        db_connection2.row_factory = _dict_factory   # for getting entities back as python dict's
    except Exception as e:
        print e
        print 'Could not load database file: %s' % dbFile
        return False
    
    cursor2 = db_connection2.cursor()   

    _setValuesToLower(columns,domainName,cursor2)

    db_connection2.commit()
    db_connection2.close()

    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyDial DB tools')
    parser.add_argument('-c', '--check', action='store_true', help='check ontology of specified domain', required=False, default=False)
    parser.add_argument('-n', '--new', action='store_true', help='create ontology of specified domain', required=False, default=False)
    parser.add_argument('-d', '--domain', help='domain name', required=False, type=str)
    parser.add_argument('-db', '--database', help='data base to be used for ontology creation', required=False, type=str)
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of domain', required=False, default=False)
#     parser.add_argument('-b', '--basic', action='store_true', help='write out only basic information', required=False)
    parser.add_argument('--type', help='set type of entity for ontology, e.g., \"hotel\"', required=False, type=str)
#     parser.add_argument('--noheading', action='store_true', help='to not print out headings for csv-style output', required=False)
    args = parser.parse_args()
    
    
    
    if not (args.check or args.new) or (args.check and args.new):
        print "Specify either -n or -c (not both)."
        parser.print_usage()
        exit()
    
    if args.check:
        if args.domain is None:
            parser.print_usage()
            exit()
            
        checkOntology(args.domain)
    
    if args.new:
        if args.domain is None or args.database is None:
            parser.print_usage()
            exit()
            
        database = os.path.join(os.getcwd(),args.database)
            
        if not os.path.exists(database):
            print "{} is an invalid file. Please provide a valid database file.".format(database)
            exit()
        
        if createNewOntology(database, args.domain,args.type,args.force):
            checkOntology(args.domain)
                
    