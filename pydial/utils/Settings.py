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
Settings.py - global variables: config, random num generator 
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Creates and makes accessible system wide a config parser and random number generator.
Also contains hardcodings of the paths to the ontologies.

[GENERAL]
    root = ''    - which is the path to root directory of python system. Use when running grid jobs.

Globals::
  
    config:       python ConfigParser.ConfigParser() object
    random:       numpy.random.RandomState() random number generator
    root:         location of root directory of repository - for running on the grid

.. seealso:: CUED Imports/Dependencies: 

    none

************************

'''

__author__ = "cued_dialogue_systems_group"
__version__ = '1.0'         # version setting for all modules in repository root only.

import ConfigParser
import numpy.random as nprandom
import os.path


#==============================================================================================================
# SYSTEM WIDE RESOURCES - includes globals, the paths to ontologies, and list of domain tags
#==============================================================================================================
config = None
random = None
randomCount = 0
root = ''
global_currentturn = None

#==============================================================================================================
# Methods (for settings globals)
#==============================================================================================================
def init(config_file, seed = None):
    '''
    Called by dialog programs (simulate, texthub, dialogueserver) to init Settings globals
    '''
    # Config:
    #-----------------------------------------
    load_config(config_file)
    
    # Repository root:
    #-----------------------------------------
    load_root()
    
    # Seed:
    #-----------------------------------------
    if seed is None:
        # no seed given at cmd line (the overriding input), so check config for a seed, else use None (which means use clock).    
        if config.has_option("GENERAL",'seed'):
            seed = config.getint("GENERAL","seed")
    seed = set_seed(seed)

    return seed

def load_config(config_file):
    '''
    Loads the passed config file into a python ConfigParser().
       
    :param config_file: path to config
    :type config_file: str
    '''
    global config
    config = None
    if config_file is not None:
        try:
            config = ConfigParser.ConfigParser()
            config.read(config_file)
        except Exception as inst:
            print 'Failed to parse file', inst
    else:
        # load empty config
        config = ConfigParser.ConfigParser()

def load_root(rootIn=None):
    '''
    Root is the location (ie full path) of the cued-python repository. This is used when running things on the grid (non local
    machines in general).
    '''
    global root
    root = ''
    
    if config is not None:     
        if config.has_option("GENERAL",'root'):
            root = config.get("GENERAL",'root')
    if rootIn is not None:  # just used when called by SemI parser without a config file
        root = rootIn

def set_seed(seed):
    '''
    Intialise np random num generator

    :param seed: None
    :type seed: int
    '''
    global random
    if seed is None:
        random1 = nprandom.RandomState(None)
        seed = random1.randint(1000000000)
    random = nprandom.RandomState(seed)

    return seed
    
def randomProfiling(t = None):
    global randomCount
    randomCount += 1
    print "Random access: {} from class {}".format(randomCount,t)
    return random

def locate_file(filename):
    '''
    Locate file either as given or relative to root

    :param filename: file to check
    :return: filename possibly prepended with root
    '''
    if os.path.exists(filename):
        return filename         # file exists as given
    else:
        return root+filename    # note this may or may not exist

# END OF FILE