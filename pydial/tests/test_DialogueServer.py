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

************************


**test_DialogueServer.py** - test DialogueServer()
================================================================================


Use *utils/dummyDialogueServerClient* to create multiprocess instances of a fake
client which communicate concurrently with a running dialogue server in a separate process.

'''
import os,sys
curdir = os.path.dirname(os.path.realpath(__file__))
curdir = curdir.split('/')
curdir = '/'.join(curdir[:-1]) +'/'
os.chdir(curdir)
sys.path.append(curdir)

#from nose.tools import with_setup
from ontology import Ontology
from utils import Settings, dummyDialogueServerClient, ContextLogger
import multiprocessing as mp
import DialogueServer
import time

class TDialogueServer():
    """
    """
    def __init__(self):
        cfg = 'tests/test_configs/dialogueserver.cfg'
        assert(os.path.exists(cfg))
        Settings.init(config_file=cfg)
        ContextLogger.createLoggingHandlers(config=Settings.config)

    def ds(self):
        reload(Ontology.FlatOntologyManager)
        Ontology.init_global_ontology()
        dial_server = DialogueServer.DialogueServer()
        dial_server.run()

    def test_dialogueserver(self):
        '''Create a DialogueServer and a few dummy clients
        '''
        p = mp.Process(target=self.ds)
        p.start()
        dummyDialogueServerClient.run_fake_clients(NUM_CLIENTS=3,pause_time=0,DIALOGS_PER_CLIENT=1)
        p.terminate()


def Test():
    test = TDialogueServer()
    print "\nExecuting tests in",test.__class__.__name__
    test.test_dialogueserver()
    print "Done"

if __name__ == '__main__':
    Test()



#END OF FILE

