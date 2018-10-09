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

**test_Tasks.py** - test Task generator
==========================================================================


'''
import os,sys
curdir = os.path.dirname(os.path.realpath(__file__))
curdir = curdir.split('/')
curdir = '/'.join(curdir[:-1]) +'/'
os.chdir(curdir)
sys.path.append(curdir)

from utils import Settings
from utils import ContextLogger
from ontology import Ontology
from tasks import Tasks

class TTaskCreator():
    """
    """
    def __init__(self):
        self.configname = './tests/test_configs/tasks.cfg'
        Settings.init(config_file=self.configname)
        ContextLogger.createLoggingHandlers(config=Settings.config)
        self.taskdir = Settings.config.get('tasks','savedir')
        if not os.path.isdir(self.taskdir):
            os.mkdir(self.taskdir)
        savename = Settings.config.get('tasks','savename')
        self.taskfile = self.taskdir+savename+'.json'

    def test_TasksCreator(self):
        '''test Tasks.TaskCreator()
        '''

        reload(Ontology.FlatOntologyManager)        # since this has a singleton class - may be called by other nosetests earlier
        Ontology.init_global_ontology()
        task_generator = Tasks.TaskCreator()
        task_generator._create()
        task_generator._write()


    def test_TasksReader(self):
        '''test Tasks.TaskReader()
        '''
        reload(Ontology.FlatOntologyManager)        # since this has a singleton class - may be called by other nosetests earlier
        Ontology.init_global_ontology()
        task_reader = Tasks.TaskReader(taskfile=self.taskfile)
        task_reader.get_task_by_id(task_id=10001)
        task_reader.get_random_task()

def Test():
    test = TTaskCreator()
    print "\nExecuting tests in", test.__class__.__name__
    test.test_TasksCreator()
    test.test_TasksReader()
    print "Done"

if __name__ == '__main__':
    Test()

# END OF FILE
