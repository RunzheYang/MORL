Running PyDial Test Programs
============================



- run tests from the repository root by typing:
  
  >>> testPyDial


  This will run all tests within /tests/.  The output should be similar to the following::

    PyDial> testPyDial
    Running PyDial Tests
      1 tests/test_DialogueServer.py   time 0m3.669s
      2 tests/test_Simulate.py         time 0m19.233s
      3 tests/test_Tasks.py            time 0m0.421s
    3 tests: 924 warnings,   0 errors
    See test logs in _testlogs for details



*****************
General Notes
*****************

* **These are not unit tests**; they test higher capabilities, for example, they can initiate a simulate run.
  They are meant to be just basic tests to ensure that the repository is not broken.
  They should be run before git commits and pushes for example.

* There are some odd looking python path adjustments at the start of the test_X.py files. These aren't needed in general, but the tests don't
  run on camdial for example without them. They look like this:
  
  >>> import os,sys
  >>> curdir = os.path.dirname(os.path.realpath(__file__))
  >>> curdir = curdir.split('/')
  >>> curdir = '/'.join(curdir[:-1]) +'/'
  >>> sys.path.append(curdir) 
	
***********************
Other useful tools
***********************

* The following python tools are really helpful for checking the speed and quality of your code:

* **pylint**: detects outright errors but also gives suggestions on code styles and conventions. For example:
  
  >>> pylint Simulate.py 

* **cProfile**: Full profiling of every method called while executing some python code (including imported methods from the standard library etc).
  cProfile is part of the python std lib. For example:
  
  >>> python -m cProfile Simulate.py -c _config/simulate_multiDomains_HDC.cfg 

* **profilehooks**: use for profiling individual methods. Uses decorators to do so. Requires installing:
  
  >>> pip install profilehooks
