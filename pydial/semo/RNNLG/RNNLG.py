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
RNNLG.py - Main script to train and test RNN generator.
===========================================================

**Basic Execution**:
    >>> python RNNLG -config CONFIG -mode [train|test]

Copyright CUED Dialogue Systems Group 2015 - 2017
 
.. seealso:: CUED Imports/Dependencies: 
  
    import :mod:`semo.RNNLG.utils.commandparser` |.|
    import :mod:`semo.RNNLG.generator.ngram` |.|
    import :mod:`semo.RNNLG.generator.net` |.|
    import :mod:`semo.RNNLG.generator.knn` |.|
************************
'''

import sys

sys.path.insert(0, '.')
from semo.RNNLG.utils.commandparser    import RNNLGOptParser
from semo.RNNLG.generator.net          import Model
from semo.RNNLG.generator.ngram        import Ngram
from semo.RNNLG.generator.knn          import KNN

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

if __name__ == '__main__':
    
    args = RNNLGOptParser()
    config = args.config
    
    if args.mode=='knn':
        # knn
        knn = KNN(config,args)
        knn.testKNN()
    elif args.mode=='ngram':
        # ngram case
        ngram = Ngram(config,args)
        ngram.testNgram()
    else: 
        # NN case        
        model = Model(config,args)
        if args.mode=='train' or args.mode=='adapt':
            model.trainNet()
        elif args.mode=='test':
            model.testNet()
        elif args.mode=='interactive':
            model.loadConverseParams()
            while True:
                da  = raw_input('DAct : ')
                print model.generate(da)
        
