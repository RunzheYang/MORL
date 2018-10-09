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
Scanner.py - string scanning utility 
====================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    none

************************

'''

import cStringIO
import tokenize


def remove_comments(src):
    """
    This reads tokens using tokenize.generate_tokens and recombines them
    using tokenize.untokenize, and skipping comment/docstring tokens in between
    """
    f = cStringIO.StringIO(src)
    class SkipException(Exception): pass
    processed_tokens = []
    last_token = None
    # go thru all the tokens and try to skip comments and docstrings
    for tok in tokenize.generate_tokens(f.readline):
        t_type, t_string, t_srow_scol, t_erow_ecol, t_line = tok

        try:
            if t_type == tokenize.COMMENT:
                raise SkipException()

            elif t_type == tokenize.STRING:

                if last_token is None or last_token[0] in [tokenize.INDENT]:
                    # FIXEME: this may remove valid strings too?
                    #raise SkipException()
                    pass

        except SkipException:
            pass
        else:
            processed_tokens.append(tok)

        last_token = tok

    return tokenize.untokenize(processed_tokens)


class Scanner(object):
    '''
    Class to maintain tokenized string.
    '''
    def __init__(self, string):
        src = cStringIO.StringIO(string).readline
        self.tokens = tokenize.generate_tokens(src)
        self.cur = None

    def next(self):
        while True:
            self.cur = self.tokens.next()
            if self.cur[0] not in [54, tokenize.NEWLINE] and self.cur[1] != ' ':
                break
        return self.cur

    def check_token(self, token, message):
        if type(token) == int:
            if self.cur[0] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
        else:
            if self.cur[1] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
