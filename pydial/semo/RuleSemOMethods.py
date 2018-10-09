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
RuleSemOMethods.py - Classes for all Rule based Generators
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017


.. seealso:: CUED Imports/Dependencies: 

    import :mod:`semo.SemOManager` |.|
    import :mod:`utils.Settings` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.dact` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.OntologyUtils` |.|

************************

'''

__author__ = "cued_dialogue_systems_group"

'''
    Modifications History
    ===============================
    Date        Author  Description
    ===============================
    Sep 07 2016 lmr46   Shuffle list of items in DiaAct.
    Jul 20 2016 lmr46   Supporting only the domains configured in the config-file
                        Domain in _set_NL_DOMAINS ( Note that these domains are hancoded here.)
'''

import tokenize
from collections import defaultdict
import re
import os.path
import copy
from random import shuffle

import SemOManager
from utils import Settings, DiaAct, ContextLogger, dact
from ontology import OntologyUtils
logger = ContextLogger.getLogger('')  



def parse_output(input_string):
    '''Utility function used within this file's classes. 

    :param input_string: None
    :type input_string: str
    '''
    from utils import Scanner
    output_scanner = Scanner.Scanner(input_string)
    output_scanner.next()
    words = []
    prevWasVariable = False
    while output_scanner.cur[0] != tokenize.ENDMARKER:
        if output_scanner.cur[0] == tokenize.NAME:
            words.append(output_scanner.cur[1])
            output_scanner.next()
            prevWasVariable=False
        elif output_scanner.cur[1] == '$':
            variable = '$'
            output_scanner.next()
            variable += output_scanner.cur[1]
            words.append(variable)
            output_scanner.next()
            prevWasVariable=True
        elif output_scanner.cur[1] == '%':
            function = '%'
            output_scanner.next()
            while output_scanner.cur[1] != ')':
                function += output_scanner.cur[1]
                output_scanner.next()
            function += output_scanner.cur[1]
            words.append(function)
            output_scanner.next()
            prevWasVariable=True
        else:
            if prevWasVariable:
                words.append(output_scanner.cur[1])
                output_scanner.next()
            else:
                words[-1] += output_scanner.cur[1]
                output_scanner.next()
            prevWasVariable=False
    return words


#------------------------------------------------------------------------
# RULE BASED SEMO CLASSES
#------------------------------------------------------------------------
class PassthroughSemO(SemOManager.SemO):
    '''**Does nothing** - simply pass system act directly through.
    '''
    def __init__(self):
        pass

    def generate(self, act):
        '''
        :param act: the dialogue act to be verbalized
        :type act: str
        :returns: **EXACT** act as was input
        '''
        return act

class BasicTemplateRule(object):
    '''
    The template rule corresponds to a single line in a template rules file.
    This consists of an act (including non-terminals) that the rule applies to with an output string to generate
    (again including non-terminals).
    Example::
    
        select(food=$X, food=dontcare) : "Sorry would you like $X food or you dont care";
         self.rue_items = {food: [$X, dontcare]}
    '''
    def __init__(self, scanner):
        '''
        Reads a template rule from the scanner. This should have the form 'act: string' with optional comments.
        '''
        
        self.rule_act = self.read_from_stream(scanner)
        rule_act_str = str(self.rule_act)

        if '))' in rule_act_str:
            logger.warning('Two )): ' + rule_act_str)
        if self.rule_act.act == 'badact':
            logger.error('Generated bac act rule: ' + rule_act_str)

        scanner.check_token(':', 'Expected \':\' after ' + rule_act_str)
        scanner.next()
        if scanner.cur[0] not in [tokenize.NAME, tokenize.STRING]:
            raise SyntaxError('Expected string after colon')

        # Parse output string.
        self.output = scanner.cur[1].strip('"\'').strip()
        self.output_list = parse_output(self.output)

        scanner.next()
        scanner.check_token(';', 'Expected \';\' at the end of string')
        scanner.next()

        # rule_items = {slot: [val1, val2, ...], ...}
        self.rule_items = defaultdict(list)
        for item in self.rule_act.items:
            self.rule_items[item.slot].append(item.val)

    def __str__(self):
        s = str(self.rule_act)
        s += ' : '
        s += self.output + ';'
        return s

    def read_from_stream(self, scanner):
        sin = ''
        while scanner.cur[1] != ';' and scanner.cur[0] != tokenize.ENDMARKER and scanner.cur[1] != ':':
            sin += scanner.cur[1]
            scanner.next()
        return DiaAct.DiaAct(sin)

    def generate(self, input_act):
        '''
        Generates a text from using this rule on the given input act.
        Also edits the passed variables to store the number of matched items,
        number of missing items and number of matched utterance types.
        Note that the order of the act and rule acts must be exactly the same.

        :returns: output, match_count, missing_count, type_match_count
        '''
        type_match_count = 0
        match_count = 0
        missing_count = 0
        non_term_map = {}
        if self.rule_act.act == input_act.act:
            type_match_count += 1
            match_count, missing_count, non_term_map = self.match_act(input_act)

        return self.generate_from_map(non_term_map), match_count, missing_count, type_match_count, non_term_map

    def generate_from_map(self, non_term_map):
        '''
        Does the generation by substituting values in non_term_map.

        :param non_term_map: {$X: food, ...}
        :return: list of generated words
        '''
        num_missing = 0
        word_list = copy.deepcopy(self.output_list)

        for i, word in enumerate(word_list):
            if word[0] == '$': # Variable $X
                if word not in non_term_map:
                    # logger.debug('%s not found in non_term_map %s.' % (word, str(non_term_map)))
                    num_missing += 1
                else:
                    word_list[i] = non_term_map[word]
            # %$ function in output will be transformed later.

        return word_list

    def match_act(self, act):
        '''
        This function matches the given act against the slots in rule_map
        any slot-value pairs that are matched will be placed in the non-terminal map.

        :param act: The act to match against (i.e. the act that is being transformed, with no non-terminals)
        :returns (found_count, num_missing): found_count = # of items matched, num_missing = # of missing values.
        '''
        non_term_map = {} # Any mathced non-terminals are placed here.
        rules = {}
        dollar_rules = {}
        for slot in self.rule_items:
            if slot[0] == '$':
                # Copy over rules that have an unspecified slot.
                value_list = copy.deepcopy(self.rule_items[slot])
                if len(value_list) > 1:
                    logger.error('Non-terminal %s is mapped to multiple values %s' % (slot, str(value_list)))
                dollar_rules[slot] = value_list[0]
            else:
                # Copy over rules that have a specified slot.
                rules[slot] = copy.deepcopy(self.rule_items[slot])

        logger.debug(' rules: ' + str(rules))
        logger.debug('$rules: ' + str(dollar_rules))

        found_count = 0
        # For each item in the given system action.
        rnditems=act.items
        shuffle(rnditems)
        for item in rnditems:
            found = False
            if item.slot in rules:
                if item.val in rules[item.slot]:
                    # Found this exact terminal in the rules. (e.g. food=none)
                    found = True
                    found_count += 1
                    rules[item.slot].remove(item.val)
                else:
                    # Found the rule containing the same slot but no terminal.
                    # Use the first rule which has a non-terminal.
                    val = None
                    for value in rules[item.slot]:
                        if value[0] == '$':
                            # Check if we've already assigned this non-terminal.
                            if value not in non_term_map:
                                found = True
                                val = value
                                break
                            elif non_term_map[value] == item.val:
                                # This is a non-terminal so we can re-write it if we've already got it.
                                # Then this value is the same so that also counts as found.
                                found = True
                                val = value
                                break

                    if found:
                        non_term_map[val] = item.val
                        rules[item.slot].remove(val)
                        found_count += 1

            if not found and len(dollar_rules) > 0:
                # The slot doesn't match. Just use the first dollar rule.
                for slot in dollar_rules:
                    if item.val == dollar_rules[slot]: # $X=dontcare
                        found = True
                        non_term_map[slot] = item.slot
                        del dollar_rules[slot]
                        found_count += 1
                        break

                if not found:
                    for slot in dollar_rules:
                        if dollar_rules[slot] is not None and dollar_rules[slot][0] == '$': # $X=$Y
                            found = True
                            non_term_map[slot] = item.slot
                            non_term_map[dollar_rules[slot]] = item.val
                            del dollar_rules[slot]
                            found_count += 1
                            break

        num_missing = len([val for sublist in rules.values() for val in sublist])
        return found_count, num_missing, non_term_map


class BasicTemplateFunction(object):
    '''
    A function in the generation rules that converts a group of inputs into an output string.
    The use of template functions allows for simplification of the generation file as the way
    a given group of variables is generated can be extended over multiple rules.
        
    The format of the function is::

        %functionName($param1, $param2, ...) {
            p1, p2, ... : "Generation output";}

    :param scanner: of :class:`Scanner`
    :type scanner: instance
    '''
    def __init__(self, scanner):
        scanner.check_token('%', 'Expected map variable name (with %)')
        scanner.next()
        self.function_name = '%'+scanner.cur[1]
        scanner.next()
        scanner.check_token('(', 'Expected open bracket ( after declaration of function')

        self.parameter_names = []
        while True:
            scanner.next()
            # print scanner.cur
            if scanner.cur[1] == '$':
                scanner.next()
                self.parameter_names.append(scanner.cur[1])
            elif scanner.cur[1] == ')':
                break
            elif scanner.cur[1] != ',':
                raise SyntaxError('Expected variable, comma, close bracket ) in input definition of tempate function rule')

        if len(self.parameter_names) == 0:
            raise SyntaxError('Must have some inputs in function definition: ' + self.function_name)

        scanner.next()
        scanner.check_token('{', 'Expected open brace after declaration of function ' + self.function_name)
        scanner.next()

        self.rules = []
        while scanner.cur[1] != '}':
            new_rule = BasicTemplateFunctionRule(scanner)
            if len(new_rule.inputs) != len(self.parameter_names):
                raise SyntaxError('Different numbers of parameters (%d) in rules and definition (%d) for function: %s' %
                                  (len(new_rule.inputs), len(self.parameter_names), self.function_name))
            self.rules.append(new_rule)
        scanner.next()

    def transform(self, inputs):
        '''
        :param inputs: Array of function arguments.
        :returns: None
        '''
        
        inputs = [w.replace('not available', 'none') for w in inputs]
            
        for rule in self.rules:
            if rule.is_applicable(inputs):
                return rule.transform(inputs)

        logger.error('In function %s: No rule to transform inputs %s' % (self.function_name, str(inputs)))


class BasicTemplateFunctionRule(object):
    '''
    A single line of a basic template function. This does a conversion of a group of values into a string.
    e.g. p1, p2, ... : "Generation output"

    :param scanner: of :class:`Scanner`
    :type scanner: instance
    '''
    def __init__(self, scanner):
        '''
        Loads a template function rule from the stream. The rule should have the format:
            input1, input2 : "output string";
        '''
        self.inputs = []
        self.input_map = {}
        while True:
            # print scanner.cur
            if scanner.cur[1] == '$' or scanner.cur[0] in [tokenize.NUMBER, tokenize.STRING, tokenize.NAME]:
                input = scanner.cur[1]
                if scanner.cur[1] == '$':
                    scanner.next()
                    input += scanner.cur[1]
                # Add to lookup table.
                self.input_map[input] = len(self.inputs)
                self.inputs.append(input.strip('"\''))
                scanner.next()
            elif scanner.cur[1] == ':':
                scanner.next()
                break
            elif scanner.cur[1] == ',':
                scanner.next()
            else:
                raise SyntaxError('Expected string, comma, or colon in input definition of template function rule.')

        if len(self.inputs) == 0:
            raise SyntaxError('No inputs specified for template function rule.')

        # Parse output string.
        scanner.check_token(tokenize.STRING, 'Expected string output for template function rule.')
        self.output = scanner.cur[1].strip('\"').strip()
        self.output = parse_output(self.output)

        scanner.next()
        scanner.check_token(';', 'Expected semicolon to end template function rule.')
        scanner.next()

    def __str__(self):
        return str(self.inputs) + ' : ' + str(self.output)

    def is_applicable(self, inputs):
        '''
        Checks if this function rule is applicable for the given inputs.

        :param inputs: array of words
        :returns: (bool) 
        '''
        if len(inputs) != len(self.inputs):
            return False

        for i, word in enumerate(self.inputs):
            if word[0] != '$' and inputs[i] != word:
                return False

        return True

    def transform(self, inputs):
        '''
        Transforms the given inputs into the output. All variables in the output list are looked up in the map
        and the relevant value from the inputs is chosen.

        :param inputs: array of words.
        :returns: Transformed string.
        '''
        result = []
        for output_word in self.output:
            if output_word[0] == '$':
                if output_word not in self.input_map:
                    logger.error('Could not find variable %s' % output_word)
                result.append(inputs[self.input_map[output_word]])
            else:
                result.append(output_word)
        return ' '.join(result)


class BasicTemplateGenerator(object):
    '''
    The basic template generator loads a list of template-based rules from a string.
    These are then applied on any input dialogue act and used to generate an output string.

    :param filename: the template rules file
    :type filename: str
    '''
    def __init__(self, filename):
        from utils import Scanner
        fn = Settings.locate_file(filename)
        if os.path.exists(fn):
            f = open(fn)
            string = f.read()
            string.replace('\t', ' ')
            file_without_comment = Scanner.remove_comments(string)
            scanner = Scanner.Scanner(file_without_comment)
            scanner.next()
            self.rules = []
            self.functions = []
            self.function_map = {}
            self.parse_rules(scanner)
            f.close()
        else:
            logger.error("Cannot locate template file %s",filename)

    def parse_rules(self, scanner):
        '''Check the given rules

        :param scanner: of :class:`Scanner`
        :type scanner: instance
        '''
        try:
            while scanner.cur[0] not in [tokenize.ENDMARKER]:
                if scanner.cur[0] == tokenize.NAME:
                    self.rules.append(BasicTemplateRule(scanner))
                elif scanner.cur[1] == '%':
                    ftn = BasicTemplateFunction(scanner)
                    self.functions.append(ftn)
                    self.function_map[ftn.function_name] = ftn
                else:
                    raise SyntaxError('Expected a string or function map but got ' +
                                      scanner.cur[1] + ' at this position while parsing generation rules.')

        except SyntaxError as inst:
            print inst

    def transform(self, sysAct):
        '''
        Transforms the sysAct from a semantic utterance form to a text form using the rules in the generator.
        This function will run the sysAct through all variable rules and will choose the best one according to the
        number of matched act types, matched items and missing items.

        :param sysAct: input system action (semantic form).
        :type sysAct: str
        :returns: (str) natural language 
        '''
        input_utt = DiaAct.DiaAct(sysAct)
        
        # FIXME hack to transform system acts with slot op "!=" to "=" and add slot-value pair other=true which is needed by NLG rule base
        # assumption: "!=" only appears if there are no further alternatives, ie, inform(name=none, name!=place!, ...)
        negFound = False
        for item in input_utt.items:
            if item.op == "!=":
                item.op = u"="
                negFound = True
        if negFound:
            otherTrue = dact.DactItem(u'other',u'=',u'true')
            input_utt.items.append(otherTrue)        
            
        # Iterate over BasicTemplateRule rules.
        best_rule = None
        best = None
        best_matches = 0
        best_type_match = 0
        best_missing = 1000
        best_non_term_map = None
        for rule in self.rules:
            logger.debug('Checking Rule %s' % str(rule)) 
            out, matches, missing, type_match, non_term_map = rule.generate(input_utt)
            if type_match > 0:
                logger.debug('Checking Rule %s: type_match=%d, missing=%d, matches=%d, output=%s' %
                             (str(rule), type_match, missing, matches, ' '.join(out)))

            # Pick up the best rule.
            choose_this = False
            if type_match > 0:
                if missing < best_missing:
                    choose_this = True
                elif missing == best_missing:
                    if type_match > best_type_match:
                        choose_this = True
                    elif type_match == best_type_match and matches > best_matches:
                        choose_this = True

            if choose_this:
                best_rule = rule
                best = out
                best_missing = missing
                best_type_match = type_match
                best_matches = matches
                best_non_term_map = non_term_map

                if best_type_match == 1 and best_missing == 0 and best_matches == len(input_utt.items):
                    break

        if best_rule is not None:
            if best_missing > 0:
                logger.warning('While transforming %s, there were missing items.' % sysAct)
        else:
            logger.debug('No rule used.')

        best = self.compute_ftn(best, best_non_term_map)
        return ' '.join(best)

    def compute_ftn(self, input_words, non_term_map):
        '''
        Applies this function to convert a function into a string.

        :param input_words: of generated words. Some words might contain function. `(e.g. %count_rest($X) or %$Y_str($P) ...)`
        :type input_words: list
        :param non_term_map:  
        :returns: (list) modified input_words
        '''
        for i, word in enumerate(input_words):
            if '%' not in word:
                continue
            logger.debug('Processing %s in %s...' % (word, str(input_words)))
            m = re.search('^([^\(\)]*)\((.*)\)(.*)$', word.strip())
            if m is None:
                logger.error('Parsing failed in %s' % word.strip())
            ftn_name = m.group(1)
            ftn_args = [x.strip() for x in m.group(2).split(',')]
            remaining = ''
            if len(m.groups()) > 2:
                remaining = m.group(3)

            # Processing function name.
            if '$' in ftn_name:
                tokens = ftn_name.split('_')
                if len(tokens) > 2:
                    logger.error('More than one underbar _ found in function name %s' % ftn_name)
                var = tokens[0][1:]
                if var not in non_term_map:
                    logger.error('Unable to find nonterminal %s in non terminal map.' % var)
                ftn_name = ftn_name.replace(var, non_term_map[var])

            # Processing function args.
            for j, arg in enumerate(ftn_args):
                if arg[0] == '%':
                    logger.error('% in function argument %s' % str(word))
                elif arg[0] == '$':
                    ftn_args[j] = non_term_map[arg]

            if ftn_name not in self.function_map:
                logger.error('Function name %s is not found.' % ftn_name)
            else:
                input_words[i] = self.function_map[ftn_name].transform(ftn_args) + remaining

        return input_words


class BasicSemO(SemOManager.SemO):
    '''
    Template-based output generator.  Note that the class inheriting from object is important - without this the super method
    can not be called -- This relates to 'old-style' and 'new-style' classes in python if interested ...

    :parameter [basicsemo] templatefile: The template file to use for generation.
    :parameter [basicsemo] emphasis: Generate emphasis tags.
    :parameter [basicsemo] emphasisopen: Emphasis open tag (default: &ltEMPH&lt).
    :parameter [basicsemo] emphasisclose: Emphasis close tag (default: &lt/EMPH&lt).
    '''
    def __init__(self, domainTag=None): 
        template_filename = None
        if Settings.config.has_option('semo_'+domainTag, 'templatefile'):            
            template_filename = str(Settings.config.get('semo_'+domainTag, 'templatefile'))
        self.emphasis = False
        if Settings.config.has_option('semo_'+domainTag, 'emphasis'):            
            self.emphasis = Settings.config.getboolean('semo_'+domainTag, 'emphasis')
        self.emphasis_open = '<EMPH>'
        if Settings.config.has_option('semo_'+domainTag, 'emphasisopen'):            
            self.emphasis = Settings.config.get('semo_'+domainTag, 'emphasisopen')
        self.emphasis_close = '</EMPH>'
        if Settings.config.has_option('semo_'+domainTag, 'emphasisclose'):            
            self.emphasis = Settings.config.get('semo_'+domainTag, 'emphasisclose')

        self.generator = BasicTemplateGenerator(template_filename)

    def generate(self, act):
        if self.emphasis:
            logger.warning('Emphasis is not implemented.')
        return self.generator.transform(act)

class TopicManagerBasicSemO(BasicSemO):
    '''
    The generator class for topic manager domain. This is used for handling topic manager specfic conversations.
    '''
    def __init__(self, domainTag=None):
        super(TopicManagerBasicSemO, self).__init__(domainTag)
        self._set_NL_DOMAINS()   # templates are slightly dynamic. This init's some messages.
    
    # Methods just for TopicManager:
    def _set_NL_DOMAINS(self):
        """Natural language for domain names
        """
        domains = Settings.config.get("GENERAL",'domains') # a Hub has checked this exists 
        possible_domains = domains.split(',') 
        #lmr46: Adapting only the domains available in the config file
        NL_DOMAINS = dict.fromkeys(OntologyUtils.available_domains)
        for dom in possible_domains:
            text=""
            if dom=="CamRestaurants":
                text= "Cambridge Restaurant"
            elif dom=="CamHotels":
                text='Cambridge Hotel'
            elif dom=="Laptops6":
                text= "Laptops"
            elif dom=="camtourist":
                text= "Cambridge Restaurants or hotels"
            elif dom=="SFRestaurants":
                text="San Francisco Restaurant"
            elif dom=="SFHotels":
                text="San Francisco Hotel"

            NL_DOMAINS[dom] = text
            #NL_DOMAINS["CamHotels"] = 'Cambridge Hotel'
            #NL_DOMAINS["Laptops6"] = "Laptops"
            #NL_DOMAINS["camtourist"] = "Restaurant or a hotel"

        #TODO    -- OTHER DOMAIN LABELS -- topic tracker only works for CamRestaurants and CamHotels at present - so only these here now.
        
        self.possible_domains_NL = []
        for dstring in possible_domains:
            self.possible_domains_NL.append(NL_DOMAINS[dstring])
        if len(possible_domains) > 1:
            self.possible_domains_NL[-1] = 'or a ' + self.possible_domains_NL[-1]
        return


    def _filter_prompt(self,promptIn):
        """
        """
        #Just used by the Generic Dialogue Manager at present
        DTAG = "_DOMAINS_"
        if DTAG in promptIn:
            domains = ', '.join(self.possible_domains_NL)
            return promptIn.replace(DTAG,domains)
        return promptIn
    
    def generate(self, act):
        '''Overrides the BasicSemO generate() method to do some additional things just for topic manager domain.
        '''
        nlg = super(TopicManagerBasicSemO,self).generate(act)
        return self._filter_prompt(nlg)
        


if __name__ == '__main__':
    BasicTemplateGenerator('semo/templates/CamRestaurantsMessages.txt')


#END OF FILE
