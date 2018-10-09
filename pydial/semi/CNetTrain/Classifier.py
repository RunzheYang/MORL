import sutils, Tuples, Features
import json, sys, cPickle, time, math, os
from collections import defaultdict
from Features import cnet as cnet_extractor
from scipy.sparse import lil_matrix
import numpy

names_to_classes = {}

class classifier(object):
    def __init__(self, config):
        # classifier type
        self.type = "svm"
        if config.has_option("classifier", "type") :
            self.type = config.get("classifier", "type")
        
        # min_examples
        self.min_examples = 10
        if config.has_option("classifier", "min_examples") :
            self.min_examples = int(config.get("classifier","min_examples"))
            
        # features
        self.features = ["cnet"]
        if config.has_option("classifier", "features") :
            self.features = json.loads(config.get("classifier", "features"))
        self.feature_extractors = []
        for feature in self.features:
            self.feature_extractors.append(
                    sutils.import_class("Features." + feature)(config)
                )
            
        self.tuples = Tuples.tuples(config)
        self.config = config
        self.cnet_extractor = cnet_extractor(config)
        
        # store data:
        self.X = {}
        self.y = {}
        self.baseXs = []
        self.baseX_pointers = {}
        self.fnames = {} 
        
    def extractFeatures(self, dw, log_input_key="batch"):
        # given a dataset walker,
        # adds examples to self.X and self.y
        total_calls = len(dw.session_list)
        self.keys = set([])
        for call_num, call in enumerate(dw) :
            print "[%i/%i] %s"%(call_num+1, total_calls, call.log["session-id"])
            for log_turn, label_turn in call:
                
                if label_turn != None:
                    uacts = label_turn['semantics']['json']
                    these_tuples = self.tuples.uactsToTuples(uacts)
                    # check there aren't any tuples we were not expecting:
                    for this_tuple in these_tuples:
                        if this_tuple not in self.tuples.all_tuples :
                            print "Warning: unexpected tuple", this_tuple
                    # convert tuples to specific tuples:
                    these_tuples = [Tuples.generic_to_specific(tup) for tup in these_tuples]
                
                # which tuples would be considered (active) for this turn?
                active_tuples = self.tuples.activeTuples(log_turn)
                
                # calculate base features that are independent of the tuple
                baseX = defaultdict(float)
                for feature_extractor in self.feature_extractors:
                    feature_name = feature_extractor.__class__.__name__
                    new_feats = feature_extractor.calculate(log_turn, log_input_key=log_input_key)
                    for key in new_feats:
                        baseX[(feature_name, key)] += new_feats[key]
                        self.keys.add((feature_name, key))
                self.baseXs.append(baseX)
                            
                for this_tuple in active_tuples:
                    if label_turn != None :
                        y = (Tuples.generic_to_specific(this_tuple) in these_tuples)
                    
                    X = defaultdict(float)
                    for feature_extractor in self.feature_extractors:
                        feature_name = feature_extractor.__class__.__name__
                        new_feats = feature_extractor.tuple_calculate(this_tuple, log_turn, log_input_key=log_input_key)
                        for key in new_feats:
                            X[(feature_name, key)] += new_feats[key]
                            self.keys.add((feature_name, key))
                            
                    if this_tuple not in self.X :
                        self.X[this_tuple] = []
                    if this_tuple not in self.y :
                        self.y[this_tuple] = []
                    if this_tuple not in self.baseX_pointers :
                        self.baseX_pointers[this_tuple] = []
                    if this_tuple not in self.fnames :
                        self.fnames[this_tuple] = []
                    
                    self.X[this_tuple].append(X)
                    if label_turn != None :
                        self.y[this_tuple].append(y)
                    
                    self.baseX_pointers[this_tuple].append(len(self.baseXs) - 1)
                    
                    self.fnames[this_tuple].append(log_turn["input"]["audio-file"])


    def extractFeatures(self, sentinfo, log_input_key="batch"):
        # given a dataset walker,
        # adds examples to self.X and self.y
        total_calls = 1
        self.keys = set([])


        # calculate base features that are independent of the tuple
        baseX = defaultdict(float)
        for feature_extractor in self.feature_extractors:
            feature_name = feature_extractor.__class__.__name__
            new_feats = feature_extractor.calculate_sent(sentinfo, log_input_key=log_input_key)
            for key in new_feats:
                baseX[(feature_name, key)] += new_feats[key]
                self.keys.add((feature_name, key))
        self.baseXs.append(baseX)

        for this_tuple in self.classifiers:
                X = defaultdict(float)
                for feature_extractor in self.feature_extractors:
                    feature_name = feature_extractor.__class__.__name__
                    new_feats = feature_extractor.tuple_calculate(this_tuple, sentinfo, log_input_key=log_input_key)
                    for key in new_feats:
                        X[(feature_name, key)] += new_feats[key]
                        self.keys.add((feature_name, key))

                if this_tuple not in self.X :
                    self.X[this_tuple] = []
                if this_tuple not in self.y :
                    self.y[this_tuple] = []
                if this_tuple not in self.baseX_pointers :
                    self.baseX_pointers[this_tuple] = []
                if this_tuple not in self.fnames :
                    self.fnames[this_tuple] = []

                self.X[this_tuple].append(X)

                self.baseX_pointers[this_tuple].append(len(self.baseXs) - 1)


    def createDictionary(self):
        self.dictionary = {}
        for i, key in enumerate(self.keys):
            self.dictionary[key] = i
        
        
    def train(self, dw, config=None):
        if config == None :
            config = self.config
        log_input_key = "batch"
        if config.has_option("train","log_input_key") :
            log_input_key = config.get("train","log_input_key")
        print "extracting features from turns"
        self.extractFeatures(dw, log_input_key=log_input_key)
        print "finished extracting features"
        print "creating feature dictionary"
        self.createDictionary()
        print "finished creating dictionary (of size",len(self.dictionary),")"
        self.classifiers = {}
        for this_tuple in self.tuples.all_tuples:
            print "training", this_tuple
            if this_tuple not in self.X :
                print "Warning: not enough examples of", this_tuple
                self.classifiers[this_tuple] = None
                continue
            baseXs = [self.baseXs[index] for index in self.baseX_pointers[this_tuple]]
            y = map(int, self.y[this_tuple])
            if sum(y) < self.min_examples :
                print "Warning:  not enough examples of", this_tuple
                self.classifiers[this_tuple] = None
                continue
            X = toSparse(baseXs, self.X[this_tuple], self.dictionary)
            
            
            # pick the right classifier class
            self.classifiers[this_tuple] = names_to_classes[self.type](self.config)
            
            self.classifiers[this_tuple].train(X, y)
                
        
        no_models = [this_tuple for this_tuple in self.classifiers if self.classifiers[this_tuple] == None]
        
        if no_models:
            print "Not able to learn about: "
            print ", ".join(map(str, no_models))
            
    def decode(self):
        # run the classifiers on self.X, return results
        results = {}
        for this_tuple in self.classifiers:
            if this_tuple not in self.X :
                print "warning: Did not collect features for ", this_tuple
                continue
            n = len(self.X[this_tuple])
            if self.classifiers[this_tuple] == None :
                results[this_tuple] = numpy.zeros((n,))
                continue
            baseXs = [self.baseXs[index] for index in self.baseX_pointers[this_tuple]]
            X = toSparse(baseXs, self.X[this_tuple], self.dictionary)
            results[this_tuple] = self.classifiers[this_tuple].predict(X)
        return results
        
    
    def decodeToFile(self, dw, output_fname, config=None):
        if config == None :
            config = self.config
        t0 = time.time()
        results = {
            "wall-time":0.0,  # add later
            "dataset": dw.datasets,
            "sessions": []
                }
        log_input_key = "batch"
        if config.has_option("decode","log_input_key") :
            log_input_key = config.get("decode","log_input_key")
        
        self.extractFeatures(dw,log_input_key=log_input_key)
        decode_results = self.decode()
        counter = defaultdict(int)
        for call_num, call in enumerate(dw):
            session = {"session-id" : call.log["session-id"], "turns":[]}
            for log_turn, _ in call:
                active_tuples = self.tuples.activeTuples(log_turn)
                tuple_distribution = {}
                for this_tuple in active_tuples:
                    index = counter[this_tuple]
                    p = decode_results[this_tuple][index]
                    tuple_distribution[Tuples.generic_to_specific(this_tuple)] = p
                    # check we are decoding the right utterance
                    assert self.fnames[this_tuple][index] == log_turn["input"]["audio-file"]
                    counter[this_tuple] += 1
                slu_hyps = self.tuples.distributionToNbest(tuple_distribution)
                session["turns"].append({
                    "slu-hyps":slu_hyps
                })
            results["sessions"].append(session)
                    
        
        results["wall-time"] =time.time() - t0
        output_file = open(output_fname, "wb")
        json.dump(results, output_file, indent=4)
        output_file.close()

    
    def decode_sent(self, sentinfo, output_fname, config=None):
        if config == None :
            config = self.config
        t0 = time.time()
        self.X = {}
        self.y = {}
        self.baseXs = []
        self.baseX_pointers = {}
        self.fnames = {}
        log_input_key = "batch"
        if config.has_option("decode","log_input_key") :
            log_input_key = config.get("decode","log_input_key")

        self.extractFeatures(sentinfo,log_input_key=log_input_key)
        decode_results = self.decode()
        counter = defaultdict(int)

        active_tuples = self.tuples.activeTuples_sent(sentinfo)
        tuple_distribution = {}
        for this_tuple in active_tuples:
            index = counter[this_tuple]
            p = decode_results[this_tuple][index]
            tuple_distribution[Tuples.generic_to_specific(this_tuple)] = p
            # check we are decoding the right utterance
            counter[this_tuple] += 1
        slu_hyps = self.tuples.distributionToNbest(tuple_distribution)

        return slu_hyps




    def save(self, save_fname):
        classifier_params = {}
        for this_tuple in self.classifiers:
            if self.classifiers[this_tuple] == None :
                classifier_params[this_tuple] = None
            else :
                classifier_params[this_tuple]  = self.classifiers[this_tuple].params()
        
        obj = {
            "classifier_params":classifier_params,
            "dictionary":self.dictionary
        }
        save_file = open(save_fname, "wb")
        cPickle.dump(obj, save_file)
        save_file.close()
        
    
    def load(self, fname):
        rootpath=os.getcwd()
        if "semi"not in rootpath:
            fname=rootpath+"/semi/CNetTrain/"+fname
        else:
            fname=rootpath+"/CNetTrain/"+fname
        print "loading saved Classifier"
        obj = cPickle.load(open(fname))
        print "loaded."
        classifier_params =  obj["classifier_params"]
        self.classifiers = {}
        for this_tuple in classifier_params:
            if classifier_params[this_tuple] == None :
                self.classifiers[this_tuple] = None
            else :
                self.classifiers[this_tuple] = names_to_classes[self.type](self.config)
                self.classifiers[this_tuple].load(classifier_params[this_tuple])
        
        self.dictionary = obj["dictionary"]
    
    def export(self, models_fname, dictionary_fname, config_fname):
        print "exporting Classifier for Caesar to read"
        print "models to be saved in", models_fname
        print "dictionary to be saved in", dictionary_fname
        print "config to be saved in", config_fname
        
        if self.type != "svm" :
            print "Only know how to export SVMs"
            return
        lines = []
        for this_tuple in self.classifiers:
            if self.classifiers[this_tuple] != None:
                t = this_tuple
                if Tuples.is_generic(this_tuple[-1]) :
                    t = this_tuple[:-1] + ("<generic_value>",)
                lines += ['('+','.join(t)+')']
                lines += utils.svm_to_libsvm(self.classifiers[this_tuple].model)
                lines += [".",""]
        models_savefile = open(models_fname, "wb")
        for line in lines:
            models_savefile.write(line+"\n")
        models_savefile.close()
        
        # save dictionary
        json_dictionary = []
        dictionary_items = self.dictionary.items()
        dictionary_items.sort(key = lambda x:x[1])
        assert [x[1] for x in dictionary_items] == range(len(self.dictionary))
        keys = [list(x[0]) for x in dictionary_items]
        
        json.dump( keys, open(dictionary_fname, "w"))
        
        
        # save config
        config_savefile = open(config_fname, "w")
        config_savefile.write("# Automatically generated by CNetTrain scripts\n")
        options = {
            "FEATURES":json.dumps(self.features),
            "MAX_ACTIVE_TUPLES":str(self.tuples.max_active),
            "TAIL_CUTOFF":str(self.tuples.tail_cutoff),
            "MODELS":os.path.join(os.getcwd(), models_fname),
            "DICTIONARY":os.path.join(os.getcwd(), dictionary_fname),
            
        }
        if "cnet" in self.features :
            index = self.features.index("cnet")
            cnf = self.feature_extractors[index]
            options["MAX_NGRAM_LENGTH"] = str(cnf.max_length)
            options["MAX_NGRAMS"] = str(cnf.max_ngrams)
        for key in options:
            this_line = "CNET   : %s"% key
            this_line = this_line.ljust(30)
            this_line += "= "+options[key]
            config_savefile.write("\t"+this_line+"\n")
        config_savefile.close()
        print "exported Classifier."
            
        
    
def toSparse(baseX, X, dictionary):
    # convert baseX & X (a list of dictionaries), to a sparse matrix, using dictionary to map to indices
    out = lil_matrix((len(X),len(dictionary)))
    for i, (basex, x) in enumerate(zip(baseX, X)) :
        for key in basex :
            if key not in dictionary :
                continue
            out[i,dictionary[key]] = basex[key]  
        for key in x :
            if key not in dictionary :
                continue
            out[i,dictionary[key]] = x[key]
            
    out = out.tocsr()
    return out

    
# classifiers define :
#  train(X,y)
#  predict(X)
#  params()
#  load(params)
#  X is a sparse matrix, y is a vector of class labels (ints)
from sklearn import svm
class SVM():
    def __init__(self, config):
        self.C = 1
        
    def pickC(self, X, y):
        Cs =      [1, 0.1, 5, 10, 50] # 1 goes first as it should be preferred
        scores = []
        n = X.shape[0]
        dev_index = max([int(n*0.8), 1+y.index(1)])
        max_score = 0.0
        self.C = Cs[0]
        print "Warning, not picking C from validation"
        return
        for i, C in enumerate(Cs) :
            this_model = svm.sparse.SVC(C=C, kernel='linear')
            this_model.probability = False
            this_model.class_weight = 'auto'
            
            this_model.fit(X[:dev_index,:],y[:dev_index])
            pred = this_model.predict(X)
            train_correct = 0.0
            dev_correct = 0.0
            for j, y_j in enumerate(y):
                if j < dev_index :
                    train_correct += int(y_j == pred[j])
                else :
                    dev_correct += int(y_j == pred[j])
            train_acc = train_correct/dev_index
            dev_acc = dev_correct/(n-dev_index)
            score = (0.1*train_acc + 0.9*dev_acc)
            print "\tfor C=%.2f;\n\t\t train_acc=%.4f, dev_acc=%.4f, score=%.4f" % (C, train_acc, dev_acc, score)
            if score > max_score :
                max_score = score
                self.C = C
            if score == 1.0 :
                break
        print "Selected C=%.2f"%self.C
    
        
    def train(self, X, y):
        self.pickC(X, y)
        #model = svm.sparse.SVC(kernel='linear', C=self.C)
        model = svm.SVC(kernel='linear', C=self.C)
        model.probability=True
        model.class_weight = 'auto'
        model.fit(X,y)
        self.model = model
        
    def predict(self, X):
        y = self.model.predict_proba(X)
        return y[:,1]
    
    def params(self, ):
        return self.model
    
    def load(self, params):
        self.model = params
        
names_to_classes["svm"] = SVM
    
from sklearn.linear_model import SGDClassifier
class SGD():
    def __init__(self, config):
        pass
    
    def train(self, X, y):
        model = SGDClassifier(loss="log", penalty="l2")
        model.probability=True
        model.fit(X,y)
        self.model = model
        
    def predict(self, X):
        y = self.model.predict_proba(X)
        return y[:,1]
    
    def params(self, ):
        return self.model
    
    def load(self, params):
        self.model = params

names_to_classes["sgd"] = SGD
   
