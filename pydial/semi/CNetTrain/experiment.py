# a script to run a bunch of different types of experiments
# e.g. training with varying parameters

import sys, signal,  os, sutils, ConfigParser, json, itertools, shutil, copy

import train, decode, evaluate
from multiprocessing import Process, Lock, Queue, Pool

def log(string) :
    print string
    if experiment_log != None :
        experiment_log.write(string + "\n")
        experiment_log.flush()
    
def train_decode_evaluate(config) :
    log("Working on " + config.get("DEFAULT", "name"))
    train.train(config)
    print "Tracking..."
    decode.decode(config)
    print "Evaluating..."
    evaluate.evaluate(config)
    log("Finished working on " + config.get("DEFAULT", "name"))

if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print "Usage : "
        print "python experiment.py config/eg.cfg"
        sys.exit()
        
    config = ConfigParser.ConfigParser()
    try :
        config.read(sys.argv[1])
    except Exception as inst:
        print "Failed to parse file", inst
    experiment_type = None
    if config.has_option("experiment","type") :
        experiment_type = config.get("experiment", "type")
    experiment_name = None
    if config.has_option("experiment","name") :   
        experiment_name = config.get("experiment", "name")
        
    output_dir = None
    if experiment_type != None :
        output_dir = os.path.join("output", "experiments", experiment_name)
        print "Outputting to:"
        print "\t", output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config.set("DEFAULT", "output_dir", output_dir) 
        save_cfg = os.path.join(output_dir, "experiment_config.cfg")
        if save_cfg != sys.argv[1] :
           shutil.copy2(sys.argv[1], save_cfg)
           
    if output_dir != None :
        experiment_log = open(os.path.join(output_dir, "log.txt"), "w")
    else :
        experiment_log = None
    
    
    
    configs = []
    if experiment_type == None :
        this_config = ConfigParser.ConfigParser()
        this_config.read(sys.argv[1])
        configs = [this_config]
    elif experiment_type == "vary_train" :
        repeat = 1
        if config.has_option("experiment","repeat") :
            repeat = int(config.get("experiment","repeat"))
        # experiment where config options are changed
        vary = json.loads(config.get("experiment", "vary"))
        vary_vals =[x[2] for x in vary]
        for exp_num, vals in enumerate(itertools.product(*vary_vals)) :
            for repeat_num in range(repeat) :
                # set the vals to this in config, and generate a good name for this run
                
                # new config
                this_config = ConfigParser.ConfigParser()
                this_config.read(sys.argv[1])
                
                this_config.set("DEFAULT", "output_dir", output_dir)
                this_config.remove_section("experiment")
                
                run_name = "run_" + str(exp_num)
                if repeat > 1 :
                    run_name += "_"+str(repeat_num)
                this_config.set("DEFAULT", "name", run_name)
                log("Configuring:")
                log("\t" + this_config.get("DEFAULT", "name"))
                for i, (section, option, value_set) in enumerate(vary) :
                    log("\t Setting:")
                    log("\t\t" + section+"_"+option+" = " +str(vals[i]))
                    this_config.set(section,option,str(vals[i]))
                this_config.write(open(os.path.join(output_dir, run_name +".cfg"),"w"))
                print "putting ", this_config.get("DEFAULT","name")
                configs.append(this_config)
    else :
        print "Did not recognise experiment type", experiment_type
        
        
    # run the jobs
    num_processes = 1
    if config.has_option("experiment","num_processes") :
        num_processes = int(config.get("experiment","num_processes"))
        
    pool = Pool(processes=num_processes)
    pool.map_async(train_decode_evaluate, configs).get(9999999)
