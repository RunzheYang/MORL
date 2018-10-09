#!/usr/bin/env python

import sys
import os
from sets import Set


#-----------------------------------------
# UTILS:
#-----------------------------------------
def Execute(command):
    print command
    os.system(command)    
    
def Execute_py(command, thisTask, step):
    print command
    scriptName = str(step)+'_'+str(thisTask)+'_'+"script.sh"
    f = open(scriptName,"w")
    f.write("#!/bin/bash\n")
    f.write("python "+command)
    f.close()
    os.system("bash "+scriptName)   

def getCommand(config,error,seed,thisTask,step,numDialogs,path):
    # removed the -l policy settings - do this in config now.
    return "{}/Simulate.py -C {} -r {} -s {} -n {} --nocolor > tra_{}_{}.log".format(path,config,str(error),\
                    str(seed),str(numDialogs),str(thisTask),str(step))
    
def seed(step, totalDialogues, totalTasks, thisTask):
    return (step-1)*totalDialogues*totalTasks + (thisTask-1)*totalDialogues + 10
    
def getName(name,task, step):
    return name+"_"+str(task)+"."+str(step)
    
def getDictParam(name,task, step):
    fullname = getName(name, task, step)
    dictionary = fullname+".dct"
    parameters = fullname+".prm"
    return [dictionary, parameters]
    
def addPrior(configname):
    # TODO - this is wrong almost certain. 
    config=open(configname, 'a+')
    for line in config:
        if "[gpsarsa_" in line:
            config.write("saveasprior = True"+"\n")
            break
    #config.write("\nMCGPTDPOLICY: SAVEASPRIOR = T\n")
    config.close()
    
def extractGlobalandLocalPolicies(line): 
    elems = line.strip().split('=')[1].lstrip().split(';');
    return elems 
    
def getGlobalandLocalPolicies(configs, term="inpolicyfile"):
    policyset=Set([])  # just use list?
    for config in configs:
        configfile=open(config, 'r')
        for line in configfile:
            if term in line:
                elems=extractGlobalandLocalPolicies(line)
                for elem in elems:
                    policyset.add(elem)
        configfile.close()
    names = list(policyset)
    if len(names) ==1:
        if names[0] == '':
            names = []
    return names

#-----------------------------------------
# SCRIPT:
#-----------------------------------------
if len(sys.argv)<6:
    print "usage: grid_pyGPtraining.py totaldialogues step pathtoexecutable errorrate config1 config2 config3..."
    exit(1)


print sys.argv
totalDialogues = int(sys.argv[1])
step = int(sys.argv[2])
path = sys.argv[3]
error = int(sys.argv[4])  # int() doesn't actually matter here
configs = []

i=5
# as in run_grid_pyGPtraining.py -- only entering a single config
while i<len(sys.argv):
    configs.append(sys.argv[i])
    i=i+1

thisTask = 1
totalTasks = 10
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    thisTask = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #totalTasks = int(os.environ['SGE_TASK_LAST'])

# Write the config file for this task and step number, working from raw config input
suffConfigs=[]
policynames = getGlobalandLocalPolicies(configs, term="outpolicyfile")

for i in range(len(configs)):
    configName = configs[i].split('/')[-1]
    suffConfig = str(thisTask)+"_"+str(step)+"_"+configName   #+configs[i]
    suffConfigs.append(suffConfig)
    outfile=open(suffConfig, 'w');
    openConfig = open(configs[i],'r')
    foundIN, foundOUT = False, False
    for line in openConfig:
        # Note: need to be careful of comments in config file. will still be read here ...
        if 'outpolicyfile' in line:
            if '#' in line:
                print "Warning - be carefull about comments in config - this isnt #inpolicyfile is it?"
            #elems=extractGlobalandLocalPolicies(line)
            elems = policynames
            policies=[]
            for elem in elems:  
                policies.append(getName(elem,thisTask, step))  # such that out has same task and step as config file
            if len(policies) > 1:
                policy=';'.join(policies)
            else:
                policy=''.join(policies)
            outfile.write('inpolicyfile = '+policy+"\n")
            foundIN = True
            continue
        else:
            # for rpg policy
            if 'learning = True' in line:
                outfile.write('learning = False\n')
            elif 'inpolicyfile' in line:
                continue
            elif 'scale' in line:
                outfile.write('scale = 1\n')
            else:
                outfile.write(line)


    if not foundIN: 
        exit("you must specify inpolicyfile - can add section in this script here to write it to config")
    outfile.close()
    openConfig.close()
  
seed=seed(step, totalDialogues, totalTasks, thisTask);

if len(suffConfigs)>1:
    for config in suffConfigs:
        command=getCommand(config,error,seed,thisTask,step,totalDialogues,path)
        Execute(command)
        seed+=totalDialogues
else:
    command=getCommand(suffConfigs[0],error,seed,thisTask,step,totalDialogues,path)
    Execute_py(command, thisTask, step)

#END OF FILE
