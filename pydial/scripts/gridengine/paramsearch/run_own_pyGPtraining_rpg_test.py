#!/usr/bin/env python

import os
import sys
import subprocess
from subprocess import Popen, PIPE, STDOUT


def Execute(command):
    print(command)
    os.system(command)

def getJobNum(output):
    jobNum=output.split(" ")[2]
    if '.' in jobNum:
        jobNum=jobNum.split('.')[0]
    return jobNum

def formCommand(name, appendixname, configs, dialogues, step, path, erorrrate):
    #command = "qsub -q {} -e {}.e.log -o {}.o.log -t 1-{} -S /usr/bin/env {}_{}.py {} {} {} {}".format(queuedest, name, name, parproc, name, appendixname, dialogues, step, path, errorrate)
    command = "python {}_{}.py {} {} {} {}".format(name, appendixname, dialogues, step, path, errorrate)
    for config in configs:
        command = command+" "+config
    return command

def submitJob(command):
    print "Submitting ",command
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    val = p.stdout.read()
    #val=subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print val
    jobNum=getJobNum(val)
    return jobNum
#-----------------------------------------
# Script:
#-----------------------------------------
if len(sys.argv) < 8:
    print "usage: run_grid_pyGPtraining.py dirname appendixname steps numdialoguesperstep parallelprocesses path errorrate stepfromwhichtocontinur config1 [config2, ..]"
    exit(1)

name = sys.argv[1]
print "name is {}".format(name)
appendixname = sys.argv[2]
print "appendixname is {}".format(appendixname)
steps = int(sys.argv[3])
print "steps are {}".format(steps)
dialogues = int(sys.argv[4])
print "dialogues are {}".format(dialogues)
parproc = int(sys.argv[5])
print "parallel processes are {}".format(parproc)
path = sys.argv[6]
print "path is {}".format(path)
errorrate = int(sys.argv[7])
print "simulation error rate is: {}".format(errorrate)
beginning = int(sys.argv[8])
print "beginning is {}".format(beginning)
assert(beginning > 0)  # steps are 1,2,3,  etc

i=9
configs=[]

# WARNING: only entering 1 config for now. 
while i<len(sys.argv):
    configs.append(sys.argv[i])
    i+=1

if beginning == 1:
    command="mkdir {}_{}".format(name,appendixname)
    Execute(command)
    for config in configs:
        command="cp {} {}_{}".format(config,name,appendixname)
        Execute(command)
    command="cp grid_pyGPtraining_rpg_test.py {}_{}/{}_{}.py".format(name,appendixname,name,appendixname)
    Execute(command)

    command="cp {}/*policyFile* {}_{}/".format(name,name,appendixname)
    Execute(command)
    command="cp {}/*prm* {}_{}/".format(name,name,appendixname)
    Execute(command)
    command="cp {}/*dct* {}_{}/".format(name,name,appendixname)
    Execute(command)
else:
    print "Starting from ",beginning

command="./{}_{}".format(name,appendixname)
print command
os.chdir(command)

# SUBMIT THE ACTUAL JOBS:  ---------------------
for step in range(beginning,steps+1):
    print "Running {} step\n".format(step)
    command=formCommand(name, appendixname, configs, dialogues, step, path, errorrate)
    Execute(command)

# END OF FILE
