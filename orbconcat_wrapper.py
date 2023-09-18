#!/usr/bin/env python

# AUTHOR:  Katie Fasbender
#          katiefasbender@montana.edu

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
#from argparse import ArgumentParser
from astropy.table import Table,Column
from astropy.io import fits
from dlnpyutils import utils as dln, coords
import logging
import numpy as np
import os
import socket
import subprocess
import sys
import time

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def makedir(dir):
    '''makes a directory with name "dir" if it does not exist
    Arguments:
    ----------
    dir (str)
            directory name
    Returns:
    --------
        None; directory "dir" is created if not already there
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)


def write_jscript(job_name,partition,cmd,outdir):
    '''writes a SLURM job script to "job_name.sh"
    Arguments:
    ----------
    job_name (str)
            name of job, job script file
    partition (str)
            node/partition the job will run on
    cmd (str)
            python command to run exposure
    outdir (str)
            base directory
    Returns:
    --------
    job_file (str)
            job filename the job script is written to
    '''
    job_file = outdir+job_name+".sh"
    # The following code writes lines to the "job_name.sh" file.
    # Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
    # All other lines are read by the shell
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        if partition=="priority": fh.writelines("#SBATCH --account=priority-davidnidever\n")       #specify the account to use
        fh.writelines("#SBATCH --job-name="+job_name+"\n")        # job name
        fh.writelines("#SBATCH --output="+outdir+job_name+".out\n")      # output file (%j = jobid)
        fh.writelines("#SBATCH --error="+outdir+job_name+".err\n")       # error file
        fh.writelines("#SBATCH --partition="+partition+"\n")     # queue partition to run the job in
        fh.writelines("#SBATCH --ntasks=1\n")                    # for running in parallel
        fh.writelines("#SBATCH --nodes=1\n")                     # number of nodes to allocate
        fh.writelines("#SBATCH --ntasks-per-node 1\n")           # number of cores to allocate; set with care
        fh.writelines("#SBATCH --mem=6000\n")                    # memory, set --mem with care!!!!! refer to hyalite quickstart guide
        fh.writelines("#SBATCH --time=6:00:00\n")               # Maximum job run time
        fh.writelines("module load Anaconda3\n")         # load anaconda, needed for running python on Hyalite!
        fh.writelines("source activate $HOME/condaenv/\n")
        fh.writelines(cmd+"\n")                                       # write python command to analyze exposure
        fh.writelines("conda deactivate")
    return job_file




#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------

if __name__ == "__main__":

    # Setup
    #------
    # Initiate input arguments
    #parser = ArgumentParser(description='Add orbital elements to CANFind tracklet catalog.')
    #parser.add_argument('--partition',type=str,nargs=1,help='Delimited list of partitio')
    #args = parser.parse_args()

    # Start time, get hostname (should be tempest)
    t0 = time.time()

    # Inputs
    #partitions=args.partition[0].split(',')
    partition = 'priority'

    # Establish necessary directories - figure out for tempest
    basedir = "/home/x25h971/orbits_dr2/"
    localdir = basedir+"files/"
    outdir = basedir+"comp0/"
    outfiledir = basedir+"outfiles/"
    makedir(outfiledir)

    # Create list of fgroup subdirectories in comp0
    fgroup_list = next(os.walk("/home/x25h971/orbits_dr2/comp0/"))[1]
    nfgroup = len(fgroup_list)

    # Prepare fgroup job info
    #------------------------
    dtype_expstr = np.dtype([('fgroup',str,100),('done',bool),('cmd',str,1000),('submitted',bool),
                             ('jobname',str,100),('jobid',str,100),('jobstatus',str,100)])
    expstr = np.zeros(nfgroup,dtype=dtype_expstr)  # string array for job info
    expstr['jobid']="-99.99"
    expstr['fgroup'] = Column([i.split("_")[-1] for i in fgroup_list])
    print("running orbit_concat.py on "+str(nfgroup)+" fgroups")
    # prepare runfile to store job structure
    runfile = localdir+'orb_concat'+str(t0)+'_run.fits'
    Table(expstr).write(runfile)
    sleep_time=10     # seconds to sleep between checking on job stream
    njobs = nfgroup

    # Start submitting jobs
    #----------------------
    jb = 0
    eflag = 0
    while eflag==0: # eflag=1 when jb==njobs
        print("Checking status of last job submitted")
        submitted_ind = list(np.where(expstr['submitted']==1)[0])
        unsubmitted_ind = list(np.where(expstr['submitted']==0)[0])

        # get index & status of last job submitted
        last_sub = submitted_ind
        if len(last_sub)!=0: lsub = np.sort(last_sub)[-1]
        else: lsub = 0
        last_jid = expstr[lsub]['jobid']
        last_jname = expstr[lsub]['jobname']
        if last_jid != "-99.99": lj_status = (subprocess.getoutput("sacct -n -X --format state --jobs="+last_jid).split("\n")[-1]).strip()
        else: lj_status = "NONE" #no jobs have been submitted
        expstr[lsub]['jobstatus'] = lj_status
        print("lj_status = ",lj_status,", jobname = ",last_jname,last_jid)

        # ---If last job is still running: wait!
        if (lj_status=="RUNNING" or lj_status=="PENDING" or lj_status=="REQUEUED"):
            print("Job id="+last_jid+" is still "+lj_status+", sleepin for a sec")
            time.sleep(sleep_time)

        # ---If last job is completed, failed, cancelled, killed, or none: submit a new job!
        else:
            print("--Submitting new job--")
            # if last job was completed, get some info about it
            if lj_status=="COMPLETED": expstr[lsub]['done']==1

            # get index & info of next job to submit
            next_sub = unsubmitted_ind
            if len(next_sub)!=0: jbsub = np.sort(next_sub)[0]
            else: eflag = 1
            # create and submit the job!
            cmd  = 'python '+localdir+'orbit_concat.py '+str(expstr['fgroup'][jbsub])+' 2 0'
            expstr['cmd'][jbsub] = cmd
            # --Write job script to file--
            job_name = 'orb_concat_'+str(t0)+'_'+str(jb)
            job_file=write_jscript(job_name,partition,cmd,outfiledir)
            # --Submit job to slurm queue--
            os.system("sbatch %s" %job_file)
            expstr['submitted'][jbsub] = True
            print("Job "+job_name+"  submitted, sleeping for a sec")
            time.sleep(sleep_time) #let the job get submitted
            # get jobid of submitted job, update array of exposure info
            jid = subprocess.getoutput("sacct -n -X --format jobid --name "+job_name)
            jid = jid.split("\n")[-1]
            expstr['jobname'][jbsub] = job_name
            expstr['jobid'][jbsub] = jid
            jb+=1

        # save job structure, sleep before checking/submitting again
        if os.path.exists(runfile): os.remove(runfile)
        Table(expstr).write(runfile,overwrite=True)
        if jb==njobs: eflag = 1
