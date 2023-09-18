#!/usr/bin/env python

# AUTHORS: Katie Fasbender
#          katiefasbender@montana.edu

# nsc_instcal_meas_main.py will run the NSC Measurements process on all exposures
# in a given list on tempest.montana.edu

# This script writes a "job_name.sh" file for an exposure, maintaining "maxjobs" jobs
# running across defined slurm partitions on tempest.
# What that means:
# - Each "job_name.sh" script written by THIS script includes the command to
#   run the mmt process on an exposure from provided NSC exposure list (--list)
# - User defins the slurm partitions (--partition) & how many exposures to process at once (--maxjobs)
# - This script will cycle through the exposure list, submitting jobs to the slurm queue until
#   all exposures are analyzed or something goes wrong.



#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from argparse import ArgumentParser
from astropy.table import Table,Column,vstack
from astropy.io import fits
from dlnpyutils import utils as dln, coords
import logging
import math
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


def write_jscript(job_name,partition,cmd,dir):
    '''writes a SLURM job script to "job_name.sh"
    Arguments:
    ----------
    job_name (str)
            name of job, job script file
    partition (str)
            node/partition the job will run on
    cmd (str)
            python command to run exposure
    dir (str)
            base directory
    Returns:
    --------
    job_file (str)
            job filename the job script is written to
    '''
    job_file = dir+job_name+".sh"
    job_num = int(job_name.split("_")[-1])
    # The following code writes lines to the "job_name.sh" file.
    # Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
    # All other lines are read by the shell
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        if partition=="priority": fh.writelines("#SBATCH --account=priority-davidnidever\n") #specify account
        fh.writelines("#SBATCH --job-name="+job_name+"\n")       # job name
        fh.writelines("#SBATCH --output="+dir+job_name+".out\n") # output file (%j = jobid)
        fh.writelines("#SBATCH --error="+dir+job_name+".err\n")  # error file
        fh.writelines("#SBATCH --partition="+partition+"\n")     # queue partition to run the job in
        fh.writelines("#SBATCH --ntasks=1\n")                    # for running in parallel
        fh.writelines("#SBATCH --nodes=1\n")                     # number of nodes to allocate
        fh.writelines("#SBATCH --ntasks-per-node 1\n")           # number of cores to allocate; set with care
        fh.writelines("#SBATCH --mem=1000\n")                    # memory, set --mem with care!!!!!
        fh.writelines("#SBATCH --time=48:00:00\n")               # Maximum job run time
        if job_num % 1000 == 0:
            fh.writelines("#SBATCH --mail-user katiefasbender@montana.edu\n")
            fh.writelines("#SBATCH --mail-type BEGIN\n")
        fh.writelines("module load Anaconda3\n")                 # load Anaconda module
        fh.writelines("module load GCC\n")                       # load GCC module
        fh.writelines("source activate $HOME/condaenv/\n")       # activate conda environment
        fh.writelines(cmd+"\n")                                  # python cmnd to calculate tracklet(S) orbit
        fh.writelines("conda deactivate")
    return job_file

def sacct_cmd(job_name,outputs,complete=True):
    '''parses the output of a sacct command, returning specified information
    Arguments:
    ----------
    job_name (str)
            you know what this is
    outputs (str list)
            a list of information to get with the sacct command
            see sacct manual page for options
    complete (bool, defailt=True)
            if True, job is complete (necessary to get right sacct format)
    Returns:
    --------
    outputs (list)
            a list of the sacct outputs specified
    '''
    if len(outputs)>1: spref = "sacct -n -P --delimiter=',' --format "
    else: spref = "sacct -n -X --format "
    scommand = (''.join([spref]+[i+"," for i in outputs]))[:-1]+" --name "+job_name
    if complete: job_info = (subprocess.getoutput(scommand).split("\n")[-1]).split(",")
    else: job_info = (subprocess.getoutput(scommand).split("\n")[0]).split(",")
    jinfo = [i.strip() for i in job_info]
    if len(outputs)>1 and len(jinfo)>1: return(jinfo)
    elif len(outputs)>1 and len(jinfo)<=1: return(["" for ou in outputs])
    else: return(jinfo[0])


#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------

if __name__ == "__main__":

    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Calculate orbit of tracklet(s) with Find_Orb')
    parser.add_argument('--comp',type=str,nargs=1,help='Comparison number')
    parser.add_argument('--tlist',type=str,nargs=1,help='Filename with tracklet (or tracklet combo) list')
    parser.add_argument('--partitions',type=str,nargs=1,help='List of slurm partitions to sue')
    parser.add_argument('--maxjobs',type=str,nargs=1,help='Number of jobs to maintain at any given time')
    parser.add_argument('--ntracklets',type=str,nargs=1,help='Number of tracklets to submitt per job')
    parser.add_argument('-c','--combine',action='store_true', help='Combine the input tracklets (treat as one object)')
    args = parser.parse_args()

    # Inputs
    sleep_time=10     # seconds to sleep after submitting a job before checking its job id
    t0 = time.time()
    comp = args.comp[0]                       # 0 = individual tracklets, 1 = pairs, 2 = so on
    combine = args.combine
    partitions=args.partitions[0].split(',')  # the slurm partitions to submit jobs to
    npar=len(partitions)                      # number of slurm partitions
    maxjobs = int(args.maxjobs[0])            # maximum number of jobs to maintain running at any time
    nchan = maxjobs//npar                     # number of job channels kept running on each partition
    ntracklets = int(args.ntracklets[0])      # number of tracklets to submit per job
    inputlist = args.tlist[0]                 # list of tracklet ids to analyze

    # Establish necessary directories - figure out for tempest
    basedir = "/home/x25h971/orbits_dr2/"
    localdir = basedir+"files/"
    outfiledir = basedir+"outfiles/"          # a place for the job files
    makedir(outfiledir)

    # Log File
    #---------
    # Create Log file name;
    # format is nsc_combine_main.DATETIME.log
    ltime = time.localtime()
    # time.struct_time(tm_year=2019, tm_mon=7, tm_mday=22, tm_hour=0, tm_min=30, tm_sec=20, tm_wday=0, tm_yday=203, tm_isdst=1)
    smonth = str(ltime[1])
    if ltime[1]<10: smonth = '0'+smonth
    sday = str(ltime[2])
    if ltime[2]<10: sday = '0'+sday
    syear = str(ltime[0])[2:]
    shour = str(ltime[3])
    if ltime[3]<10: shour='0'+shour
    sminute = str(ltime[4])
    if ltime[4]<10: sminute='0'+sminute
    ssecond = str(int(ltime[5]))
    if ltime[5]<10: ssecond='0'+ssecond
    logtime = smonth+sday+syear+shour+sminute+ssecond
    logfile = basedir+'orbit_wrapper.'+logtime+'.log'

    # Set up logging to screen and logfile
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.NOTSET)

    rootLogger.info("Calculating orbits for NSC tracklets")
    rootLogger.info("partitions = "+str(partitions))

    # Loading the tracklet list
    #--------------------------
    rootLogger.info('Using input list: '+inputlist)
    lstr = Table.read(inputlist)
    nlstr = len(lstr)
    # Define number of orbits to calc
    rootLogger.info(str(nlstr)+' objects')
    obj = np.arange(nlstr)
    nobj = nlstr

    # Set up job structure
    #---------------------
    jstr = lstr.copy()
    jstr['cmd'] = Column(length=len(lstr),dtype="U1000")
    jstr['submitted'] = Column(np.repeat(False,len(lstr)))
    jstr['done'] = Column(np.repeat(False,len(lstr)))
    jstr['jobname'] = Column(length=len(lstr),dtype="U100")
    jstr['jobid'] = Column(np.repeat("      -99.99",len(lstr)))
    jstr['cputime'] = Column(length=len(lstr),dtype="U20")
    jstr['jobstatus'] = Column(length=len(lstr),dtype="U30")
    jstr['partition'] = Column(length=len(lstr),dtype="U30")
    jstr['maxrss'] = Column(length=len(lstr),dtype="U20")

    # Split exposures evenly among defined "partitions" as run with nsc_meas_wrapper
    chans = np.reshape([[i+"_"+str(parchan) for i in partitions] for parchan in range(0,nchan)],maxjobs)
    nchantot = len(chans)
    njobs = math.ceil(nobj/ntracklets)
    print("job channels = ",chans)
    #jstr['partition'] = [chans[(i-maxjobs*(i//maxjobs))] for i in range(njobs)]
    #jstr['partition'] = np.reshape([[chans[(i-maxjobs*(i//maxjobs))] for j in range(ntracklets)] for i in range(njobs)],nobj)
    jstr['partition'] = np.reshape([[chans[(i-maxjobs*(i//maxjobs))] for j in range(ntracklets)] for i in range(njobs)],njobs*ntracklets)[0:nobj]
    if njobs == 0:
        rootLogger.info('No objects to process.')
        sys.exit()
    rootLogger.info('Processing '+str(nobj)+' objects in '+str(njobs)+' jobs, '+str(ntracklets)+' objects/job, on '+str(maxjobs)+' channels across '+str(npar)+' slurm partitions.')

    # Start submitting jobs
    #----------------------
    runfile = basedir+'/orbit_wrapper_'+logtime+'_run.fits'
    jstr.write(runfile)
    jb = 0
    jb_flag = 0
    print("job channels = ",chans)
    while jb_flag==0: # jb_flag = 1 when (jb < njobs)
        for part in chans:
            rootLogger.info("Checking status of last job submitted to "+part+" channel")
            partition_ind = set(np.where(jstr['partition']==part)[0])
            submitted_ind = set(np.where(jstr['submitted']==1)[0])
            unsubmitted_ind = set(np.where(jstr['submitted']==0)[0])

            # get index & status of last job submitted
            last_sub = list(partition_ind & submitted_ind)
            if len(last_sub)==0:
                lsub = [np.sort(list(partition_ind))[0:ntracklets]][0]
            else:
                lsub = [np.sort(last_sub)[(-1)*ntracklets:-1]][0]
            print("indices of last submitted job = ",lsub)
            lj_id = jstr[lsub[0]]['jobid']
            last_jname = jstr[lsub[0]]['jobname']
            print("last job info = ",lj_id,last_jname)
            if last_jname!='':
                lj_info = sacct_cmd(last_jname,["state","jobid"])
                print("last job info = ",lj_info)
                if len(lj_info)>1:
                    lj_status = lj_info[0].strip()
                    lj_id = lj_info[1].strip()
                else:
                    lj_status = "SACCT_ERR"
                    lj_id = "-99.99"
            else:
                lj_status = "NONE" #no jobs have been submitted
                lj_id = "-99.99"
            jstr['jobid'][lsub] = lj_id
            jstr['jobstatus'][lsub] = lj_status
            print("lj_status = ",lj_status,", jobname = ",last_jname,lj_id)
            # ---If last job is still running: skip to next partition!
            if (lj_status=="RUNNING" or lj_status=="PENDING" or lj_status=="REQUEUED"):
                rootLogger.info("Job id="+lj_id+" is still "+lj_status+", check next partition")
                time.sleep(1)
            # ---If last job is completed, failed, cancelled, killed, or none: submit a new job!
            else:
                rootLogger.info("--Submitting new job to "+part+" partition--")
                print("last job status = ",lj_status)
                # if last job was completed, get some info about it
                if lj_status=="COMPLETED":
                    ljinfo = sacct_cmd(last_jname,['cputimeraw','maxrss'])
                    jstr['done'] = True
                    print("last job info = ",ljinfo)
                    if len(ljinfo)>1:
                        jstr['cputime'][lsub] = ljinfo[0]
                        jstr['maxrss'][lsub] = ljinfo[1]

                # get index & info of next job to submit
                next_sub = list(partition_ind & unsubmitted_ind)
                if len(next_sub)==0:
                    jbsub = [np.sort(next_sub)[(nobj-ntracklets):]][0]
                else:
                    jbsub = [np.sort(next_sub)[0:ntracklets]][0]
                # create and submit the job!
                tlets = ",".join([str(i) for i in jstr['tracklet_id'][jbsub]])
                p32s = ",".join([str(i) for i in jstr['pix32'][jbsub]])
                foids = ",".join([str(i) for i in jstr['fo_id'][jbsub]])
                cmd = 'python '+localdir+'orbit_calc.py --comp '+str(comp)+' --tracklet '+tlets+' --pix32 '+p32s+' --foid '+foids
                jstr['cmd'][jbsub[0]] = cmd
                partition = jstr['partition'][jbsub[0]].split("_")[0]
                rootLogger.info("-- Submitting new job--")
                # --Write job script to file--
                job_name = 'orbc'+str(comp)+"_"+str(logtime)+'_'+str(jb)
                job_file=write_jscript(job_name,partition,cmd,outfiledir)
                # --Submit job to slurm queue--
                os.system("sbatch %s" %job_file)
                jstr['submitted'][jbsub] = True
                rootLogger.info("Job "+job_name+"  submitted to "+part+" partition")
                jstr['jobname'][jbsub] = job_name
                jb+=1

            # save job structure
            jstr.write(runfile,overwrite=True)
            time.sleep(5)
        if jb==njobs: jb_flag = 1
