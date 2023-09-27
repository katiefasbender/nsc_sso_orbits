#!/usr/bin/env python

# AUTHORS: Katie Fasbender
#          katiefasbender@montana.edu

# comp_wrapper.py will execute orbit_calc.py for all path pairs in a given list.

# This script writes a "job_name.sh" file for each pair of paths,
# maintaining "maxjobs" jobs running across defined slurm partitions on tempest.msu.montana.edu
# What that means:
# - Each "job_name.sh" script written by THIS script includes the command to
#   run orbit_calc.py on NSC tracklet(s) (given in --tlist)
# - The user defins the slurm partitions (--partitions) & how many exposures to process at once (--maxjobs)
# - This script will cycle through the tracklet list, submitting jobs to the slurm queue until
#   all tracklets have been processed or something goes wrong.



#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from argparse import ArgumentParser
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
        fh.writelines("#SBATCH --mem=1000\n")                    # memory, set --mem with care!!!!! refer to hyalite quickstart guide
        fh.writelines("#SBATCH --time=48:00:00\n")               # Maximum job run time
        fh.writelines("module load Anaconda3\n")                 # load Anaconda module
        fh.writelines("module load GCC\n")                       # load GCC module
        fh.writelines("source activate $HOME/condaenv/\n")       # activate conda environment
        fh.writelines(cmd+"\n")                                  # python cmnd to calculate tracklet(S) orbit
        fh.writelines("conda deactivate")
    return job_file


#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------

if __name__ == "__main__":

    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Calculate orbit of tracklet(s) with Find_Orb (FO)')
    parser.add_argument('--comp', type=str, nargs=1, help='Comparison number')
    parser.add_argument('--list', type=str, nargs=1, help='Filename with list of paths to analyze with orbit_calc.py')
    parser.add_argument('--partitions', type=str, nargs=1, help='List of slurm partitions to use')
    parser.add_argument('--maxjobs', type=str, nargs=1, help='Number of jobs to maintain at any given time')
    parser.add_argument('-c','--combine', action='store_true', help='Combine input tracklets of each FO cmd (treat as one object)')
    parser.add_argument('-r','--redo', action='store_true', help='Redo tracklets that were previously processed')
    args = parser.parse_args()

    # Inputs
    comp = args.comp[0]                      # 0 = individual tracklets, 1 = pairs, 2 = so on
    redo = args.redo
    combine = args.combine
    partitions=args.partitions[0].split(',')  # the slurm partitions to submit jobs to
    npar=len(partitions)                     # number of slurm partitions
    maxjobs = int(args.maxjobs[0])           # maximum number of jobs to maintain running at any time
    cpar = maxjobs//npar                     # number of job channels kept running on each partition
    inputlist = args.tlist                    # list of exposures to analyze
    if inputlist is not None:
        inputlist = inputlist[0]

    # Establish necessary directories - figure out for tempest
    basedir = "/home/x25h971/orbits_dr2/"
    localdir = basedir+"files/"
    outfiledir = basedir+"outfiles/"                     # a place for the job files
    makedir(outfiledir)
    t0 = time.time()

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
    rootLogger.info("redo = "+str(redo))

    # Loading the tracklet list
    #--------------------------
    rootLogger.info('Using input list: '+inputlist)
    lstr = Table.read(inputlist)
    nlstr = dln.size(lstr)
    rootLogger.info(str(nlstr)+' objects')
    gdobj = np.arange(nlstr)
    ngdobj = nlstr


    # Check the tracklet input list for Find_Orb output files
    #--------------------------------------------------------
    rootLogger.info('Checking on the tracklets')
    jstr = lstr.copy()
    jstr['outfile'] = Column(length=len(lstr),dtype="U100")
    jstr['cmd'] = Column(length=len(lstr),dtype="U1000")
    jstr['submitted'] = Column(np.repeat(False,len(lstr)))
    jstr['torun'] = Column(np.repeat(False,len(lstr)))
    jstr['done'] = Column(np.repeat(False,len(lstr)))
    jstr['jobname'] = Column(length=len(lstr),dtype="U100")
    jstr['jobid'] = Column(np.repeat("      -99.99",len(lstr)))
    jstr['cputime'] = Column(length=len(lstr),dtype="U20")
    jstr['jobstatus'] = Column(length=len(lstr),dtype="U30")
    jstr['partition'] = Column(length=len(lstr),dtype="U30")
    jstr['maxrss'] = Column(length=len(lstr),dtype="U20")

    partions = np.reshape([[i+"_"+str(parchan) for i in partitions] for parchan in range(0,cpar)],maxjobs)
    nchan = len(partions)
    pt = 0
    njobs = (ngdobj//10)+1 # number of potential jobs = number of tracklets/10, OR
    if combine: njobs = ngdobj                      # = number of tracklet combos
    for i in range(njobs): # for each set of 10 tracklets or for each tracklet combo,
        ind = np.array(list(np.array(range(10))+(10*i)))
        ind = list(ind[ind<ngdobj])
        if combine: ind = [i]

        # Check if the output already exists.
        subdir = int(jstr['fo_id'][ind[0]].split("t")[1])//10
        if combine: subdir = jstr['comp_id'][i]
        outdir = basedir+"comp"+comp+"/fgroup_"+str(subdir//1000)
        outfile = outdir+"/fo_comp"+str(comp)+"_"+str(subdir)+".txt"
        jstr['outfile'][ind] = outfile
        # Does the output file exist?
        if os.path.exists(outfile): jstr['done'][ind] = True
        # If no outfile exists or yes redo, set up the command
        if (jstr['done'][ind][0]==False) or (redo==True):
            tlets = list(jstr['tracklet_id'][ind])
            p32s = [str(i) for i in jstr['pix32'][ind]]
            foids = list(jstr['fo_id'][ind])
            jstr['cmd'][ind] = 'python '+localdir+'orbit_calc.py --comp '+str(comp)+' --tracklet '+(",".join(list(tlets)))+' --pix32 '+(",".join(list(p32s)))+' --foid '+(",".join(list(foids)))
            jstr['torun'][ind] = True
            jstr['partition'][ind] = partions[pt]
            pt+=1
            pt = pt-(pt//nchan)*nchan

    # Parcel out jobs
    #----------------
    # Define number of orbits to calculate & total #jobs/partition
    torun,nalltorun = dln.where(jstr['torun'] == True)    # Total number of jobs to run (# exposures)
    ntorun = len(torun)
    njtorun = (ntorun//10)+1
    if combine: njtorun = ntorun
    rootLogger.info(str(ntorun)+" jobs...")
    if ntorun == 0:
        rootLogger.info('No objects to process.')
        sys.exit()
    rootLogger.info('...on '+str(maxjobs)+' channels across '+str(npar)+' slurm partitions.')
    sleep_time=10     # seconds to sleep after submitting a job before checking its job id

    # Start submitting jobs
    #----------------------
    runfile = basedir+'/orbit_wrapper.'+logtime+'_run.fits'
    Table(jstr).write(runfile)
    jb = 0
    jb_flag = 0
    while jb_flag==0: # jb_flag = 1 when (jb < ntorun)
        for part in partions:
            rootLogger.info("Checking status of last job submitted to "+part+" channel")
            partition_ind = set(np.where(jstr['partition'][torun]==part)[0])
            submitted_ind = set(np.where(jstr['submitted'][torun]==1)[0])
            unsubmitted_ind = set(np.where(jstr['submitted'][torun]==0)[0])

            # get index & status of last job submitted
            last_sub = list(partition_ind & submitted_ind)
            if len(last_sub)==0:
                lsub = list(np.sort(list(partition_ind))[0:10])
                if combine: lsub = list(np.sort(list(partition_ind))[0])
            else:
                lsub = list(np.sort(last_sub)[-10:])
                if combine: lsub = list(np.sort(last_sub)[-1])
            lj_id = jstr[torun[lsub[0]]]['jobid']
            last_jname = jstr[torun[lsub[0]]]['jobname']
            if last_jname!='':
                lj_info = (subprocess.getoutput("sacct -n -P --delimiter=',' --format state,jobid --name "+last_jname).split("\n")[0]).split(",")
                print("last job info = ",lj_info)
                if len(lj_info)>1:
                    lj_status = lj_info[0].strip()
                    lj_id = lj_info[1].strip()
                else:
                    lj_status = "RUNNING"
                    lj_id = "-99.99"
            else:
                lj_status = "NONE" #no jobs have been submitted
                lj_id = "-99.99"
            jstr['jobid'][torun[lsub]] = lj_id
            jstr['jobstatus'][torun[lsub]] = lj_status
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
                    ljinfo = (subprocess.getoutput("sacct -n -P --delimiter=',' --format cputimeraw,maxrss --jobs "+lj_id)).split("\n")[-1].split(",")
                    jstr['done'] = True
                    print("last job info = ",ljinfo)
                    if len(ljinfo)>1:
                        jstr['cputime'][torun[lsub]] = ljinfo[0]
                        jstr['maxrss'][torun[lsub]] = ljinfo[1]

                # get index & info of next job to submit
                next_sub = list(partition_ind & unsubmitted_ind)
                if len(next_sub)==0:
                    jbsub = list(np.array(range(10))+(ntorun-10))
                    if combine: jbsub = list(ntorun-1)
                else:
                    jbsub = list(np.sort(next_sub)[0:10])
                    if combine: jbsub = list(np.sort(next_sub)[0])

                # create and submit the job!
                cmd = jstr['cmd'][torun[jbsub[0]]]
                partition = jstr['partition'][torun[jbsub[0]]].split("_")[0]
                print("-- Submitting new job--")

                # --Write job script to file--
                job_name = 'orbc'+str(comp)+"_"+str(logtime)+'_'+str(jb)
                job_file=write_jscript(job_name,partition,cmd,outfiledir)

                # --Submit job to slurm queue--
                os.system("sbatch %s" %job_file)
                jstr['submitted'][torun[jbsub]] = True
                rootLogger.info("Job "+job_name+"  submitted to "+part+" partition")
                jstr['jobname'][torun[jbsub]] = job_name
                jb+=1

            # save job structure
            Table(jstr).write(runfile,overwrite=True)
