#!/usr/bin/env

# AUTHOR: Katie Fasbender
#         katiefasbender@montana.edu


# This script will submit a slurm job for each healpix (NSIDE=32) to create
# lists of tracklet pairs from the NOIRLab Source Catalog (NSC),
# then will submit each pair for Find_Orb analysis as a slurm job.

# Input: full list of HP32 with tracklets in the NSC, fits file/astropy table
#         -> for each HP32, a list of tracklet pairs will be created
#               (with path_link.py)
#         -> for each pair of tracklets, orbital elements will be calculated
#               (with Find_Orb & orbit_calc.py)
# Output =
#         -> for each HP32, full list of tracklet pairs
#         -> for each pair, appropriate Find_Orb output files for...
#               ephemeris
#               orbital elements
#               #mmts used for calculation
#               computed observations used for calculation

#-------------
# Imports
#-------------
from argparse import ArgumentParser
from astropy.coordinates import SkyCoord
from astropy.table import Column,Row,Table,vstack
from astropy.time import Time
import astropy.units as u
import healpy as hp
import numpy as np
import os
import subprocess
import sys
import time


#-------------
# Functions
#-------------

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


def write_jscript(job_name,partition,cmds,dir,single=True):
    '''writes a SLURM job script to "job_name.sh"
    Arguments:
    ----------
    job_name (str)
            name of job, job script file
    partition (str)
            node/partition the job will run on
    cmds (str or str list)
            python command(s) to run 
    dir (str)
            directory for output files
    single (bool, default True)
            if you have a list of commands, set single=False
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
        if single==True: fh.writelines("#SBATCH --ntasks=1\n")   # for running in parallel
        else: fh.writelines("#SBATCH --ntasks="+str(len(cmds))+"\n")
        fh.writelines("#SBATCH --cpus-per-task 4\n")             # number of cores to allocate; set with care
        fh.writelines("#SBATCH --mem-per-cpu=3000\n")           # memory, set --mem with care!!!!! refer to hyalite quickstart guide
        if single==True: fh.writelines("#SBATCH --time=48:00:00\n")
        else: fh.writelines("#SBATCH --time=00:05:00\n")         # Maximum job run time
        fh.writelines("module load Anaconda3\n")                 # load anaconda, needed for running python on hyalite/tempest
        fh.writelines("module load GCC\n")                       # load GCC module
        fh.writelines("source activate $HOME/condaenv/\n")       # activate conda environment
        if single: fh.writelines(cmds+"\n")                       # write python command to analyze exposure
        else:                                                    # OR, write python cmds to download exposures
            for cmd in cmds:
                fh.writelines("srun -n 1 "+cmd+" & \n")
            fh.writelines("wait\n")
        fh.writelines("conda deactivate")
    return job_file

def sacct_cmd(job_name,outputs,c=False,m=False):
    '''parses the output of a sacct command, returning specified information
    Arguments:
    ----------
    job_name (str)
            you know what this is
    outputs (str list)
            a list of information to get with the sacct command
            see sacct manual page for options
    c (bool)
            if job is known to be completed, default=False
    m (bool)
            if multiple tasks in the job, default=False
    Returns:
    --------
    outputs (list)
            a list of the sacct outputs specified
    '''
    if len(outputs)>1: spref = "sacct -n -P --delimiter=',' --format "
    else: spref = "sacct -n -X --format "
    scommand = (''.join([spref]+[i+"," for i in outputs]))[:-1]+" --name "+job_name
    if c and not m: jindex=1
    elif c and m: jindex=2
    else: jindex=0
    job_info = (subprocess.getoutput(scommand).split("\n")[jindex]).split(",")
    jinfo = [i.strip() for i in job_info]
    print("sacct cmd = ",scommand)
    print("sacct output = ",jinfo)
    if len(outputs)>1 and len(jinfo)>1: return(jinfo)
    elif len(outputs)>1 and len(jinfo)<=1: return(["" for ou in outputs])
    else: return(jinfo[0])


#-------------
# Main Code
#-------------
if __name__=="__main__":


    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Link tracklets detected in the NOIRLab Source Catalog')
    parser.add_argument('--trackletlist',type=str,nargs=1,help='List of healpix (NSIDE=32) with tracklet mmts')
    parser.add_argument('--comp',type=int,nargs=1,help='Iteration of path comparison')
    parser.add_argument('--partitions',type=str,nargs=1,help='List of slurm partitions to sue')
    parser.add_argument('--maxjobs',type=str,nargs=1,help='Number of jobs to maintain at any given time')
    parser.add_argument('-r','--redo',action='store_true', help='Redo lists')
    args = parser.parse_args()

    # Inputs
    logtime = time.time()
    listname = args.trackletlist[0]
    comp = int(args.comp[0])                  # 1 for tracklet pairs, 2+ for everything else
    if comp==1:
        nside = 32
    else:
        nside = 64
    partitions=args.partitions[0].split(',')  # the slurm partitions to submit jobs to
    npar=len(partitions)                      # number of slurm partitions
    maxjobs = int(args.maxjobs[0])            # maximum number of jobs to maintain running at any time
    nchan = maxjobs//npar                     # number of job channels kept running on each partition
    redo = args.redo
    tlet_list = Table.read(listname)
    hplist = np.unique(tlet_list['pix'+str(nside)])
    nhp = len(hplist)
    basedir = "/home/x25h971/orbits/"
    outdir = basedir+"files/lists/comp"+str(comp)+"/"
    outfiledir = basedir+"outfiles/"

    # Setup Job structure
    print("Checking tracklet lists for "+str(len(hplist))+" HEALPix (NSIDE"+str(nside)+")")
    jstruct = Table()
    jstruct['pix'+str(nside)] = hplist
    jstruct['partition'] = Column(dtype="U20",length=nhp)
    jstruct['cmd'] = Column(dtype="U1000",length=nhp)
    jstruct['outfile'] = Column(dtype="U500",length=nhp)
    jstruct['done'] = Column(np.repeat(False,nhp))
    jstruct['torun'] = Column(np.repeat(True,nhp))
    jstruct['submitted'] = Column(np.repeat(False,nhp))
    jstruct['jobname'] = Column(dtype="U500",length=nhp)
    jstruct['jobid'] = Column(dtype="U20",length=nhp)
    jstruct['jobstatus'] = Column(dtype="U20",length=nhp)
    jstruct['cputime'] = Column(dtype="U20",length=nhp)
    jstruct['maxrss'] = Column(dtype="U20",length=nhp)
    for px in range(nhp):
        if px%1000==0: print(px,"/",nhp)
        # --get the pix32 and setup its list outdir--
        pix = np.unique(tlet_list['pix'+str(nside)])[px]
        hgroupdir = outdir+"hgroup"+str(nside)+"_" +str(int(pix//1000))+"/"
        makedir(hgroupdir)
        # --get the list of hgroup128s for this pix32--
        pix_list = tlet_list[np.where(tlet_list['pix'+str(nside)]==pix)[0]]
        pix128s = list(np.unique(pix_list['pix128']))
        p128s = ","
        for i in pix128s:
            p128s = p128s+str(int(i))+","
        p128s = p128s[1:-1]
        #print("pix ",pix,", ",pix128s,p128s)
        hgroups128=[int(int(i)//1000) for i in np.unique(pix128s)]
        cmd = "python "+basedir+"files/path_link.py --nside "+str(nside)+" --hp "+str(int(pix))+" --hp128 "+p128s+" --comp "+str(comp)
        outfile = hgroupdir+str(int(pix))+".fits"
        jstruct['cmd'][px] = cmd
        jstruct['outfile'][px] = outfile
        # --should we make a list for this pix?
        #print(outfile)
        if os.path.exists(outfile):      # if the outfile exists...
            #print(outfile," exists")
            jstruct['done'][px] = True
            if not redo:                 # ...and if we're not redoing,
                jstruct['torun'][px] = 0 # don't make a list.
    # Define hpix to run
    torun = np.where(jstruct['torun']==True)[0]
    ntorun = len(jstruct[torun])
    # Split hpix evenly among defined "partitions"/channels
    chans = np.reshape([[i+"_"+str(parchan) for i in partitions] for parchan in range(0,nchan)],maxjobs)
    nchantot = len(chans)
    njobs = ntorun
    print("job channels = ",chans)
    jstruct['partition'][torun] = [chans[(i-maxjobs*(i//maxjobs))] for i in range(njobs)]
    #jstr['partition'] = np.reshape([[chans[(i-maxjobs*(i//maxjobs))] for j in range(njobs)] for i in range(njobs)],njobs)
    print(jstruct['partition'])
    #jstr['partition'] = np.reshape([[chans[(i-maxjobs*(i//maxjobs))] for j in range(ntracklets)] for i in range(njobs)],njobs*ntracklets)[0:nobj]
    if njobs == 0:
        print('No objects to process.')
        sys.exit()
    print('Processing '+str(njobs)+' healpix (NSIDE'+str(nside)+'), on '+str(maxjobs)+' channels across '+str(npar)+' slurm partitions.')



    # Start submitting jobs
    #----------------------
    runfile = basedir+'files/lists/runfiles/comp'+str(comp)+'.'+str(logtime)+'_run.fits'
    jstruct['jobid']="-99.99"
    jstruct.write(runfile)

    print("creating tracklet lists for ",ntorun," pix32")
    jb = 0        # for jobname
    endflag = 0   #
    while endflag==0:
        # - For each partition/channel----------------------------------------------------------------------------------------------------
        for part in chans:
            partition_ind = set(np.where(jstruct['partition'][torun]==part)[0])  # indices for this partition
            submitted_ind = set(np.where(jstruct['submitted'][torun]==1)[0])     # indices for submitted hp32 (to get lsub for last sub)
            unsubmitted_ind = set(np.where(jstruct['submitted'][torun]==0)[0])   # indices for unsubmitted hp32(to get tsub for this sub)
            sub = list(partition_ind & submitted_ind)                            # indices of jobs submitted to channel (last_sub)
            unsub = list(partition_ind & unsubmitted_ind)                        # indices of jobs unsubmitted to channel (this_sub)
            subflag = 0                                                          # whether to submit a new job, 1 if yes
            # -- Check on last job submitted to channel---------------------------------------------------------------------------------
            if len(sub)>0:                                                     # if there were jobs submitted to this channel...
                lsub = [np.sort(sub)[-1]]
                lsub_jname = jstruct['jobname'][torun[lsub]][0]
                lsub_jstat_logged = jstruct['jobstatus'][torun[lsub]][0].strip()
                print("lsub_jname = ",lsub_jname,lsub_jstat_logged)
                if maxjobs<5:
                    print("wait a sec before checking last proc job...")	       # check the status of last proc job
                    time.sleep(10)
                lsub_jstat,lsub_jid = sacct_cmd(lsub_jname,["state","jobid"])
                jstruct['jobstatus'][torun[lsub]] = lsub_jstat
                jstruct['jobid'][torun[lsub]] = lsub_jid
                if lsub_jstat!="RUNNING" and lsub_jstat!="PENDING" and lsub_jstat!="REQUEUED":
                    subflag = 1                                                        # we can submit a new job to this chan!
                    if lsub_jstat=="COMPLETED":                                        # if last job is now complete,
                        ofile = jstruct['outfile'][torun[lsub]][0].strip()                 # check for outfile
                        print("outfile = ",ofile)
                        if os.path.exists(ofile):
                            print("outfile exists")
                            #jcomp+=1
                            cputime,maxrss = sacct_cmd(lsub_jname,["cputimeraw","maxrss"],c=True)
                            jstruct['done'][torun[lsub]] = 1
                            jstruct['cputime'][torun[lsub]] = cputime
                            jstruct['maxrss'][torun[lsub]] = maxrss
                print("last job in this partition is ",lsub_jstat)
            else: subflag = 1                                                  # else, if nothing submitted to channel, submit a job!
            print("subflag = ",subflag," for ",part," partition")
            # -- Line up next job to be submitted---------------------------------------------------------------------------------------
            if len(unsub)>0 and subflag==1:                                    # if there are jobs to submit and the flag says yes,
                nsb = [np.sort(unsub)[0]]
                cmd = jstruct['cmd'][torun[nsb]][0]
                partition = jstruct['partition'][torun[nsb]][0].split("_")[0]
                print("--Submit Job for healpix32 "+str(jstruct['pix32'][torun[nsb]][0])+"--")
                # -> Write job script to file
                job_name = 'comp'+str(comp)+'_'+str(logtime)+'_'+str(jb)
                job_file=write_jscript(job_name,partition,cmd,outfiledir,single=True)
                # -> Submit job to slurm queue
                os.system("sbatch "+str(job_file))
                jstruct['submitted'][torun[nsb]] = True
                jb+=1
                print("Job "+job_name+"  submitted to "+partition+" partition")
                jstruct['jobname'][torun[nsb]] = job_name
            else:
                if len(jstruct[jstruct['done']==1])==ntorun: endflag==1    # we can end it!
            time.sleep(1)
        # - Save runfile and move on to next partition------------------------------------------------------------------------------------
        jstruct.write(runfile,overwrite=True)
        if ntorun<10 or maxjobs<5:
           print("wait a sec...")
           time.sleep(15)
        else: time.sleep(1)
    print("tracklet lists created for all healpix32!")
