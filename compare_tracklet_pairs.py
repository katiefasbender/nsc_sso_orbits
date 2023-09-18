#!/usr/bin/env

# AUTHOR: Katie Fasbender
#         katiefasbender@montana.edu


# This script will submit a slurm job for each healpix (NSIDE=32) to create
# lists of tracklet pairs from the NOIRLab Source Catalog (NSC),
# then will submit each pair for Find_Orb analysis as a slurm job.  

# Input: full list of HP32 with tracklets in the NSC, fits file/astropy table
#         -> for each HP32, a list of tracklet pairs will be created 
#               (with tracklet_link.py)
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
import sys
import time


#-------------
# Functions
#-------------

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
        fh.writelines("#SBATCH --cpus-per-task 2\n")             # number of cores to allocate; set with care
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

def sacct_cmd(job_name,outputs):
    '''parses the output of a sacct command, returning specified information
    Arguments:
    ----------
    job_name (str)
            you know what this is
    outputs (str list)
            a list of information to get with the sacct command
            see sacct manual page for options
    Returns:
    --------
    outputs (list)
            a list of the sacct outputs specified
    '''
    if len(outputs)>1: spref = "sacct -n -P --delimiter=',' --format "
    else: spref = "sacct -n -X --format "
    scommand = (''.join([spref]+[i+"," for i in outputs]))[:-1]+" --name "+job_name
    job_info = (subprocess.getoutput(scommand).split("\n")[0]).split(",")
    jinfo = [i.strip() for i in job_info]
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
    parser.add_argument('--dr',type=int,nargs=1,help='NSC data release')
    parser.add_argument('-r','--redo',action='store_true', help='Redo lists')
    args = parser.parse_args()

    # Inputs
    listname = args.trackletlist[0]
    dr = int(args.dr[0])
    redo = args.redo
    tlet_list = Table.read(listname)
    hp32list = np.unique(tlet_list['pix32'])
    basedir = "/home/x25h972/orbits/dr"+str(dr)+"/"

    # Setup Job structure
    print("Checking tracklet lists for "+str(len(hp32list))+" HEALPix (NSIDE32)")
    jstruct = Table()
    jstruct['pix32'] = hp32list
    jstruct['cmd'] = Column(dtype="U1000",length=len(hp32list))    
    jstruct['outfile'] = Column(dtype="U500",length=len(hp32list))
    jstruct['done'] = Column(np.repeat(False,len(hp32list)))
    jstruct['torun'] = Column(np.repeat(True,len(hp32list)))
    jstruct['jobname'] = Column(dtype="U500",length=len(hp32list)) 
    jstruct['jobid'] = Column(dtype="U20",length=len(hp32list)) 
    jstruct['jobstatus'] = Column(dtype="U20",length=len(hp32list)) 
    jstruct['cputime'] = Column(dtype="U20",length=len(hp32list)) 
    jstruct['maxrss'] = Column(dtype="U20",length=len(hp32list)) 
    for px in range(len(hp32list)):
        # --get the pix32 and setup its list outdir--
        pix32 = np.unique(tlet_list['pix32'])[px]
        hgroup32dir = basedir+"tpair_lists/hgroup32_" +str(pix32//1000)+"/"
        # --get the list of hgroup128s for this pix32--
        pix_list = tracklets[np.where(tracklets['pix32']==pix32)[0]]
        pix128s = list(np.unique(pix_list['pix128']))
        hgroups128=np.unique(pix128s//1000)
        cmd = "python "+basedir+"tracklet_link.py "+str(int(pix32))+" "+str(pix128s)
        outfile = hgroup32dir+str(pix32)+".fits"
        # --should we make a list for this pix32?
        if os.path.exists(outfile):      # if the outfile exists...
            jstruct['done'][px] = True
            if not redo:                 # ...and if we're not redoing,
                jstruct['torun'] = False # don't make a list.

    torun = np.where(jstruct['torun']==1)[0]
    ntorun = len(jstruct[torun]))
    endflag = 0
    print("creating tracklet lists for ",ntorun," pix32")
    while endflagg==0: