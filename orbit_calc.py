#!/usr/bin/env python

# AUTHOR: Katie Fasbender
#         katiefasbender@montana.edu


# This script will calculate the orbit for a tracklet(s) or a combination of tracklets,
# for a HEALPix (NSIDE=32) in the NSC.  For the individual tracklet(s) or combination,
# Find_Orb will calculate the orbit, outputting 1 set of 3 files for the orbit.

# The wrapper for this script should be fed a list of tracklets or tracklet combos.

# Input = (a) tracklet (combo) list,  with a column for the number of tracklets
#         (b) fo_id; the Find_Orb ID for the tracklet(s) you want an orbit for,
#             fo_id is unique to the above tracklet list

# Output = (a) 1 text file for the measurements of each tracklet (combo)
#              in MPC 80-col format, the input format for Find_Orb
#          (b) appropriate Find_Orb output files will be written (3 per orbit):
#              1) fo_comp<comp#>_<fo_id>_elem.txt  # fo orbital elements outfile, in MPC 80-col format
#              2) fo_comp<comp#>_<fo_id>_ephem.txt # fo ephemeris outfile
#              3) fo_comp<comp#>_<fo_id>_cobs.txt  # fo calculated observations outfile

#-------------
# Imports
#-------------
from argparse import ArgumentParser
from astropy.coordinates import SkyCoord
from astropy.table import Table,Column,vstack
from astropy.time import Time
import astropy.units as u
from dlnpyutils.utils import *
import numpy as np
import os
import sys
import subprocess
import time
from orbit_func import *

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

#-------------
# Main Code
#-------------

if __name__=="__main__":

    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Calculate orbit of tracklet(s) with Find_Orb')
    parser.add_argument('--comp', type=str, nargs=1, help='Comparison number')
    parser.add_argument('--tracklets',type=str,nargs=1,help="Delimited lislt of tracklets' ID(s)")
    parser.add_argument('--foids',type=str,nargs=1,help="Delimited lislt of tracklets' Find_Orb ID(s), if comp0")
    parser.add_argument('--testid',type=str,nargs=1,help="Delimited lislt of tracklets' test ids ID(s), if comp1+") # when a path is created to be tested
    parser.add_argument('--pix32s', type=str, nargs=1, default=1, help='HEALPix value(s) (NSIDE=32) of tracklet(s) location')
    parser.add_argument('-c','--combine', action='store_true', help='Combine the input tracklets (treat as one object)')
    parser.add_argument('-r','--redo', action='store_true', help='Redo tracklets that were previously processed')
    args = parser.parse_args()

    # Inputs
    tracklets = args.tracklets[0].split(",") # the tracklet id(s) to calculate orbit(s) for (index#.pix128)
    fo_ids = args.foids[0].split(',')        # the tracklet fo_id(s) to calculate orbit(s) for
    test_id = args.testid[0]
    testid = "p"+str(int(test_id)).zfill(6)
    #paths = args.paths[0].split(',')
    comp = args.comp[0]                      # 0 = individual tracklets, 1 = pairs, 2 = so on
    pix32 = args.pix32s[0].split(',')        # if tracklets in multiple pix32s, will be in fmt "pix32.<tracklet_#>"
    tracklet_sdirs = [int(i.split(".")[-1])//1000 for i in tracklets]
    mpc_files = list(np.unique([int(i.split(".")[-1])//1000 for i in tracklets])) #get a list of unique pix128 where tracklets are
    redo = args.redo
    combine = args.combine

    # Directories
    basedir = "/home/x25h971/orbits/"
    outdir = basedir+"dr2/comp"+comp+"/"  # +str(fo_id//10000) added later in code, for fo output
    cfdir = "/home/x25h971/canfind_dr2/"
    t0 = time.time()
    print("time0 = ",t0)

    #print("mpc_files = ",mpc_files)
    ## --get the MPC 80-col lines from the appropriate hgroup text files--
    ##--------------------------------------------------------------------
    hgroups_cat = Table()
    for fil in mpc_files:
        print("file = ",fil,", time = ",time.time())
        ftbools = np.array(tracklet_sdirs)==fil
        file_tlets = list(np.array(tracklets)[ftbools]) # tracklet mmtss found in file
        ftable = read_cfmpc80(cfdir+"concats/cf_dr2_hgroup_"+str(fil)+".txt",specific=True,tlets=file_tlets)
        print("ftable = \n",ftable)
        if ftable: hgroups_cat = vstack([hgroups_cat,ftable])
        #print("file = ",fil," time = ",time.time())
    print(len(hgroups_cat)," tracklet mmts")

    ## --for each tracklet (or tracklet combo), get Find_Orb (fo) info--
    ##--------------------------------------------------------------------
    #print("arbitrary test point 1 ",time.time())
    if combine: fo_hgroup32 = outdir+"hgroup32_"+str(int(pix32[0])//1000)
    else: fo_hgroup32 = outdir + "fgroup_" + str(int(fo_ids[0][1:])//10000)
    makedir(fo_hgroup32)
    if combine: fo_filebase = fo_hgroup32+"/fo_comp"+str(comp)+"_"+str(pix32[0])+"."+str(test_id)
    else: fo_filebase = fo_hgroup32+"/fo_comp"+str(comp)+"_"+str(fo_ids[0]).split("t")[-1]
    fo_infile = fo_filebase+"_obs.txt" # fo input file, change pair_id to fo_id
    if os.path.exists(fo_infile): os.remove(fo_infile)
    fo_file = open(fo_infile,'w')
    fo_outfile_elem = fo_filebase+"_elem.txt"   # fo output file (orbital elements, MPC 8-line fmt)
    fo_outfile_ephem = fo_filebase+"_ephem.txt" # fo output file (ephemeris)
    fo_outfile_cobs = fo_filebase+"_cobs.txt"   # fo output file (calculated observations)

    # Write a text file to submit to Find_Orb
    tracklet_mmts = Table()
    ##if combine: # First, we're combining the tracklets or paths as a "test path", get individual elems
    ##    indiv_orbit_elems=Table()
    ##    for path in np.unique(paths):
    ##        elems = read_fo_elem()
    if combine: fo_ids = np.repeat("0",len(tracklets))
    for foid,tid in zip(fo_ids,tracklets): # for each tracklet,
        # get mmt MPC80col lines for all tracklets (from hgroups_cat),
        # and write lines to a text file to put into Find_Orb
        print("looking for tracklet mmts...stand by!")
        t_mmts = hgroups_cat[hgroups_cat['tracklet_id']==tid] # tracklet mmts
        #print("tracklet mmts : ",t_mmts)
        #if combine: # if it's a tracklet combination, must get individual tracklet orit information
            #indiv_elem_file = "cfdr2_"+str(id)+"_"+str()+".txt" # file with tracklet information
            #indiv_elems = read_fo_elem(indiv_elem_file)
        lines = t_mmts['line']
        obs_codes = []
        for ln in lines:
            if combine: fo_file.write("     "+testid+" "+ln+"\n")
            else: fo_file.write("     "+foid+" "+ln+"\n")
            #print(ln)
            obs_codes.append(ln.split(" ")[-3])
        tracklet_mmts = vstack([tracklet_mmts,t_mmts])
    fo_file.close() # finish writing the fo input file
    print("Find_Orb input file ",fo_infile," written")
    tracklet_mmts.sort('mjd')

    # if this is a tracklet combination, we need to get some info for
    # our output ephemeris, to add to the fo command
    if combine:
        # Figure out timespan for ephemeris reporting
        obs_code = obs_codes[0]
        print("path mjd = ",tracklet_mmts['mjd'][0])
        mjd = Time(tracklet_mmts['mjd'][0],format="mjd",scale="utc") #***********MAY BE INCORRECT
        mjd2 = Time(tracklet_mmts['mjd'][-1],format="mjd",scale="utc")
        print("tracklet datetime = ",mjd.datetime)
        #dmjd = tracklet_mmts['mjd'][-1]-tracklet_mmts['mjd'][0] #daays
        dmj = (mjd2 - mjd)
        dmjd = dmj.jd
        print("path dmjd = ",dmj.sec," sec")
        ephem_start = ' "EPHEM_START='+str(mjd.datetime.year)+' '+str(mjd.datetime.month).zfill(2)+' '+str(mjd.datetime.day).zfill(2)+' '+str(mjd.datetime.hour).zfill(2)+':'+str(mjd.datetime.minute).zfill(2)+'"'    #" 0:00"
        if dmjd<=(1/24):
            stepsize="5m"
            nsteps= int(dmjd/(float(stepsize[:-1])/60/24))
        elif dmjd>(1/24) and dmjd<=1:
            stepsize = "30m"
            nsteps= int(dmjd/(float(stepsize[:-1])/60/24))
        elif dmjd>1.0 and dmjd<=5:
            stepsize = "4h" # [hr] if >5 days btw tracklets, 1 hr btwn ephemeris pts
            nsteps= int(dmjd/(float(stepsize[:-1])/24))
        elif dmj>5.0 and dmjd<=10:
            stepsize = "12h" # [hr]
            nsteps= int(dmjd/(float(stepsize[:-1])/24))
        elif dmj>10.0:
            stepsize = "24h" # [hr]
            nsteps= int(dmjd/(float(stepsize[:-1])/24))
        #nsteps = 20
        #stepsize = (dmjd/nsteps) # seconds
        ephem_step_size = " EPHEM_STEP_SIZE="+str(stepsize)
        ephem_steps = " EPHEM_STEPS="+str(nsteps)
        # Setup Find_Orb command inputs for outfiles
        combine_arg = " " #" -c"    # treat tracklets as same object
        elem_arg = " -g "+fo_outfile_elem
        ephem_arg = " -C "+str(obs_code)+" -E 6,16,17 -e "+fo_outfile_ephem+ephem_start+ephem_steps+ephem_step_size
        cobs_arg = " -F "+fo_outfile_cobs
    # if this is a single tracklet, we only need the orbital elements file.  for now.
    else:
        # Setup Find_Orb command inputs for outfiles
        combine_arg = ""    # no need to combine individual tracklets
        elem_arg = " -g "+fo_outfile_elem
        ephem_arg = ""
        cobs_arg = " -F "+fo_outfile_cobs
    print(fo_infile," (Find_Orb input file) written, running Find_Orb on tracklet(s)")

    # --Write & run Find_Orb command--
    fo_cmd = "fo "+fo_infile+elem_arg+cobs_arg+ephem_arg+combine_arg
    print(fo_cmd)
    fo_info= subprocess.getoutput(fo_cmd)
    print("Find_Orb output = \n",fo_info)
    #fo_elems = read_fo_elem(fo_outfile_elem) # Find_Orb orbital elements
    if os.path.exists(fo_outfile_elem): print("elements file written to ",fo_outfile_elem)
    else: print("elements file not written!")
    t1 = time.time()
    print("time1 = ",t1)
    print("dt = ",t1-10)
