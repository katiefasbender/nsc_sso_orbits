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

def read_cfmpc80(filename):
    '''Read a text file (filename.txt) written in "CANFind" MPC 80-column format,
       which means the first 13 columns are CF-style measurement ids.
    Arguments:
    ----------
    filename (str)
        Name of file to be read, <filename>.txt
    Returns:
    --------
    mmt_cat (astropy table)
        Astropy table with 1 row for each mmt in the file, and the following columns:
        - mmt_id (measurement id) XX.YYY.ZZZZZZ
        - tracklet_id (tracklet id)  YYY.ZZZZZZ where
              X = unique mmt # within tracklet "Y"
              Y = unique tracklet # within HP "Z"
              Z = unique HEALPIX # (NSIDE=64, nested ordering)
        - line (corresponding line without measurement/tracklet ids, for Find_Orb purposes)
        - mjd, ra, dec, mag_auto, filter
    '''
    # Make sure the file exists
    fil=filename
    if os.path.exists(fil) is False:
        print(fil+" NOT found")
        return None
    lines = readlines(fil)
    nmmts = len(lines)-2
    if nmmts == 0:
        print("No measurements in "+file)
        return None
    mmt_cat=Table(names=("mmt_id","tracklet_id","line","mjd","ra","dec","mag_augo","filter"),
                  dtype=["U13","U10","U92","float64","float64","float64","float64","U1"]) # output table (cat)
    for ln in lines[2:]: #don't include the first two header lines
        mmt_id = ln[0:13] # get the measurement(observation) id
        tracklet_id = mmt_id[3:13]
        #tracklet_id_ln = "   "+ln[3:92]
        temp_id_ln = ln[13:] #"            "
        # get string info
        yyyy=ln[15:19]
        mm=ln[20:22]
        dd=ln[23:31]
        ra_hh=ln[32:34]
        ra_mm=ln[35:37]
        ra_ss=ln[38:44]
        dec_sign=ln[44]
        dec_dd=ln[45:47]
        dec_mm=ln[48:50]
        dec_ss=ln[51:56]
        mag=ln[65:69]
        filt=ln[70]
        ra_err=ln[81:86]
        dec_err=ln[87:92]
        # transform strings to values
        ddd = float(dd) - int(float(dd))
        h= int(ddd*24)
        m= int(((ddd*24)-h)*60)
        s= ((((ddd*24)-h)*60)-m)*60
        mjd = Time({'year':int(yyyy),'month':int(mm),'day':int(float(dd)),'hour':h,'minute':m,'second':s},scale='utc')
        mjd = mjd.mjd
        c = SkyCoord(ra_hh+'h'+ra_mm+'m'+ra_ss+'s', dec_sign+dec_dd+'d'+dec_mm+'m'+dec_ss+'s', frame='icrs')
        ra= c.ra.value
        dec=c.dec.value
        mag_val=float(mag)
        # add row to table
        row=[mmt_id,tracklet_id,temp_id_ln,mjd,ra,dec,mag_val,filt]
        mmt_cat.add_row(row)
    return(mmt_cat)

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
    parser.add_argument('--foids',type=str,nargs=1,help="Delimited lislt of tracklets' Find_Orb ID(s)")
    parser.add_argument('--pix32s', type=str, nargs=1, default=1, help='HEALPix value(s) (NSIDE=32) of tracklet(s) location')
    #parser.add_argument('--hgfiles',type=str,nargs=1,default=None,help='Unique hgroup32 filenames with tracklet(s) mmts')
    parser.add_argument('-c','--combine', action='store_true', help='Combine the input tracklets (treat as one object)')
    parser.add_argument('-r','--redo', action='store_true', help='Redo tracklets that were previously processed')
    args = parser.parse_args()

    # Inputs
    tracklets = args.tracklets[0].split(",") # the tracklet id(s) to calculate orbit(s) for
    fo_ids = args.foids[0].split(',')        # the tracklet fo_id(s) to calculate orbit(s) for
    comp = args.comp[0]                      # 0 = individual tracklets, 1 = pairs, 2 = so on
    pix32 = args.pix32s[0].split(',')  # if tracklets in multiple pix32s, will be in fmt "pix32.<tracklet_#>"
    mpc_files = np.unique([int(i.split(".")[-1])//1000 for i in tracklets]) #args.hgfiles[0].split(',')   # files with tracklet mmt MPC lines (hgroup128)
    redo = args.redo
    combine = args.combine
    basedir = "/home/x25h971/orbits_dr2/"
    outdir = basedir+"comp"+comp+"/fgroup_"  # +str(fo_id//10000) added later in code, for fo output
    t0 = time.time()

    ## --get the MPC 80-col lines from the appropriate hgroup text files--
    ##--------------------------------------------------------------------
    hgroups_cat = Table()
    for fil in np.unique(mpc_files):
        ftable = read_cfmpc80("/home/x25h971/canfind_dr2/concats/cf_dr2_hgroup_"+str(fil)+".txt")
        hgroups_cat = vstack([hgroups_cat,ftable])

    ## --for each tracklet (or tracklet combo), get Find_Orb (fo) info--
    ##--------------------------------------------------------------------
    fo_hgroup32 = outdir + str(int(fo_ids[0][1:])//10000)
    makedir(fo_hgroup32)
    fo_filebase = fo_hgroup32+"/fo_comp"+str(comp)+"_"+str(int(fo_ids[0][1:])//10)
    fo_infile = fo_filebase+".txt" # fo input file, change pair_id to fo_id
    if os.path.exists(fo_infile): os.remove(fo_infile)
    fo_file = open(fo_infile,'w')
    fo_outfile_elem = fo_filebase+"_elem.txt"   # fo output file (orbital elements, MPC 8-line fmt)
    fo_outfile_ephem = fo_filebase+"_ephem.txt" # fo output file (ephemeris)
    fo_outfile_cobs = fo_filebase+"_cobs.txt"   # fo output file (calculated observations)

    tracklet_mmts = Table()
    if combine: indiv_orbit_elems=Table()
    for foid,tid in zip(fo_ids,tracklets): # for each tracklet,
        # get mmt MPC80col lines for all tracklets (from hgroups_cat),
        # and write lines to a text file to put into Find_Orb
        t_mmts = hgroups_cat[hgroups_cat['tracklet_id']==tid] # tracklet mmts
        #print(t_mmts)
        #if combine: # if it's a tracklet combination, must get individual tracklet orit information
            #indiv_elem_file = "cfdr2_"+str(id)+"_"+str()+".txt" # file with tracklet information
            #indiv_elems = read_fo_elem(indiv_elem_file)
        lines = t_mmts['line']
        for ln in lines:
            fo_file.write("     "+foid+" "+ln+"\n")
        tracklet_mmts = vstack([tracklet_mmts,t_mmts])
    fo_file.close() # finish writing the fo input file
    print("Find_Orb input file ",fo_infile," written")
    tracklet_mmts.sort('mjd')

    # if this is a tracklet combination, we need to get some info for
    # our output ephemeris, to add to the fo command
    if combine:
        mjd = Time(tracklet_mmts['mjd'][0],format="mjd",scale="utc") #***********MAY BE INCORRECT
        dmjd = tracklet_mmts['mjd'][-1]-tracklet_mmts['mjd'][0]
        ephem_start = " EPHEM_START "+str(mjd.datetime.year)+" "+str(mjd.datetime.month)+" "+str(mjd.datetime.day)+" 0:00"
        if dmjd<=5: stepsize = 1 # [hr] if >5 days btw tracklets, 1 hr btwn ephemeris pts
        elif dmj>5 and dmjd<=50: stepsize = 12 # [hr]
        elif dmj>50: stepsize = 24 # [hr]
        ephem_step_size = " EPHEM_STEP_SIZE "+str(stepsize)
        nsteps= dmjd.value/(stepsize/24)
        ephem_steps = " EPHEM_STEPS "+str(nsteps)
        combine_arg = " -c"    # treat tracklets as same object
        elem_arg = " -g "+fo_outfile_elem
        ephem_arg = " -e "+fo_outfile_ephem+ephem_start+ephem_steps+ephem_step_size
        cobs_arg = " -F "+fo_outfile_cobs
    # if this is a single tracklet, we only need the orbital elements file.  for now.
    else:
        combine_arg = ""    # no need to combine individual tracklets
        elem_arg = " -g "+fo_outfile_elem
        ephem_arg = ""
        cobs_arg = ""
    print(fo_infile," (fo input file) written, running Find_Orb on tracklet(s)")

    # --Write & run Find_Orb command--
    fo_cmd = "fo "+fo_infile+elem_arg+ephem_arg+cobs_arg+combine_arg
    print(fo_cmd)
    fo_info= subprocess.getoutput(fo_cmd)
    print("Find_Orb output = \n",fo_info)
    #fo_elems = read_fo_elem(fo_outfile_elem) # Find_Orb orbital elements
