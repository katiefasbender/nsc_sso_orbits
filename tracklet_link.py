#! usr/bin/env python

# you (YOU!) should know:
# -----------------------
# 'approaching' or 'app' = near in time
# 'adjacent' or 'adj' = near in space
# 'measurement' or 'mmt' = a detection of...thing in sky
# 'tracklet' or 'tlet' = 3+ mmts of a moving object over one night
# 'track' = multiple tracklets, up to 5 nights
# 'ephemeris' = 10+ days?

# AUTHOR: Katie Fasbender
#         katiefasbender@montana.edu

# tracklet_link.py will, for a given healpix:
#    a. generate a list of exposures that satisfy the following conditions-
#            i. measured within X days of each other
#           ii. measured within Y degrees of each other
#    b. g

    # Get exposures taken within X days before or after the tracklet being tested ('approaching' exposures)

#with mmts (which have been designated


#------------------------------
# Imports
#------------------------------
from argparse import ArgumentParser
from astropy.table import Column,Table,join,vstack
from itertools import combinations
import numpy as np
import os
import subprocess

#------------------------------
# Functions
#------------------------------


#------------------------------
# Main Code
#------------------------------
if __name__=="__main__":


    # Setup
    #------
    # Initiate input arguments
    parser = ArgumentParser(description='Link tracklets detected in the NOIRLab Source Catalog')
    parser.add_argument('--hp32',type=str,nargs=1,help='Healpix (NSIDE32)')
    parser.add_argument('--hp128',type=str,nargs=1,help='comma-delimited list of Healpix (NSIDE128)')
    parser.add_argument('--dr',type=int,nargs=1,help='NSC data release')
    args = parser.parse_args()

    # Inputs
    pix32 = int(args.hp32[0])                  # HEALPix (NSIDE=32)
    pix128s = [int(i) for i in args.hp128[0].split(",")]
    hgroups128 = np.unique(pix128s//1000)
    dr = int(args.dr[0])
    basedir = "/home/x25h971/orbits/dr"+str(dr)+"/"
    cfdir = "/home/x25h971/canfind_dr2/concats/"
    fbase = cfdir+"cf_dr"+str(dr)+"_hgroup_" # the base name for tracklet_concat files 
    outdir = basedir+"comp1/lists/"

    # for the pix32, make a list of tracklet pairs to fit to orbits

    # --get all mmts & tracklets for this pix32--
    mmts_32 = Table()        # a table for the pix32 tracklet measurements
    tlets_32 = Table()       # a table for the pix32 tracklets
    for hg in hgroups128:    #for each pix128 in this pix32, get the mmts and tracklets 
        tlets_hgroup128=Table.read(fbase+str(hg)+"_tracklets.fits") #read in the tracklet & mmt concat files 
        mmts_hgroup128=Table.read(fbase+str(hg)+"_mmts.fits")
        # get the mmts/tracklets with the correct hpix
        # mmts
        mbools = [(mmts_hgroup128['pix128']==pix128s[i]) for i in range(len(pix128s))]
        mpx = np.repeat(False,len(mmts_hgroup128))
        for mb in range(len(mbools)):
            mpx = np.logical_or(mpx,mbools[mb])
        # tlets
        tbools = [(tlets_hgroup128['pix128']==pix128s[i]) for i in range(len(pix128s))]
        tpx = np.repeat(False,len(tlets_hgroup128))
        for tb in range(len(tbools)):
            tpx = np.logical_or(tpx,tbools[tb])
        mmts_32=vstack([mmts_32,mmts_hgroup128[mpx]])
        tlets_32=vstack([tlets_32,tlets_hgroup128[tpx]])
    mmts_32.sort('mjd')
    tlets_32.sort('mjd')

    # --make list of all tracklet pair combinations--
    print("looking at healpix (NSIDE=32) ",pix32)
    tpid = 0 #a unique number for each testable tracklet pair in the pix32 (pair_id)
    tracklet_pairs = Table(names=("tracklet_0","tracklet_1","hgroup_filename0","hgroup_filename1","pix32","pair_id"),
                           dtype=["U10","U10","U100","U100","float64","U100"]) # a table to hold the tracklet pairs list 
    if len(tlets_32)>1:
        tlets_32['night'] = tlets_32['mjd']+0.5            # offset obs_time by 1/2 day to get in terms of nights
        mmts_32['night'] = mmts_32['mjd']+0.5
        tspan_full = (mmts_32['night'][-1]-mmts_32['night'][0]) # full observation timespan for pix32
        exposures = np.unique(mmts_32['exposure'])  # all unique exposures in pix32 with tracklet mmts
        ematches,i1,i2 = np.intersect1d(exposures,mmts_32['exposure'],return_indices=True)
        exposure_nights = np.floor(mmts_32['night'][i2]) # integer mjd night of each exposure
        mmt_nights = np.floor(mmts_32['night']) # integer mjd night of each mmt
        tlet_nights = np.floor(tlets_32['night'])    # first night of observation for each tracklet
        nnights = len(np.unique(mmt_nights))   # number of unique nights of observation for pix32
        # If more than one unique night of observation....................................................................
        if nnights>1:
            diffs = np.array(np.diff(np.unique(mmt_nights))) # number of days between each night of obs
            close_nights = np.where(diffs<=30)[0]            # only nights separated by 30 days or less
            #print("mjds = ",list(ms['mjd']))
            #print("nights = ",list(nights))
            # If there are nights of observation separated by 30 days or less.............................................
            if len(close_nights)>0:
                nightstarts = list(np.unique(mmt_nights)[close_nights])
                nightends = list(np.unique(mmt_nights)[close_nights+1])
                print("diffs = ",list(diffs))
                print("first night in each range =",nightstarts)
                print("last night in range = ",nightends)
                # For each of the <30 day ranges, check all tracklet pairs!...............................................
                for t in range(len(nightstarts)):
                    low = nightstarts[t]
                    high = nightends[t]
                    tlet_mmts = ms[(mmt_nights<=high) & (mmt_nights>=low)]
                    tlets = ts[(tlet_nights<=high) & (tlet_nights>=low)]
                    #print("mmt mjds = ",tlet_mmts['tracklet_id','mjd'])
                    print(len(tlets)," tracklets in ",low," - ",high," range = ",list(tlets['tracklet_id']))
                    tlist=list(combinations(tlets['tracklet_id'],2))
                    print(len(tlist)," pair-combinations of ",len(tlets)," tracklets")
                    # Loop through pairs and prune list...................................................................
                    for tpair in tlist:
                        # get the tracklets and their mmts
                        pair_mmts = tlet_mmts[(tlet_mmts['tracklet_id']==tpair[0]) | (tlet_mmts['tracklet_id']==tpair[1])]
                        pair_tlets = tlets[(tlets['tracklet_id']==tpair[0]) | (tlets['tracklet_id']==tpair[1])]
                        # --each mmt of both tracklets must have a unique date
                        p1 = len(pair_mmts)==len(np.unique(pair_mmts['mjd']))
                        # --the first mmt date of the tracklet that was detected second must be >= the last mmt date of the first-detected tracklet
                        p2 = pair_tlets['mjd'][0]+pair_tlets['dmjd'][0] < pair_tlets['mjd'][1] # --(mjd_2>=mjd_1+dmjd+1)
                        if p1 and p2:  #keep only the tracklets that pass both conditions
                            #print("both conditions true")
                            hp_subdirs = [int(i)//1000 for i in pair_tlets['pix128']
                            fname0 = "cf_dr2_hgroup_"+str(hp_subdirs[0])+".txt"
                            fname1 = "cf_dr2_hgroup_"+str(hp_subdirs[1])+".txt"
                            tracklet_pairs.add_row([pair_tlets['tracklet_id'][0],pair_tlets['tracklet_id'][1],fname0,fname1,pix32,str(tpid)])
                            tpid+=1
                print("pix32 = ",i,", ",nnights," unique nights, ",len(tlets_32)," tracklets, ",
                      len(np.unique(mmts_32['exposure']))," unique exposures, ",tspan_full," day timespan")
                print(len(tpairs)," tracklet pairs in pix32 ",i," \n ")
                tlist_name = outdir+"hgroup_"+str(pix32//1000)+"/cfdr2_"+str(pix32)+"_tpairs.fits"
                tracklet_pairs.write(tlist_name,overwrite=True)
                print("tracklet pair list "+tlist_name+" written for pix32 = ",str(pix32))
            else: print("too much time between nights of observation")
        else: print("not enough unique nights of observation")
    else: print("1> tracklets, not enough to compare in this healpix")