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

# path_link.py will, for a given healpix:
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
    parser = ArgumentParser(description='Link paths detected in the NOIRLab Source Catalog')
    parser.add_argument('--nside',type=str,nargs=1,help='NSIDE of healpix to make lists for')
    parser.add_argument('--hp',type=str,nargs=1,help='Healpix')
    parser.add_argument('--hp128',type=str,nargs=1,help='comma-delimited list of Healpix (NSIDE128)')
    parser.add_argument('--comp',type=int,nargs=1,help='comparison iteration')
    args = parser.parse_args()

    # Inputs
    nside = int(args.nside[0])                 # NSIDE of healpix to make list for
    pix = int(args.hp[0])                  # HEALPix (NSIDE = above)
    comp = args.comp[0]
    pix128s = [int(i) for i in args.hp128[0].split(",")]
    hgroups128 = np.unique([int(int(i)//1000) for i in np.unique(pix128s)])
    basedir = "/home/x25h971/orbits/"
    cfdir = "/home/x25h971/canfind_dr2/concats/"
    fbase = cfdir+"cf_dr2_hgroup_" # the base name for tracklet_concat files
    outdir = basedir+"files/lists/comp"+str(comp)+"/"
    print("hgroups = ",hgroups128)

    # --get all mmts & tracklets for this pix--
    mmts = Table()        # a table for the pix tracklet measurements
    tlets = Table()       # a table for the pix tracklets
    for hg in hgroups128:    #for each pix128 in this pix32, get the mmts and tracklets
        tlets_hgroup128=Table.read(fbase+str(hg)+"_tracklets.fits") #read in the tracklet & mmt concat files
        mmts_hgroup128=Table.read(fbase+str(hg)+"_mmts.fits")
        # get the mmts/tracklets with the correct hpix
        # mmts
        ##mbools = [(mmts_hgroup128['pix128']==pix128s[i]) for i in range(len(pix128s))]
        ##mpx = np.repeat(False,len(mmts_hgroup128))
        ##for mb in range(len(mbools)):
        ##    mpx = np.logical_or(mpx,mbools[mb])
        # tlets
        ##tbools = [(tlets_hgroup128['pix128']==pix128s[i]) for i in range(len(pix128s))]
        ##tpx = np.repeat(False,len(tlets_hgroup128))
        ##for tb in range(len(tbools)):
        ##    tpx = np.logical_or(tpx,tbools[tb])
        mpx = mmts_hgroup128['pix32']==float(pix)
        tpx = tlets_hgroup128['pix32']==float(pix)
        mmts = vstack([mmts,mmts_hgroup128[mpx]])
        tlets = vstack([tlets,tlets_hgroup128[tpx]])
    mmts.sort('mjd')
    tlets.sort('mjd')
    print(len(tlets)," tracklets")

    # --make list of all tracklet pair combinations--
    print("looking at healpix (NSIDE="+str(nside)+") ",pix)
    tpid = 0 #a unique number for each testable tracklet pair in the pix32 (pair_id)
    # a table to hold the tracklet pairs list vvvvv
    tracklet_pairs = Table(names=("tracklet_0","tracklet_1","foid_0","foid_1","pix"+str(nside),"path_id"),
                           dtype=["U10","U10","U100","U100","float64","U100"])
    #tracklet_pairs = Table(names=("tracklet_0","tracklet_1","hgroup_filename0","hgroup_filename1","pix"+str(nside),"path_id"),
    #                       dtype=["U10","U10","U100","U100","float64","U100"])
    if len(tlets)>1:
        tlets['night'] = tlets['mjd']+0.5            # offset obs_time by 1/2 day to get in terms of nights
        mmts['night'] = mmts['mjd']+0.5
        tspan_full = (mmts['night'][-1]-mmts['night'][0]) # full observation timespan for pix32
        exposures = np.unique(mmts['exposure'])  # all unique exposures in pix32 with tracklet mmts
        #print(exposures)
        ematches,i1,i2 = np.intersect1d(Column(exposures),Column(mmts['exposure']),return_indices=True)
        exposure_nights = np.floor(mmts['night'][i2]) # integer mjd night of each exposure
        mmt_nights = np.floor(mmts['night']) # integer mjd night of each mmt
        tlet_nights = np.floor(tlets['night'])    # first night of observation for each tracklet
        nnights = len(np.unique(mmt_nights))   # number of unique nights of observation for pix32
        # If more than one unique night of observation....................................................................
        if nnights>1:
            diffs = np.array(np.diff(np.unique(mmt_nights))) # number of days between each night of obs
            close_nights = np.where(diffs<=15)[0]            # only nights separated by 15 days or less
            #print("mjds = ",list(ms['mjd']))
            #print("nights = ",list(nights))
            # If there are nights of observation separated by 30 days or less.............................................
            if len(close_nights)>0:
                nightstarts = list(np.unique(mmt_nights)[close_nights])
                nightends = list(np.unique(mmt_nights)[close_nights+1])
                #print("diffs = ",list(diffs))
                #print("first night in each range =",nightstarts)
                #print("last night in range = ",nightends)
                # For each of the <30 day ranges, check all tracklet pairs!...............................................
                for t in range(len(nightstarts)):
                    low = nightstarts[t]
                    high = nightends[t]
                    rtlet_mmts = mmts[(mmt_nights<=high) & (mmt_nights>=low)]
                    rtlets = tlets[(tlet_nights<=high) & (tlet_nights>=low)]
                    #print("mmt mjds = ",tlet_mmts['tracklet_id','mjd'])
                    print(len(rtlets)," tracklets in ",low," - ",high," range") # = ",list(tlets['tracklet_id']))
                    tlist=list(combinations(rtlets['tracklet_id'],2))
                    print(len(tlist)," pair-combinations of ",len(rtlets)," tracklets")
                    # Loop through pairs and prune list...................................................................
                    for tpair in tlist:
                        # get the tracklets and their mmts
                        pair_mmts = rtlet_mmts[(rtlet_mmts['tracklet_id']==tpair[0]) | (rtlet_mmts['tracklet_id']==tpair[1])]
                        pair_tlets = rtlets[(rtlets['tracklet_id']==tpair[0]) | (rtlets['tracklet_id']==tpair[1])]
                        # --each mmt of both tracklets must have a unique date
                        p1 = len(pair_mmts)==len(np.unique(pair_mmts['mjd']))
                        # --the first mmt date of the tracklet that was detected second must be >= the last mmt date of the first-detected tracklet
                        p2 = pair_tlets['mjd'][0]+pair_tlets['dmjd'][0] < pair_tlets['mjd'][1] # --(mjd_2>=mjd_1+dmjd+1)
                        if p1 and p2:  #keep only the tracklets that pass both conditions
                            #print("adding tracklet pair to list")
                            #hp_subdirs = [int(int(i)//1000) for i in pair_tlets['pix128']]
                            #fname0 = "cf_dr2_hgroup_"+str(hp_subdirs[0])+".txt"
                            #fname1 = "cf_dr2_hgroup_"+str(hp_subdirs[1])+".txt"
                            #tracklet_pairs.add_row([pair_tlets['tracklet_id'][0],pair_tlets['tracklet_id'][1],
                            #                        fname0,fname1,pix,str(tpid)])
                            tracklet_pairs.add_row([pair_tlets['tracklet_id'][0],pair_tlets['tracklet_id'][1],
                                                    pari_tlets['fo_id'][0],pair_tlets['fo_id'][1],pix,str(tpid)])
                            tpid+=1
                print("pix = ",pix,", ",nnights," unique nights, ",len(tlets)," tracklets, ",
                      len(np.unique(mmts['exposure']))," unique exposures, ",tspan_full," day timespan")
                print(len(tracklet_pairs)," tracklet pairs in pix ",pix," \n ")
            else: print("too much time between nights of observation")
        else: print("not enough unique nights of observation")
    else: print("1> tracklets, not enough to compare in this healpix")
    tlist_name = outdir+"hgroup"+str(nside)+"_"+str(pix//1000)+"/"+str(pix)+".fits"
    tracklet_pairs.write(tlist_name,overwrite=True)
    print("tracklet pair list "+tlist_name+" written for pix = ",str(pix))
