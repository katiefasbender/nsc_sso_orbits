# AUTHOR: Katie Fasbender
#         katiefasbender@montana.edu

# orbit_concat.py is a script that reads text files in a given subdir (fgroup_<subdir#>) with
# Find_Orb output files, of which there are currently three types:

# elements file _elem.txt
# ephemeris file _ehem.txt

# Output: fits file with astropy table



#---------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------

import astropy.units as u
from astropy.table import Table,Column,join,vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
from dl import queryClient as qc
import healpy as hp
import matplotlib
import numpy as np
from scipy.optimize import curve_fit, least_squares
import subprocess
import sys
import os
from dlnpyutils.utils import *
from orbit_func import *

#---------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------
def makedir(dir):
    '''Makes a directory with name "dir" if it does not exist.
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


#---------------------------------------------------------------------------------
# Main Code
#---------------------------------------------------------------------------------
if __name__ == "__main__":

    # --Initiating & Defining--
    # -------------------------
    # grab inputs
    sdir = str(sys.argv[1])               # fgroup subdir# or path hgroup32 subdir#
    drnum = str(sys.argv[2])              # NSC DR#
    comp = str(sys.argv[3])               # comparison number
    if len(sys.argv)>4: ftype = str(sys.argv[4])              # "elem", "ephem", or "cobs", or "obs"
    else: ftype - "elem"
    # -Define directory structure
    basedir = "/home/x25h971/orbits/"                  # base location of operations
    filedir = basedir+"files/"                         # where scripts and concat catalogs go
    compdir = basedir+"dr"+drnum+"/comp"+str(comp)+"/" # where FO in/output files are (subdir below)
    if int(comp)>0: sudir = "hgroup32_"+sdir+"/"           # to create an output concat catalog if comp1+
    else: sudir = "fgroup_"+sdir+"/"
    subdir = compdir+sudir
    # -Set up output columns
    if int(comp)>0: id_col = "path_id"
    else: id_col = "fo_id"
    if ftype=="elem": # orbital element columns
        orb_cols = ["pix32",id_col,"a","a_err","e","e_err","h",
                    "inc","inc_err","m","m_err","mean_res","n","n_err",
                    "node","node_err","n_obs","n_tot","q","q_err","w","w_err",
                    "sv_x","sv_y","sv_z","sv_vx","sv_vy","sv_vz"]
        orb_dts = ["int","U7","float64","float64","float64","float64","float64",
                   "float64","float64","float64","float64","float64","float64","float64",
                   "float64","float64","int","int","float64","float64","float64","float64",
                   "float64","float64","float64","float64","float64","float64"]
    # -If comp0, set up tables & files to update tracklet cat (cat_tracklet)
    if int(comp)==0:
        cat_tracklet_filename = filedir+"cfdr"+str(drnum)+"_tracklet_cat_orbs.fits.gz" # tracklet (combo) cat
        cat_tracklets = Table.read(cat_tracklet_filename)
        cat_cols = np.array(cat_tracklets.colnames) # existing catalog columns
        # add orbital element columns
        for cl in range(len(orb_cols)):
            if orb_cols[cl] not in cat_cols:
                cat_tracklets[orb_cols[cl]] = Column(np.zeros(len(cat_tracklets)),dtype=orb_dts[cl])
    # -Set up table to receive orbital elements we're about to read in from the subdir
    orbit_info = Table()
    print("concatenating orbit information for subdir: ",subdir)

    # -Loop through fo output files in subdir
    # ------------------------------------------
    fsuffix = "_"+str(ftype)
    #if ftype=="obs": fsuffix=str("")
    g = (subprocess.getoutput("ls "+subdir+str(ftype)+"/*"+fsuffix+".txt")).split("\n")
    #if ftype=="obs": g = [d for d in g if ((d.split("_")[-1]!="elem.txt") & (d.split("_")[-1]!="ephem.txt") & (d.split("_")[-1]!="cobs.txt"))]
    for fn in range(0,len(g)):
        if fn%1000==0: print(fn,"/",len(g))
        fname = g[fn]
        if 1==1:
            if 2==2:
    #for root,dirs,files in os.walk(subdir):
        #for name in files:
            #fname = os.path.join(root,name)
            #if fname[-9:] == "_elem.txt": #os.stat(os.path.join(root,name)).st_size!=0 and len(fits.open(os.path.join(root,name)))>1:
            #if fname.split("_")[-1]=="elem.txt":
                # read elements file, add info to orbit catalog
                if ftype=="elem": dat = read_fo_elem(fname)
                elif ftype=="ephem": dat = read_fo_ephem(fname)
                elif ftype=="cobs": dat = read_mpc80(fname)
                elif ftype=="obs": dat = read_mpc80(fname)
                if len(dat)>0:
                    if int(comp)>0: # get the pix32!
                        #if ftype=="obs": p32,pa_id,suf = (fname.split("_")[-1]).split(".")
                        p32,pa_id = (fname.split("_")[-2]).split(".") #else:
                        dat['pix32'] = Column(np.repeat(p32,len(dat)))
                        dat['path_id'] = Column(np.repeat(pa_id,len(dat)))
                    orbit_info = vstack([orbit_info,dat])
    # -If comp1+, write the hgroup32 subdir orbital elemets to a file
    if int(comp)>0:
        hgroup32_outfile = filedir+"lists/comp"+str(comp)+"/"+sudir+"hgroup32_"+str(sdir)+fsuffix+".fits.gz"
        orbit_info.write(hgroup32_outfile,overwrite=True)
        print(hgroup32_outfile," written")
    # -Else, if comp0, update tracklet cat
    else:
        matches,cat_ind,orbinfo_ind = np.intersect1d(Column(cat_tracklets['fo_id']),Column(orbit_info['fo_id']),return_indices=True)
        for col in orb_cols:
            cat_tracklets[col][cat_ind] = orbit_info[col][orbinfo_ind]
        #cat_tracklets = join(cat_tracklets,orbit_info,keys="fo_id",join_type="left")
        cat_tracklets.write(cat_tracklet_filename,overwrite=True)
        print(cat_tracklet_filename," written")
