#!/usr/bin/env python

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
import sys
import os
from dlnpyutils.utils import *

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

def read_fo_elem(filename):
    '''Read an "elements.txt-type" text file (filename.txt) written in Find_Orb's edited MPC 8-line format.
       This type of file contains orbital elements calculated with Find_Orb.  
    Arguments:
    ----------
    filename (str)
        Name of file to be read, <filename>.txt
    Returns:
    --------
    object_table (astropy table)
        One row for each object represented in the file, with columns:
        - object_id
        - a (semimajor axis) [AU]
        - a_err
        - e (eccentricity)
        - e_err
        - h (absolute magnitude)
        - inc (inclination) [deg]
        - inc_err
        - m (mean anomaly) [deg]
        - m_err
        - mean_res (mean residuals) ['']
        - node (ascending node) [deg]
        - node_err
        - n (mean daily motion)
        - n_err
        - n_obs (number of observations used to calc orbit)
        - n_tot (total number of tracklet observations)
        - q (perihelion distance) [AU]
        - q_err
        - w (argument of perihelion) [deg]
        - w_err
        - sv_x,y,z (state vector,position) [AU]
        - sv_vx,vy,vz (state vector, velocity) [mAU/day]
    '''
    # Make sure the file exists
    fil=filename
    if os.path.exists(fil) is False:
        print(fil+" NOT found")
        return None
    lines = readlines(fil)

    object_cat=Table(names=("fo_id","a","a_err","e","e_err","h",
                            "inc","inc_err","m","m_err","mean_res","n","n_err",
                            "node","node_err","n_obs","n_tot","q","q_err","w","w_err",
                            "sv_x","sv_y","sv_z","sv_vx","sv_vy","sv_vz"),
                  dtype=["U7","float64","float64","float64","float64","float64",
                         "float64","float64","float64","float64","float64","float64","float64",
                         "float64","float64","int","int","float64","float64","float64","float64",
                         "float64","float64","float64","float64","float64","float64"]) # output table (cat)
    test=False
    # for each object...
    obj_inds = grep(lines,"Orbital elements:",index=True)+[-1]
    for oi in range(0,len(obj_inds)-1):
        # --------------------
        # get orbital elements
        # --------------------
        # figure out if uncertainties were reported (only if all mmts used)---------
        olines=lines[obj_inds[oi]:obj_inds[oi+1]]
        #print("filename = ",filename, olines[0])
        if grep(olines,"observations",index=True)[0]==9: uncert=True
        else: uncert=False
        # object name --------------------------------------------------------------
        name_ind = grep(olines,"Name=",index=True)[0]+obj_inds[oi]
        name_str = lines[name_ind].split("$")[1].split("Name=")[-1]
        if test: print("--------Object ",name_str,"------------")
        try:
            # semimajor axis (a) and ascending node (node) -------------------------
            #node_ind = grep(olines," Node ",index=True)[0]+obj_inds[oi]
            node_ind = 5
            semimajor_strs = np.array(olines[node_ind].split("    "))[0].split("+/-")
            semimajor = float(semimajor_strs[0].split("a")[-1])
            if uncert: semimajor_err = float(semimajor_strs[1])
            else: semimajor_err = -99.99
            if test: print("uncertainty = ",uncert)
            node_strs = np.array(olines[node_ind].split("    "))[-1].split("+/-")
            node = float(node_strs[0].split("Node")[-1])
            if uncert: node_err = float(node_strs[1])
            else: node_err = -99.99
            if test: print("node: ",node,node_err," semimajor: ",semimajor,semimajor_err)
            # eccentricity (e) and inclination (inc) -------------------------------
            #inc_ind = grep(olines,"Incl.",index=True)[0]+obj_inds[oi]
            inc_ind = 6
            ecc_strs = np.array(olines[inc_ind].split("    "))[0].split("+/-")
            ecc = float(ecc_strs[0].split("e")[-1])
            if uncert: ecc_err = float(ecc_strs[1])
            else: ecc_err = -99.99
            inc_strs = np.array(olines[inc_ind].split("    "))[-1].split("+/-")
            inc = inc_strs[0].split("Incl")[-1].split("(J2000 ecliptic)")[0]
            if inc[0]==".": inc = float(inc[1:])
            else: inc = float(inc)
            if uncert: inc_err = float(inc_strs[1])
            else: inc_err = -99.99
            if test: print("eccentricity: ",ecc,ecc_err," inclination: ",inc,inc_err)
            # mean anomaly (m) -----------------------------------------------------
            #m_ind = grep(olines,"ecliptic",index=True)[0]+obj_inds[oi]
            m_ind = 3
            m_strs = np.array(olines[m_ind].split("    "))[0].split("+/-")
            m = float(m_strs[0].split("M")[-1])
            if uncert: m_err = float(m_strs[1])
            else: m_err = -99.99
            if test: print("mean anomaly: ",m,m_err)
            # absolute magnitude (h) -----------------------------------------------
            mag_ind = grep(olines,"H=",index=True)[0]+obj_inds[oi]
            mag_str = lines[mag_ind].split("$")[-1].split("H=")[-1]
            mag = float(mag_str)
            if test: print("magnitude: ",mag)
            # argument of perihelion (w) and mean daily motion (n) -----------------
            #aperi_ind = grep(olines,"Peri. ",index=True)[0]+obj_inds[oi]
            aperi_ind = 4
            n_strs = np.array(olines[aperi_ind].split("    "))[0].split("+/-")
            n = float(n_strs[0].split("n")[-1])
            if uncert: n_err = float(n_strs[1])
            else: n_err = -99.99
            aperi_strs = np.array(olines[aperi_ind].split("    "))[-1].split("+/-")
            aperi = aperi_strs[0].split("Peri")[-1]
            if aperi[0]==".": aperi = float(aperi[1:])
            else: aperi = float(aperi)
            if uncert: aperi_err = float(aperi_strs[1])
            else: aperi_err = -99.99
            if test: print("arg of peri: ",aperi,aperi_err)
            # perihelion distance (q) ----------------------------------------------
            dperi_ind = grep(olines,"q",index=True)[0]+obj_inds[oi]
            dperi_strs = np.array(lines[dperi_ind].split("+/-"))
            if uncert:
                dperi_strs = dperi_strs[[0,1]]
                dperi = float(dperi_strs[0].split("q")[-1])
                dperi_err = float(dperi_strs[1].split('Q')[0])
            else:
                dperi_strs = [dperi_strs[0].split("q")[1]]
                dperi = float(dperi_strs[0].split("Q")[0])
                dperi_err = -99.99
            if test: print("dperi = ",dperi, dperi_err)
            # total # observations used (num_obs) and total # obs (n_tot) ----------
            nobs_ind = grep(olines,"observations",index=True)[0]
            #if uncert:
            if olines[nobs_ind][0:4]=="From":
                nobs_str = (olines[nobs_ind].split("From")[-1]).split("observations")[0]
                if test: print("nobs_str = ",nobs_str)
                nobs = int(nobs_str)
                ntot = nobs
            else:
                nobs_strs = olines[nobs_ind].split("of") #[[0,-1]]
                nobs = int(nobs_strs[0])
                if test: print("nobs_str = ",nobs)
                ntot = int(nobs_strs[1].split("observations")[0])
            if test: print("perihelion: ",dperi,dperi_err," number obs: ",nobs,ntot)
            # mean residuals (mean_res) --------------------------------------------
            r_ind = grep(olines,"residual",index=True)[0]+obj_inds[oi]
            resid_str = lines[r_ind].split("residual")[-1]
            resid = int(resid_str.split('"')[0])+float(resid_str.split('"')[-1])
            if test: print("mean residuals: ",resid)
            # state vector (s_x,y,z and s_vx,vy,vz) --------------------------------
            sv_ind = grep(olines,"State vector",index=True)[0]+obj_inds[oi]
            svp_line = lines[(sv_ind+1)]
            svv_line = lines[(sv_ind+2)]
            svp_inds = [index for (index,ob) in enumerate(svp_line) if ob=="-" or ob=="+"]
            svv_inds = [index for (index,ob) in enumerate(svv_line) if ob=="-" or ob=="+"]
            svp_strs = [svp_line[i:j] for i,j in zip(svp_inds,svp_inds[1:]+[-7])]
            svv_strs = [svv_line[i:j] for i,j in zip(svv_inds,svv_inds[1:]+[-7])]
            svp = [float(st) for st in svp_strs if st!=""]
            svv = [float(st) for st in svv_strs if st!=""]
            if test: print("state vector: ",svp,svv)
            # --------------------------
            # add row to object table...
            # --------------------------
            row=[name_str.strip(),semimajor,semimajor_err,ecc,ecc_err,mag,
                 inc,inc_err,m,m_err,resid,n,n_err,node,node_err,nobs,ntot,
                 dperi,dperi_err,aperi,aperi_err]+svp+svv
            object_cat.add_row(row)
        except Exception as e:
            print("object ",name_str," misbehaving in file ",filename,"----------------------------")
            #print(olines)
            print("exception = ",e)
    return(object_cat)

#---------------------------------------------------------------------------------
# Main Code
#---------------------------------------------------------------------------------
if __name__ == "__main__":

    # --Initiating & Defining--
    # -------------------------
    # grab inputs
    sdir = str(sys.argv[1])                           # fgroup subdir#
    drnum = str(sys.argv[2])                          # NSC DR#
    comp = str(sys.argv[3])                           # comparison number
    # define directory structure
    basedir = "/home/x25h971/orbits_dr"+drnum+"/"     # base location of operations
    filedir = basedir+"files/"                        # where scripts and concat catalogs go
    compdir = basedir+"comp"+str(comp)+"/"            # where FO in/output files are
    subdir = compdir+"fgroup_"+sdir+"/"
    # set up tables & files
    cat_tracklet_filename = filedir+"cfdr"+str(drnum)+"_tracklet_cat_orbs.fits.gz" # tracklet (combo) cat
    cat_tracklets = Table.read(cat_tracklet_filename)
    cat_cols = np.array(cat_tracklets.colnames) # existing catalog columns
    # add necessary columns to tracklet (combo) cat, if not already present ------------------------------------- NECESSARY? 
    orb_cols = ["fo_id","a","a_err","e","e_err","h",
                "inc","inc_err","m","m_err","mean_res","n","n_err",
                "node","node_err","n_obs","n_tot","q","q_err","w","w_err",
                "sv_x","sv_y","sv_z","sv_vx","sv_vy","sv_vz"]
    orb_dts = ["U7","float64","float64","float64","float64","float64",
               "float64","float64","float64","float64","float64","float64","float64",
               "float64","float64","int","int","float64","float64","float64","float64",
               "float64","float64","float64","float64","float64","float64"]
    for cl in range(len(orb_cols)):
        if orb_cols[cl] not in cat_cols:
            cat_tracklets[orb_cols[cl]] = Column(np.zeros(len(cat_tracklets)),dtype=orb_dts[cl])

    orbit_info = Table()
    print("concatenating orbit information for subdir: ",subdir)

    # --Loop through fo output files in subdir--
    # ------------------------------------------
    for root,dirs,files in os.walk(subdir):
        for name in files:
            fname = os.path.join(root,name)
            if fname[-9:] == "_elem.txt": #os.stat(os.path.join(root,name)).st_size!=0 and len(fits.open(os.path.join(root,name)))>1:
                # read elements file, add info to orbit catalog
                dat = read_fo_elem(fname)
                if len(dat)>0:
                    orbit_info = vstack([orbit_info,dat])
    matches,cat_ind,orbinfo_ind = np.intersect1d(Column(cat_tracklets['fo_id']),Column(orbit_info['fo_id']),return_indices=True)
    for col in orb_cols:
        cat_tracklets[col][cat_ind] = orbit_info[col][orbinfo_ind]
    #cat_tracklets = join(cat_tracklets,orbit_info,keys="fo_id",join_type="left")
    cat_tracklets.write(cat_tracklet_filename,overwrite=True)
    print(cat_tracklet_filename," written")


