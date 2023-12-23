#!/usr/bin/env python

# Imports

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table,Column,vstack
from astropy.time import Time
import astropy.units as u
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


# Functions

# read_cfmpc80()
# read_mpc80()
# read_fo_elem_old()
# read_fo_elem()
# read_fo_ephem()

def read_cfmpc80(filename,specific=False,tlets=[]):
    '''Read a text file (filename.txt) written in "CANFind" MPC 80-column format,
       which means the first 13 columns are CF-style measurement ids.
    Arguments:
    ----------
    filename (str)
        Name of file to be read, <filename>.txt
    specific (bool, default=False)
        if True, only return specified tracklets
    tlets (str list, default=[])
        specified tracklet ids to return, if specific=True
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
    print(len(lines)," = lines")
    if specific:
        lbool = [0,1]+sum([grep(lines,t,index=True) for t in tlets],[])
        lbool = [int(l) for l in lbool]
        lines = Column(lines)[lbool]
        print(len(lines)," = lines")
    nmmts = len(lines)-2
    print("nmmts = ",nmmts)
    if nmmts == 0:
        print("No measurements in "+fil)
        return None
    else:
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

def read_mpc80(filename,specific=False,tlets=[]):
    '''Read a text file (filename.txt) written in MPC 80-column format.
    Arguments:
    ----------
    filename (str)
        Name of file to be read, <filename>.txt
    specific (bool, default=False)
        if True, only return specified tracklets
    tlets (str list, default=[])
        specified tracklet ids to return, if specific=True
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
    #print(len(lines)," = lines")
    if specific:
        lbool = [0,1]+sum([grep(lines,t,index=True) for t in tlets],[])
        lbool = [int(l) for l in lbool]
        lines = Column(lines)[lbool]
        print(len(lines)," = lines")
    nmmts = len(lines)-2
    #print("nmmts = ",nmmts)
    if nmmts == 0:
        print("No measurements in "+fil)
        return None
    else:
        mmt_cat=Table(names=("mmt_id","tracklet_id","line","mjd","ra","dec","mag_augo","filter"),
                      dtype=["U13","U10","U92","float64","float64","float64","float64","U1"]) # output table (cat)
        mmt_lines = np.array([len(l)>=80 for l in lines])
        lines = np.array(lines)[mmt_lines]
        for ln in lines:
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



def read_fo_elem_old(filename,test=False):
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
    object_cat=Table(names=("id","a","a_err","e","e_err","h",
                            "inc","inc_err","m","m_err","mean_res","n","n_err",
                            "node","node_err","n_obs","n_tot","q","q_err","w","w_err",
                            "sv_x","sv_y","sv_z","sv_vx","sv_vy","sv_vz"),
                  dtype=["U7","float64","float64","float64","float64","float64",
                         "float64","float64","float64","float64","float64","float64","float64",
                         "float64","float64","int","int","float64","float64","float64","float64",
                         "float64","float64","float64","float64","float64","float64"]) # output table (cat)
    test=test
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
    
def read_fo_elem(filename):
    '''Read an "elements.txt-type" text file (filename.txt) "written in Find_Orb's edited MPC 8-line format" (idk what that is still).
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
        - m (mean anomaly) 
        - m_err        
        - mean_res (mean residuals) ['']
        - node (ascending node) [deg]
        - node_err
        - n (mean daily motion)
        - n_err
        - num_obs (number of observations)
        - q (perihelion distance) [AU]
        - q_err
        - w (argument of perihelion) [deg]
        - w_err
    '''
    # Make sure the file exists
    fil=filename
    #print("fil = ",fil)
    if os.path.exists(fil) is False:
        print(fil+" NOT found")
        return None
    # Set up output table
    obj_table = Table()
    # Read in the lines
    lines = readlines(fil)
    # Separate file into objects
    o_inds = grep(lines,"Orbital elements:",index=True)
    obj_inds = o_inds+[len(lines)]
    # For each object...
    test=False
    for oi in range(0,len(obj_inds)-1):
        try:
            #print("oi = ",oi)
            olines=lines[obj_inds[oi]:obj_inds[oi+1]] # get object lines
            object_data = []
            hsh = np.array([i[0] for i in olines])    # separate lines into hashed & non-hashed data
            # -Get data from first few non-hashed lines
            nonhash_lines = np.array(olines)[hsh!="#"]
            # --get the orbital elements
            possible_nonhash_data = np.array(["a","e","G","H","Incl.","M","n","Node","P","Peri.","q","Q","U"])
            possible_nonhash_errs = np.array(["a_err","e_err","Incl._err","M_err","n_err","Node_err","Peri._err","q_err"])
            possible_nonhash_dtypes=np.array(["float64","float64","float64","float64","float64","float64","float64","float64","float64","float64","float64","float64","float64"])
            alls = np.array((" ".join(nonhash_lines[3:-1])).split(" ")) # join all the lines together
            alls = alls[(alls!=" ") & (alls!="") & (alls!="(J2000") & (alls!="ecliptic)")] # get rid of useless stuff
            uncert = ("+/-" in alls) # determine whether uncertainties are included
            i,i1,i2 = np.intersect1d(possible_nonhash_data,alls,return_indices=True) # determine which orbital elements are included
            ilow = np.sort(i2) # indices of orbital element names
            d = np.array(list(np.diff(ilow))+[len(alls)+1])
            ihi = ilow+d       # indices to include orbital element values with names
            elems = [list(alls[l:h]) for (l,h) in zip(ilow,ihi)]
            for u in possible_nonhash_errs:
                err_elem = u.split("_")[0]
                if uncert:
                    u_val = ((elems[[i[0] for i in elems]==err_elem]))
                    if test: print("u_val = ",u_val)
                    u_val = u_val[3]
                    object_data+=[[u,str(u_val)]]
                else:
                    object_data+=[[u,"-9999.99"]]
            elems = [i[:2] for i in elems]
            missing_elems = np.array(list(set(possible_nonhash_data)-set(alls[ilow])))
            mi_elems = [list(m) for m in list(np.column_stack((missing_elems,np.repeat(-9999.9,len(missing_elems)))))]
            if test:
                print("uncertainties included = ",uncert)
                print("orbital elements included = ",alls[ilow])
                print("elements missing = ",list(missing_elems))
                print("nonhashed elems = ",elems+mi_elems)
            object_data+=elems
            object_data+=mi_elems
            # --get mean residuals
            resid_str = nonhash_lines[-1].split(" ")[-1]
            resid = int(resid_str.split('"')[0])+float(resid_str.split('"')[-1])
            if test: print("mean residuals = ",resid)
            object_data+=[["mean_residuals",str(resid)]]
            # --get number of observations, number used for calculation
            nobs_splitline = nonhash_lines[-1].split(" ")
            if nobs_splitline[0] == "From":
                ncalc = int(nobs_splitline[1])
                nobs = ncalc
            else:
                ncalc = int(nobs_splitline[0])
                nobs = int(nobs_splitline[2])
            if test: print("nobs,ncalc = ",nobs,ncalc)
            object_data+=[["num_obs",str(nobs)],["num_calc",str(ncalc)]]
            # -Get data from last hashed lines
            hash_lines = np.array(olines)[hsh=="#"]
            # --get the orbital elements
            possible_hashed_data = ["$Name","$Ty","$Tm","$Td","$MA","$ecc","$Eqnx","$a","$Peri","$Node","$Incl","$EpJD","$q","$T","$H"]
            alls_hash = np.array((" ".join(hash_lines[-5:-1])).split(" "))
            alls_hash = alls_hash[(alls_hash!="#") & (alls_hash!=" ") & (alls_hash!="")]
            alls_hash = [i.split("=") for i in alls_hash]
            #alls_hash = np.column_stack((np.array(alls_hash[:,0]),np.array(alls_hash[:,1])))
            if test: print("alls_hash = ",alls_hash)
            object_data+=alls_hash
            # --get sigmas, diameter, and score
            sigmas = int(hash_lines[-1].split(" ")[-1])
            diameter = float(hash_lines[-7].split(" ")[2])
            score = float(hash_lines[-6].split(" ")[-1])
            if test: print("sigmas = ",sigmas,", diameter = ",diameter,", score = ",score)
            object_data+=[["sigmas_avail",str(sigmas)],["d",str(diameter)],["score",str(score)]]
            # -- get state vector
            # state vector (s_x,y,z and s_vx,vy,vz) --------------------------------
            sv_lines = " ".join(hash_lines[[1,2]])
            sv_dat = np.array(sv_lines.split(" "))
            sv_dat = sv_dat[(sv_dat!=" ") & (sv_dat!="#") & (sv_dat!="") & (sv_dat!='AU') & (sv_dat!='mAU/day')]
            if test: print("sv_dat = ",sv_dat)
            if len(sv_dat)<6:
               sv_mags = np.array(sum([j.split("-") for j in sum([i.split("+") for i in sv_dat],[])],[]))
               sv_mags = sv_mags[sv_mags!=""]
               if test: print(sv_mags)
               sign_inds = [sv_lines.index(mg) for mg in sv_mags]
               sv = np.array([sv_lines[sign_inds[k]-1]+sv_mags[k] for k in range(0,6)])
            else: sv = np.array([(i) for i in sv_dat[[0,1,2,3,4,5]]]) #s_x,y,z and s_vx,vy,vz
            if test: print("sv = ",sv)
            object_data+=[["sv_x",sv[0]],["sv_y",sv[1]],["sv_z",sv[2]],["sv_vx",sv[3]],["sv_vy",sv[4]],["sv_vz",sv[5]]]
            possible_dat = list(possible_nonhash_data)+list(possible_nonhash_errs)+possible_hashed_data+["sv_x","sv_y","sv_z","sv_vx","sv_vy","sv_vz","sigmas_avail","d","score","num_obs","num_calc","mean_residuals"]
            #print(object_data)
            object_data = np.array(object_data)
            otab = Table(names=object_data[:,0],dtype=np.repeat("U",len(object_data)))
            otab.add_row(object_data[:,1])
            #odat = []
            #for col in range(0,len(object_data)):
            #    otab[object_data[col][0]] = Column(dtype=possible_dtypes[col])
            #    odat.append(object_data[col][1])
            #otab.add_row(odat)
            obj_table = vstack([obj_table,otab])
        #return(object_data,np.sort(possible_dat))
        except Exception as e:
            print("object misbehaving in file ",fil,"----------------------------")
            #print(olines)
            print("exception = ",e)
    for colname in obj_table.colnames:
        if colname[0]=="$":
            colname_new = "X"+colname[1:]
            obj_table[colname].name = colname_new
    return(obj_table)

def read_fo_ephem(filename):
    '''Read an "ephemeris.txt-type" text file (filename.txt) containing the ephemeris
       of either one tracklet or a combination of them.
       Columns: Date [JD], RA [deg], Dec [deg], delta [AU], r (AU), elong [deg], ph_ang [deg], mag
    Arguments:
    ----------
    filename (str)
        Name of file to be read, <filename>.txt
    Returns:
    --------
    object_table (astropy table)
        One row for each calculated measurement represented in the file, with columns:
        - mjd
        - RA [deg]
        - Dec [deg]
        - delta
        - r
        - elong (solar elongation)
        - ph_ang (phase_angle)
        - mag
    '''
    # Make sure the file exists
    fil=filename
    if os.path.exists(fil) is False:
        print(fil+" NOT found")
        return None
    # Read in data
    lines = readlines(fil)
    #h = np.array([i for i in lines[2]])
    #h_inds = [0]+list(np.where(h==" ")[0])+[len(lines[2])+1]
    #print("file = ",fil)
    ephem_dat = Table(names=['jd','ra','dec','delta','r','elong','ph_ang','mag'],dtype=np.repeat("str",8))
    for l in range(1,len(lines)):
        ln = np.array((lines[l]).split(" "))
        ln = ln[ln!=""]
        #print("ln = ",ln)
        ephem_dat.add_row(ln[:8])
    #    ln = lines[l]
    #    #print("line ",ln)
    #    row = np.array([ln[h_inds[n-1]:h_inds[n]].strip() for n in range(1,len(h_inds))])
    #    if l==1:
    #        gind = row!=""
    #        #print(row[gind])
    #        ephem_dat = Table(names=row[gind],dtype=np.repeat("str",len(row[gind])))
    #    if l>2:
    #        ephem_dat.add_row(np.array(row)[gind])
    # ---------------------------------------------------
    # Time-----------------------------------------------
    #utc_colname = (np.array(ephem_dat.colnames)[np.array([i[:5] for i in ephem_dat.colnames])=="(UTC)"])[0]
    #time_fmt = [i.strip() for i in (utc_colname.split("(UTC)")[-1]).split(":")]
    #y = [int(i) for i in ephem_dat['Date']]     #y = [int(i) for i in ephem_dat['col1']]
    #mo = [int(i.split(" ")[0]) for i in ephem_dat[utc_colname]]     #mo = [int(i) for i in ephem_dat['col2']]
    #d = [int(i.split(" ")[1]) for i in ephem_dat[utc_colname]]     #d = [int(i) for i in ephem_dat['col3']]
    #if "HH" in time_fmt: h = [int((i.split(" ")[2]).split(":")[0]) for i in ephem_dat[utc_colname]]     #h = [int(i.split(":")[0]) for i in ephem_dat['col4']]
    #else: h = [int(i) for i in np.zeros(len(ephem_dat))]
    #if "MM" in time_fmt: m = [int((i.split(" ")[2]).split(":")[1]) for i in ephem_dat[utc_colname]]     #m = [int(i.split(":")[1]) for i in ephem_dat['col4']]
    #else: m = [int(i) for i in np.zeros(len(ephem_dat))]
    #ephem_time = Time({'year': y, 'month': mo, 'day': d,'hour': h, 'minute': m},scale='utc')
    ephem_time = Time([float(i) for i in ephem_dat['jd']],scale='utc',format="jd")
    mjds = ephem_time.mjd
    # ---------------------------------------------------
    # Coordinates----------------------------------------
    #ra_split = [i.split(" ") for i in ephem_dat['RA']]
    #dec_split = [i.split(" ") for i in ephem_dat['Dec']]
    #ra_hh = [str(i.split(" ")[0]) for i in ephem_dat['RA']]     #ra_hh = [str(i) for i in ephem_dat['col5']]
    #ra_mm = [str(i.split(" ")[1]) for i in ephem_dat['RA']]     #ra_mm = [str(i) for i in ephem_dat['col6']]
    #ra_ss = [str(i.split(" ")[2]) for i in ephem_dat['RA']]     #ra_ss = [str(i) for i in ephem_dat['col7']]
    #ra = [ra_hh[i]+'h'+ra_mm[i]+'m'+ra_ss[i]+'s' for i in range(0,len(ephem_dat))]
    ra =  ephem_dat['ra'].astype("float")
    #dec_dd = [str(i.split(" ")[0]) for i in ephem_dat['Dec']]     #dec_dd = [str(i) for i in ephem_dat['col8']]
    #dec_mm = [str(i.split(" ")[1]) for i in ephem_dat['Dec']]     #dec_mm = [str(i) for i in ephem_dat['col9']]
    #dec_ss = [str(i.split(" ")[2]) for i in ephem_dat['Dec']]     #dec_ss = [str(i) for i in ephem_dat['col10']]
    #dec = [dec_dd[i]+'d'+dec_mm[i]+'m'+dec_ss[i]+'s' for i in range(0,len(ephem_dat))]
    dec =  ephem_dat['dec'].astype("float")
    #ephem_coords = SkyCoord(ra,dec,frame='icrs',unit="deg")
    # ---------------------------------------------------
    # magV
    mags = []
    for i in range(0,len(ephem_dat)):
        if "?" in ((ephem_dat['mag'][i])) or (not (ephem_dat['mag'][i]).replace(".", "").isnumeric()): mags.append(-99.99)
        else: mags.append(float(ephem_dat['mag'][i]))
    # ---------------------------------------------------
    # Create output table
    ephem_table = Table()
    ephem_table['ra'] = Column(ra) #ephem_coords.ra
    ephem_table['dec'] = Column(dec) #ephem_coords.dec
    ephem_table['mjd'] = Column(mjds)
    ephem_table['magV'] =  Column(mags)
    ephem_table['delta'] = ephem_dat['delta'].astype("float")
    ephem_table['r'] = ephem_dat['r'].astype("float")
    ephem_table['elong'] = ephem_dat['elong'].astype("float")
    ephem_table['ph_ang'] = ephem_dat['ph_ang'].astype("float")
    return(ephem_table)
