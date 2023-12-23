#!/usr/bash/env python 



# Imports 
#--------
from astropy_healpix import HEALPix
from astropy_healpix.core import boundaries_lonlat
from astropy.table import *
from astropy import units as u
from dl import queryClient as qc
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


# Functions 
#----------

def distance(x, y, x0, y0):
    '''https://stackoverflow.com/questions/67490884/distance-between-a-point-and-a-curve-in-python'''
    d_x = x - x0
    d_y = y - y0
    dis = np.sqrt( d_x**2 + d_y**2 )
    return dis

def get_adj_healpix(boundary,nside):
    '''boundary = array of lon,lat points along healpix boundary'''
    lons = boundary[0]
    lats = boundary[1]
    #print(len(lons),len(lats))
    #mn = np.array([np.mean(lons),np.mean(lats)])
    mn = np.array([lons[int(len(lons)/2)],lats[int(len(lats)/2)]])
    slp = (lats[int(len(lats)/2)+1] - lats[int(len(lats)/2)-1])/(lons[int(len(lons)/2)+1] - lons[int(len(lons)/2)-1]) #slope of boundary at middle
    slpi = -(1/slp) #perpendicular slope
    #shift_amt = np.repeat(np.sqrt(np.rad2deg(hp.nside2resol(nside))/300),2) #degree
    shift = np.sqrt(np.rad2deg(hp.nside2resol(nside))/300) #degree
    ang_perp = np.arctan(slpi)
    print("angle of perpendicular line = ",ang_perp)
    shift_amt = np.array([np.cos(ang_perp)*shift,np.sin(ang_perp)*shift])
    point_to_right = mn + shift_amt
    point_to_left = mn - shift_amt
    #print(mn,point_to_right,point_to_left)
    pix0 = hp.ang2pix(256,point_to_right[0],point_to_right[1],lonlat=True,nest=False)
    pix1 = hp.ang2pix(256,point_to_left[0],point_to_left[1],lonlat=True,nest=False)
    return(pix0,pix1,mn,point_to_right,point_to_left)

def PointsInCircum(r,n,vertx):
    '''https://stackoverflow.com/questions/8487893/generate-all-the-points-on-the-circumference-of-a-circle'''
    circ = [(np.cos(2*np.pi/n*x)*r,np.sin(2*np.pi/n*x)*r) for x in range(0,n+1)]
    return [(j[0]+k[0], j[1]+k[1]) for j,k in zip(vertx,circ)]
    


# Main Code
#----------
if __name__ == "__main__":

    nside = #256 # ring order
    hpix = HEALPix(nside=nside, order='ring')
    ind = #6009 # healpix index of nside & order "hpix"
    lon_hpix,lat_hpix = hpix.healpix_to_lonlat([ind]) # in radians
    print("lon_hpix,lat_hpix = ",lon_hpix[0].degree,lat_hpix[0].degree)
    steps = 100 # number of steps to generate for healpix "hpix" boundaries
    corner = 1
    long,lati = hpix.boundaries_lonlat([ind], step=steps)
    long = long.to(u.deg).value[0]
    lati = lati.to(u.deg).value[0]
    lonc,latc = hpix.boundaries_lonlat([ind], step=corner)
    lonc = lonc.to(u.deg).value[0]
    latc = latc.to(u.deg).value[0]
    print("corners at lonc,latc = ",lonc,latc)
    
    # Method 1: Get all adjoining healpix via vertex search
    circrad = np.sqrt(np.rad2deg(hp.nside2resol(nside))/300) # degree, radius of circle of points around each vertex
    ncirc = 10 # number of points for the circle around each vertex
    for vt in range(0,len(lonc)):
        # For a corner, get a ring of points around that corner, find all 4 healpix touching that vertex
        vertx = [(lonc[vt],latc[vt]) for i in range(0,ncirc)] # define the corner/vertex
        circ = PointsInCircum(circrad,ncirc,vertx) # get a ring of ncirc points around that vertex
        
        adj_pix = np.unique(hp.ang2pix(nside,[i[0] for i in circ],[i[1] for i in circ],nest=False,lonlat=True))
        print("adjacent pix = ",adj_pix)


    # Method 2: For each boundary, get hp to right and left
    #for i in range(4):
    #    #print("corner = ",lonc[i:(i+1)])    
    #    #print(long[steps*i:(steps*(i+1))])
    #    #print(lati[steps*i:(steps*(i+1))])
    #    bnd = [long[steps*i:(steps*(i+1))],lati[steps*i:(steps*(i+1))]]
    #    p0,p1,b_mean,p_to_right,p_to_left = get_adj_healpix(bnd,nside)
    #    #print(p0,p1,b_mean,p_to_right,p_to_left) # healpix to right, healpix to left, border midle, point to right, point to left


        print("ndatapoints = ",len(hpdat))
        
        

