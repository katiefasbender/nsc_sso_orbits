from astropy.table import *
import numpy as np
import subprocess
import sys

if __name__=="__main__":

    basedir = "/home/x25h971/orbits/files/lists/comp1/hgroup32_"
    odir = sys.argv[1]
    outdir = basedir+str(odir)+"/"

    t = Table()
    pixfiles = subprocess.getoutput("ls "+outdir).split("\n")
    print("concatenating ",len(pixfiles)," files in ",outdir)

    for f in pixfiles:
        tt = Table.read(outdir+f)
        t = vstack([t,tt])
    ofile = outdir+"hgroup32_"+str(odir)+"_comp1_list.fits.gz"
    t.write(ofile)
    print(ofile," written")
