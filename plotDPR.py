#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:17:59 2021

@author: mgrecu
"""

fname='orbits/2A.GPM.DPR.V9-20210709.20180701-S020801-E034035.024649.ITE760.HDF5'
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import numpy as np
fh=Dataset(fname)
fCMB=Dataset('orbits/2B.GPM.DPRGMI.CORRA2018.20180701-S020801-E034035.024649.V06A.HDF5')

wc=fh['FS/SLV/precipWater'][4820:4950,:,:]
pwc_cmb=fCMB['MS/precipTotWaterCont'][4820:4950,:,:]
bzd=fh['FS/VER/binZeroDeg'][4820:4950,:]
bbpeak=fh['FS/CSF/binBBPeak'][4820:4950,:]
zm=fh['FS/PRE/zFactorMeasured'][4820:4950,:,:,0]
zmka=fh['FS/PRE/zFactorMeasured'][4820:4950,:,:,1]
bbpeak=np.ma.array(bbpeak,mask=bbpeak<=0)
bcf=fh['FS/PRE/binClutterFreeBottom'][4820:4950,:]
import matplotlib
nj=26
plt.close('all')
plt.pcolormesh(wc[:,nj,:].T,norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.plot(bbpeak[:,nj])
plt.plot(bzd[:,nj])
plt.plot(bcf[:,nj])
plt.ylim(176,80)
plt.colorbar()
plt.figure()
plt.pcolormesh(zm[:,nj,:].T,cmap='jet',vmin=10,vmax=50)
plt.plot(bbpeak[:,nj])
plt.ylim(176,80)


from scattering import *
from retrRoutinesSSRG import *
freq=13.8
freqKa=35.5
#freq=94.0
wl=300/freq
wlKa=300/freqKa
mu=0
nmfreq=8
nmu=5
#!stop

import radtran as rt

rt.readtablesliang2(nmu,nmfreq)
rt.initp2()

#@jit(nopython=False)
def gett_atten(nz,a0,z_att_m,z_m,att_tot,zf):
    print(a0[0])
    for i in a0:
        pia_tot=0
        for k in range(nz-1,-1,-1):
            if z_m[k,i]>-10:
                pia_tot+=att_tot[k,i]*(zf[k+1]-zf[k])*4.343
                z_att_m[k,i]-=pia_tot
                pia_tot+=att_tot[k,i]*(zf[k+1]-zf[k])*4.343
            else:
                z_att_m[k,i]-=pia_tot
                
ncg=7500


[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
                           temp_r,mass_r,bscat_r,Deq_r,ext_r,\
                               scat_r,g_r,vfall_r]=scatTables['13.8']

[tempKa,massKa,fractionKa,bscatKa,DeqKa,extKa,scatKa,gKa,vfallKa,\
                           tempKa_r,massKa_r,bscatKa_r,DeqKa_r,\
                               extKa_r,scatKa_r,gKa_r,vfallKa_r]=scatTables['35.5']
retL=[]
for i in range(40,71):
    
    z_g,att_g,dm_g,prate_g,zka_g,attka_g,dmka_g,prateka_g,gwc,a=\
        estSnow(i,nj,zm,wc,bbpeak,mu,Deq,ext,bscat,scat,g,vfall,\
            Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl,\
            DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
            DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa)
    f=0.25
    
    pwc_mixed=estSnowBB(f,zm[i,nj,bbpeak[i,nj]-1],mu,Deq,ext,bscat,scat,g,vfall,\
            Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl,\
            DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
            DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa,rt)   
    if  zm[i,nj,bbpeak[i,nj]]>10:
        dn=-0.5
        ind=int((zm[i,nj,bbpeak[i,nj]]+12)/0.25)
        indS=int((zm[i,nj,bbpeak[i,nj]-4]-10*dn+12)/0.25)
        indR=int((zm[i,nj,bbpeak[i,nj]+4]-10*dn+12)/0.25)
        pwcBB=10**(rt.tablep2.pwcbb[ind])
        swc=10**dn*10**(rt.tablep2.swc[indS])
        rwc=10**dn*10**(rt.tablep2.rwc[indR])
        retL.append([gwc[a][-1],pwcBB,swc,rwc,wc[i,nj,bbpeak[i,nj]+4],pwc_mixed])
    #@stop
    #print(np.corrcoef(zm[i,nj,80:][a],z_g)[0,1], (zm[i,nj,80:][a]-z_g).mean())
    #print('ka',np.corrcoef(zmka[i,nj,80:][a],zka_g)[0,1], (zmka[i,nj,80:][a]-zka_g).mean())

    #stop
retL=np.array(retL)
plt.figure()
plt.plot(retL[:,0])
plt.plot(retL[:,1])
#plt.plot(retL[:,2])
plt.plot(retL[:,3])
plt.plot(retL[:,4])
plt.plot(retL[:,5])
plt.legend(['ssrg','bb','rTbl','rDPRr','mixed'])
stop   
plt.figure()
bzd=fh['FS/VER/binZeroDeg'][1200:1400,:]
wc=fh['FS/SLV/precipWater'][1200:1400,:,:]
import matplotlib
plt.pcolormesh(wc[:,24,:].T,norm=matplotlib.colors.LogNorm(),cmap='jet')
plt.plot(bzd[:,24])
plt.ylim(176,80)
plt.colorbar()