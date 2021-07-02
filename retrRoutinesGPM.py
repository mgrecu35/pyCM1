#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:28:48 2021

@author: mgrecu
"""
import numpy as np

from numba import jit

from scattering import *

freq=13.8
freqKa=35.5
#freq=94.0
wl=300/freq
wlKa=300/freqKa
mu=0

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
                
zKu1=[11.22,  1.62, -0.16,  1.1 , 14.39, 15.49, 19.37, 21.61,
     23.55, 26.08, 28.79, 31.38, 33.17, 34.95, 36.01, 38.11,
     38.84, 39.49, 38.8 , 38.78, 39.5 , 40.88, 41.25, 41.93, 
     44.03, 45.38, 48.16, 48.52, 48.5 , 48.16, 48.14, 48.51,
     49.57, 49.21, 49.55, 50.58, 50.25]
zKu2=[48.86, 47.84, 45.43, 44.41, 43.71, 43.02, 43.38, 43.03,
                   43.72, 42.34, 40.29, 39.94, 39.58, 39.21, 38.54, 37.86,
                   38.19, 38.92, 39.94, 39.25, 38.21, 38.54, 38.21, 37.53,
                   37.2 ]
ncg=7500
def calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    a=np.nonzero(gwc>0.01)
    nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
    nwg=gwc.copy()*0.0
    w_g=gwc[a].copy()*0.0
    z_g=gwc[a].copy()*0.0
    prate_g=gwc[a].copy()*0.0
    att_g=gwc[a].copy()*0.0
    dm_g=gwc[a].copy()*0.0
    get_Z(gwc[a],nw_g,lambd_g,w_g,z_g,att_g,dm_g,prate_g,Deq[22,:],bscat[-1,22,:],ext[-1,22,:],
          vfall[15,:],mu,wl)
    return z_g,att_g,dm_g,prate_g

def calcZR(rwc,ncr,zf,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    print(wl)
    att_total=rwc.copy()*0
    z_total=rwc.copy()*0.0

    a=np.nonzero(rwc>0.01)
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nwr=rwc.copy()*0.0
    w_r=rwc[a].copy()*0.0
    z_r=rwc[a].copy()*0.0
    prate_r=rwc[a].copy()*0.0
    att_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    get_Z(rwc[a],nw_r,lambd_r,w_r,z_r,att_r,dm_r,prate_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
    return z_r,att_r,dm_r,prate_r

[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
                           temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r]=scatTables['13.8']

[tempKa,massKa,fractionKa,bscatKa,DeqKa,extKa,scatKa,gKa,vfallKa,\
                           tempKa_r,massKa_r,bscatKa_r,DeqKa_r,extKa_r,scatKa_r,gKa_r,vfallKa_r]=scatTables['35.5']

gwc=np.arange(50)*0.1+0.1
rwc=np.arange(50)*0.1+0.1
wl=300/13.8
ncg=7500-500*(gwc-1)
z_g,att_g,dm_g,prate_g=calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
gwcCoeff=np.polyfit(z_g,np.log10(gwc),1)
g_attCoeff=np.polyfit(np.log10(gwc),np.log10(att_g),1)
gwcRet=10**(gwcCoeff[0]*(np.array(zKu1))+gwcCoeff[1])
gattCoeff=np.polyfit(np.log10(gwc),np.log10(att_g),1)

z_r,att_r,dm_r,prate_r=calcZR(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
rwcCoeff=np.polyfit(z_r,np.log10(rwc),1)
r_attCoeff=np.polyfit(np.log10(rwc),np.log10(att_r),1)
hwc=np.linspace(gwcRet[-1],gwcRet[-1]/2.,6)
fract=np.linspace(1.0,0,6)
gwc1=hwc[1:-1]*fract[1:-1]
ncg1=7500-500*(gwc1-1)
z_g1,att_g1,dm_g1,prate_g1=calcZG(gwc1,ncg1,mu,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
rwc1=hwc[1:-1]*(1-fract[1:-1])
ncr1=7500-500*(rwc1-1)
z_r1,att_r1,dm_r1,prate_r1=calcZR(rwc1,ncr1,mu,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
stop
def calcZ(rwc,swc,gwc,ncr,ncs,ncg,zf,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    print(wl)
    att_total=rwc.copy()*0
    z_total=rwc.copy()*0.0

    a=np.nonzero(rwc>0.01)
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nwr=rwc.copy()*0.0
    w_r=rwc[a].copy()*0.0
    z_r=rwc[a].copy()*0.0
    att_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    get_Z(rwc[a],nw_r,lambd_r,w_r,z_r,att_r,dm_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
    print(z_r.max())
    print(w_r.max())
    nwr[a]=np.log10(nw_r)
    #print(rwc[a].mean())
    #print(w_r.mean())
    #stop
    z_total[a]+=10.**(0.1*z_r)
    att_total[a]+=att_r

    a=np.nonzero(swc>0.01)
    nw_s,lambd_s=nw_lambd(swc[a],ncs[a],mu)
    nws=rwc.copy()*0.0

    w_s=swc[a].copy()*0.0
    z_s=swc[a].copy()*0.0
    att_s=swc[a].copy()*0.0
    dm_s=swc[a].copy()*0.0
    get_Z(swc[a],nw_s,lambd_s,w_s,z_s,att_s,dm_s,Deq[12,:],bscat[-1,12,:],ext[-1,12,:],\
          vfall[12,:],mu,wl)
    nws[a]=np.log10(nw_s)
    z_total[a]+=10.**(0.1*z_s)
    att_total[a]+=att_s

    a=np.nonzero(gwc>0.01)
    nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
    nwg=rwc.copy()*0.0

    w_g=gwc[a].copy()*0.0
    z_g=gwc[a].copy()*0.0
    att_g=gwc[a].copy()*0.0
    dm_g=gwc[a].copy()*0.0
    get_Z(gwc[a],nw_g,lambd_g,w_g,z_g,att_g,dm_g,Deq[22,:],bscat[-1,22,:],ext[-1,22,:],\
          vfall[22,:],mu,wl)
    nwg[a]=np.log10(nw_g)

    z_total[a]+=10.**(0.1*z_g)
    att_total[a]+=att_g

    z_total=10*np.log10(z_total+1e-9)
    z_m=np.ma.array(z_total,mask=z_total<-10)
    print(gwc.shape)
    z_att_m=z_total.copy()
    nz=z_att_m.shape[0]
    a=np.nonzero(z_m[0,:]>0)
    #print(a[0])
    gett_atten(nz,a[0],z_att_m,z_total,att_total,zf)
    print(att_total)
    z_att_m=np.ma.array(z_att_m,mask=z_att_m<-10)
    return z_m,z_att_m, att_total, nwr,nws,nwg, w_r, nw_r, z_r

[temp,mass,fraction,bscat,Deq,ext,scat,g,vfall,\
                           temp_r,mass_r,bscat_r,Deq_r,ext_r,scat_r,g_r,vfall_r]=scatTables['13.8']

[tempKa,massKa,fractionKa,bscatKa,DeqKa,extKa,scatKa,gKa,vfallKa,\
                           tempKa_r,massKa_r,bscatKa_r,DeqKa_r,extKa_r,scatKa_r,gKa_r,vfallKa_r]=scatTables['35.5']

fh=Dataset("class03.nc")
z=np.arange(80)*0.25+.125
zf=np.arange(81)*0.25
rwc=fh['rwc'][:,:].T
swc=fh['swc'][:,:].T
gwc=fh['gwc'][:,:].T
ncr=fh['ncr'][:,:].T
ncs=fh['ncs'][:,:].T
ncg=fh['ncg'][:,:].T
zku_ref=fh['zKu'][:,:].T
z_m,z_att_m,att,nwr,nws,nwg,\
    w_r, nw_r, z_r\
    =calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,Deq,ext,bscat,scat,g,vfall,\
                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)


zka_m,zka_att_m,attKa,nwr,nws,nwg,\
    w_r, nw_r, z_r=calcZ(rwc,swc,gwc,ncr,ncs,ncg,z,DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
                         DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa)

z_att_m[z_att_m<0]=0
import matplotlib.pyplot as plt

rwc1=rwc[:,0:400].copy()
ncr1=ncr[:,0:400].copy()
swc1=swc[:,0:400].copy()
ncs1=ncs[:,0:400].copy()
gwc1=gwc[:,0:400].copy()
ncg1=ncg[:,0:400].copy()
plt.figure()
plt.plot(swc.mean(axis=1))
plt.plot(rwc.mean(axis=1))
plt.plot(gwc.mean(axis=1))  
plt.figure()
plt.figure()
plt.plot(ncs.mean(axis=1))
plt.plot(ncr.mean(axis=1))
plt.plot(ncg.mean(axis=1))  
rwc1L=[]
swc1L=[]
gwc1L=[]
ncr1L=[]
ncs1L=[]
ncg1L=[]
for f in np.arange(0.1,3.,0.05):
    rwc1L.append(f*rwc.mean(axis=1))
    swc1L.append(f*swc.mean(axis=1))
    gwc1L.append(f*gwc.mean(axis=1))
    ncr1L.append(1/f*rwc.mean(axis=1))
    ncs1L.append(1/f*ncs.mean(axis=1))
    ncg1L.append(1/f*ncg.mean(axis=1))

rwc1L=np.array(rwc1L).T
swc1L=np.array(swc1L).T
gwc1L=np.array(gwc1L).T
ncr1L=np.array(ncr1L).T
ncs1L=np.array(ncs1L).T
ncg1L=np.array(ncg1L).T
z_m1,z_att_m1,att1,nwr1,nws1,nwg1,\
    w_r, nw_r, z_r\
    =calcZ(rwc1L,swc1L,gwc1L,ncr1L,ncs1L,ncg1L,\
           z,Deq,ext,bscat,scat,g,vfall,Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
        
plt.figure()
plt.pcolormesh(z_att_m1,cmap='jet',vmax=50)
plt.colorbar()
plt.figure()
plt.plot(z_att_m1[:,-1],z)
plt.plot(z_att_m1[:,0],z)
plt.plot(z_att_m1[:,10],z)
plt.plot(z_att_m1[:,-15],z)
plt.colorbar()
stop
import matplotlib
plt.subplot(211)       
plt.pcolormesh(4.343*att[0:50,:40]/10**(0.1*z_m[0:50,:40]*.792),cmap='jet',norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.subplot(212)       
plt.pcolormesh(z_att_m[0:50,:40],cmap='jet')
plt.colorbar()

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
kernel =RBF()
X=z_att_m[0:30,:700].T
Xt=z_att_m[0:30,700:].T
y=rwc[0,:700]
yt=rwc[0,700:]
gpr = GaussianProcessRegressor(kernel=kernel,
        alpha=1.05**2,random_state=0).fit(X, y)
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=30)
neigh.fit(X,y)
yp=neigh.predict(Xt)
stop
alpham=(4.343*att[0:50,:]/10**(0.1*z_m[0:50,:]*.792)).mean(axis=1)
alpha1=(4.343*att[0:50,:]/10**(0.1*z_m[0:50,:]*.792))

def HB(zm,alpham,dr):
    nz,nx=zm.shape
    piaL=[]
    
    for i in range(nx):
        k=alpham*10**(0.1*zm[:,i]*0.792)
        zeta=0.2*np.log(10)*k.sum()*dr*0.792
        #print()
        piaL.append((1-0.8*zeta)**(1/0.792))
    return piaL

def HB1(zm,alpha1,dr):
    k=alpha1*10**(0.1*zm[:]*0.792)*dr
    zeta=0.2*np.log(10)*k.sum()*0.792
    piaL=-10*np.log10(1-zeta)*(1/0.792)
    return piaL

dr=0.25
for i in range(50):
    pia1=HB1(z_att_m[:50,i],alpham,dr)
    print(pia1,z_m[0,i]-z_att_m[0,i])

#stop
def HB1Ens(zm,alpha,nEns,dr):
    ns=alpha.shape[1]
    piaL=[]
    for i in range(nEns):
        iEn=int(np.random.random()*ns)
        alpha1=alpha[:,i]
        k=alpha1*10**(0.1*zm[:]*0.792)*dr
        zeta=0.2*np.log(10)*k.sum()*0.792
        if zeta<0.999:
            piaL.append(-10*np.log10(1-zeta)*(1/0.792))
    return piaL

piaEnsL=[]

for i in range(50):
    pia1=HB1Ens(z_att_m[:50,i],alpha1[:50,50:],150,dr)
    piaEnsL.append([pia1,z_m[0,i]-z_att_m[0,i]])
    if len(pia1)<150:
        stop