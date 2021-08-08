#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 20:45:46 2021

@author: mgrecu
"""
import numpy as np
from scattering import *

def calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    a=np.nonzero(gwc>0.001)
    nw_g,lambd_g=nw_lambd(gwc[a],ncg[a],mu)
    nwg=gwc.copy()*0.0
    w_g=gwc[a].copy()*0.0
    z_g=gwc[a].copy()*0.0
    prate_g=gwc[a].copy()*0.0
    att_g=gwc[a].copy()*0.0
    dm_g=gwc[a].copy()*0.0
    get_Z(gwc[a],nw_g,lambd_g,w_g,z_g,att_g,dm_g,prate_g,Deq[19,:],bscat[-1,19,:],ext[-1,19,:],
          vfall[15,:],mu,wl)
    return z_g,att_g,dm_g,prate_g

def calcZR(rwc,ncr,zf,Deq,ext,bscat,scat,g,vfall,\
          Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl):
    print(wl)
    att_total=rwc.copy()*0
    z_total=rwc.copy()*0.0

    a=np.nonzero(rwc>0.001)
    nw_r,lambd_r=nw_lambd(rwc[a],ncr[a],mu)
    nwr=rwc.copy()*0.0
    w_r=rwc[a].copy()*0.0
    z_r=rwc[a].copy()*0.0
    prate_r=rwc[a].copy()*0.0
    att_r=rwc[a].copy()*0.0
    dm_r=rwc[a].copy()*0.0
    get_Z(rwc[a],nw_r,lambd_r,w_r,z_r,att_r,dm_r,prate_r,Deq_r,bscat_r[9,:],ext_r[9,:],vfall_r,mu,wl)
    return z_r,att_r,dm_r,prate_r

def estSnow(i,nj,zm,wc,bbpeak,mu,Deq,ext,bscat,scat,g,vfall,\
            Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl,\
            DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
            DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa):
    gwc=1.5*wc[i,nj,80:bbpeak[i,nj]-2]
    gwc1=1.6*wc[i,nj,80:bbpeak[i,nj]-2]
    dwc=gwc1-gwc+1e-4
    wl=300/13.8
    ncg=7500/2-0.50*(gwc-1)
    #stop
    z_g,att_g,dm_g,prate_g=calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
                                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
    z_g1,att_g1,dm_g1,prate_g1=calcZG(gwc1,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
                                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
    a=np.nonzero(gwc>0.001)
    dzdw=(z_g1-z_g)/(dwc[a])
    dwg=(zm[i,nj,80:][a]-z_g)*dzdw/(dzdw**2+0.1)
    gwc[a]+=0.75*dwg
    b=np.nonzero(gwc[a]<0.00101)
    gwc[a[0][b]]=0.0011
    z_g,att_g,dm_g,prate_g=calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
                                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
    zka_g,attka_g,dmka_g,prateka_g=calcZG(gwc,ncg,mu,DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
                                  DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa)
        
    return z_g,att_g,dm_g,prate_g,zka_g,attka_g,dmka_g,prateka_g,gwc,a

def estSnowBB(f,zm,mu,Deq,ext,bscat,scat,g,vfall,\
              Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl,\
                  DeqKa,extKa,bscatKa,scatKa,gKa,vfallKa,\
                      DeqKa_r,extKa_r,bscatKa_r,scatKa_r,gKa_r,vfallKa_r,wlKa,rt):
    pwc=1.0
    gwc=np.array([f*pwc])
    gwc1=1.1*gwc
    ncg=7500/2-0.50*(gwc-1)
    print(gwc,ncg)
    z_g,att_g,dm_g,prate_g=calcZG(gwc,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
                                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
    z_g1,att_g1,dm_g1,prate_g1=calcZG(gwc1,ncg,mu,Deq,ext,bscat,scat,g,vfall,\
                                  Deq_r,ext_r,bscat_r,scat_r,g_r,vfall_r,wl)
    pwcBB=np.log10((1-f)*pwc)
    pwcBB1=np.log10(1.1*(1-f)*pwc)
    ifind=rt.bisection2(rt.tablep2.pwcbb, pwcBB)
    ifind1=rt.bisection2(rt.tablep2.pwcbb, pwcBB1)
    print(ifind,ifind1)