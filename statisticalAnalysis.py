#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:33:54 2021

@author: mgrecu
"""

from netCDF4 import Dataset

fh=Dataset("trainFeaturesOcean15.nc")

X=fh["X"][:]
y=fh["y"][:]
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import matplotlib.pyplot as plt

#---4-tbs--dist(152,ic)--dist(bzd,iclutter)--iwc--pRateLowest--bdzd--
import numpy as np
pRatio2D=[]
corrcoef2D=[]
for bzd in range(136,167):
    a=np.nonzero(X[:,-1]==bzd)
    print(len(a[0]))
    pRatio=[]
    cc=[]
    for i in range(0,15):
        b=np.nonzero(X[:,4][a]==i)
        print(bzd,len(b[0]))
        pRatio.append((X[:,-2][a][b]).mean()/y[a][b].mean())
        cc.append(np.corrcoef(X[:,-2][a][b],y[a][b])[0,1])
    pRatio2D.append(pRatio)
    corrcoef2D.append(cc)
    
plt.pcolormesh(136+np.arange(31),153+np.arange(15),np.array(pRatio2D).T,cmap='viridis_r');
plt.title('Ratio of precipitation aloft to \nnear surface precipitation')
plt.ylim(167,153)
plt.xlabel('Zero degree bin')
plt.ylabel('Range bin')
plt.colorbar()
plt.savefig('precipitationRatio.png')


plt.figure()
plt.pcolormesh(136+np.arange(31),153+np.arange(15),np.array(corrcoef2D).T,cmap='jet');
plt.title('Correlation Coefficient')
plt.xlabel('Zero degree bin')
plt.ylabel('Lowest clutter free bin')
plt.ylim(167,153)
c=plt.colorbar()
plt.savefig('correlationCoeff.png')

plt.figure()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.hist2d((y[:]),(X[:,-2]),bins=np.exp(-2+np.arange(25)*0.2),cmin=1000,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('Nadir precipitation rate (mm)')
plt.ylabel('Assumed off nadir precipitation rate (mm)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('unCorrectSfcPrecip.png')