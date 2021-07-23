#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:33:54 2021

@author: mgrecu
"""

from netCDF4 import Dataset

fh=Dataset("trainFeaturesLandComb.nc")

X=fh["X"][:]
y=fh["y"][:]
#stop
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import matplotlib.pyplot as plt

#---4-tbs--dist(152,ic)--dist(bzd,iclutter)--iwc--pRateLowest--bdzd--
import numpy as np
pRatio2D=[]
corrcoef2D=[]
for bzd in range(130,176):
    a=np.nonzero(X[:,-1]==bzd)
    print(len(a[0]))
    pRatio=[]
    cc=[]
    #stop
    for i in range(0,15):
        b=np.nonzero(X[:,4][a]==i)
        print(bzd,len(b[0]))
        pRatio.append((X[:,-2][a][b]).mean()/y[a][b].mean())
        cc.append(np.corrcoef(X[:,-2][a][b],y[a][b])[0,1])
    #stop
    pRatio2D.append(pRatio)
    corrcoef2D.append(cc)
    
plt.pcolormesh(130+np.arange(46),153+np.arange(15),np.array(pRatio2D).T,cmap='viridis_r');
plt.title('Ratio of precipitation aloft to \nnear surface precipitation')
plt.ylim(167,153)
plt.xlabel('Zero degree bin')
plt.ylabel('Range bin')
plt.colorbar()
plt.savefig('precipitationRatioCMBLand.png')
#stop
from scipy.ndimage import gaussian_filter
plt.figure()
plt.pcolormesh(130+np.arange(46),153+np.arange(15),np.array(gaussian_filter(corrcoef2D,2)).T,cmap='jet');
plt.title('Correlation Coefficient')
plt.xlabel('Zero degree bin')
plt.ylabel('Lowest clutter free bin')
plt.ylim(167,153)
c=plt.colorbar()
plt.savefig('correlationCoeffCMBLand.png')

plt.figure()
#2stop
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.hist2d((y[:]),(X[:,-2]),bins=np.exp(-2+np.arange(25)*0.2),cmin=100,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('Persistance-based prediction (mm/h)')
plt.title("Ocean both types")
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('unCorrectSfcPrecipCMBLand.png')
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import StandardScaler

yL=y.copy()
xL=X.copy()
yL[yL<0.01]=0.01
#yL=np.log(yL)
scaler = StandardScaler()
scalerY = StandardScaler()
#xL=scaler.fit_transform(xL)
#yL=scalerY.fit_transform(yL[:,np.newaxis])
import pickle
r=np.random.random(yL.shape[0])
if 1==0:
    a=np.nonzero(r>0.5)
    b=np.nonzero(r<0.5) 
    pickle.dump({"a":a,"b":b},open("splitIndices.pklz","wb"))
else:
    dab=pickle.load(open("splitIndices.pklz","rb"))
    a=dab["a"]
    b=dab["b"]
    
x_train=xL[a[0],:]
y_train=yL[a[0]]
x_val=xL[b[0],:]
y_val=yL[b[0]]
import tensorflow as tf     

def dmodel(ndims=13):
    inp = tf.keras.layers.Input(shape=(ndims,))
    out1 = tf.keras.layers.Dense(16,activation='relu')(inp)
    out1 = tf.keras.layers.Dropout(0.1)(out1)
    out2 = tf.keras.layers.Dense(16,activation='relu')(out1)
    out2 = tf.keras.layers.Dropout(0.1)(out2)
    out = tf.keras.layers.Dense(1)(out2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

pRatio2D_train=[]
pRatio2D_trainP=[]
pRatio2D_trainN=[]
pRatio2D_trainT=[]
pRatio2D_trainS=[]
tb2d_train=[]
#corrcoef2D=[]
import random
for bzd in range(130,176):
    a=np.nonzero(x_train[:,-1]==bzd)
    print(len(a[0]))
    pRatio=[]
    pRatioN=[[],[]]
    pRatioP=[[],[]]
    pRatioS=[]
    pRatioT=[[],[]]
    cc=[]
    tb=[]
    for i in range(0,15):
        b=np.nonzero(x_train[:,4][a]==i)     
        print(bzd,len(b[0]))
        pRatio.append((x_train[:,-2][a][b]).mean()/y_train[a][b].mean())
        c=np.nonzero(x_train[:,5][a][b]>0.052)
        d1=np.nonzero(x_train[:,7][a][b][c]==2)
        pRatioP[1].append((x_train[:,-2][a][b][c][d1]).mean()/y_train[a][b][c][d1].mean())
        d1=np.nonzero(x_train[:,7][a][b][c]==1)
        pRatioP[0].append((x_train[:,-2][a][b][c][d1]).mean()/y_train[a][b][c][d1].mean())
        c=np.nonzero(x_train[:,5][a][b]<0.052)
        d2=np.nonzero(x_train[:,7][a][b][c]==2)
        pRatioN[1].append((x_train[:,-2][a][b][c][d2]).mean()/y_train[a][b][c][d2].mean())
        d2=np.nonzero(x_train[:,7][a][b][c]==1)
        pRatioN[0].append((x_train[:,-2][a][b][c][d2]).mean()/y_train[a][b][c][d2].mean())
        d2=np.nonzero(x_train[:,7][a][b]==2)
        d3=np.nonzero(x_train[:,7][a][b][d2]<bzd)
        if len(d3[0]>1):
            pRatioT[1].append((x_train[:,-2][a][b][d2][d3]).mean()/y_train[a][b][d2][d3].mean())
        else:
            pRatioT[1].append((x_train[:,-2][a][b][d2]).mean()/y_train[a][b][d2].mean())
        d2=np.nonzero(x_train[:,7][a][b]==1)
        pRatioT[0].append((x_train[:,-2][a][b][d2]).mean()/y_train[a][b][d2].mean())
        print(len(d1[0]),len(d2[0]))
        if(pRatioN[0][-1]!=pRatioN[0][-1]):
            pRatioN[0][-1]=pRatio[-1]       
        if(pRatioN[1][-1]!=pRatioN[1][-1]):
            pRatioN[1][-1]=pRatio[-1]
        if(pRatioT[0][-1]!=pRatioT[0][-1]):
            pRatioT[0][-1]=pRatio[-1]
        if(pRatioT[1][-1]!=pRatioT[1][-1]):
            pRatioT[1][-1]=pRatio[-1]
        if(pRatioP[0][-1]!=pRatioP[0][-1]):
            pRatioP[0][-1]=pRatio[-1]
        if(pRatioP[1][-1]!=pRatioP[1][-1]):
            pRatioP[0][-1]=pRatio[-1]
        s1=[]
        for k in range(4):
            s1.append(0.)
        tb.append(x_train[a[0][b],:].mean(axis=0))
        pRatioS.append(np.std(s1))
        #stop
    pRatio2D_train.append(pRatio)
    pRatio2D_trainP.append(pRatioP)
    pRatio2D_trainN.append(pRatioN)
    pRatio2D_trainS.append(pRatioS)
    pRatio2D_trainT.append(pRatioT)
    #stop
    tb2d_train.append(tb)
ypL=[]
pRatio2D_train=np.array(pRatio2D_train)  
pRatio2D_trainP=np.array(pRatio2D_trainP) 
pRatio2D_trainN=np.array(pRatio2D_trainN) 
pRatio2D_trainT=np.array(pRatio2D_trainT) 
for i,x1 in enumerate(x_train): 
    ibzd=int(x1[-1]-130)
    irng=int(x1[4])
    irng=min(14,irng)
    ibzd=min(46,ibzd)
    #x_train[i,-2]=(x1[-2]/pRatio2D_train[ibzd,irng])
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()

#x_train=scaler.fit_transform(x_train)

pRatio2D_train2=gaussian_filter(pRatio2D_train, sigma=2)
pRatio2D_trainN[:,0,:]=gaussian_filter(pRatio2D_trainN[:,0,:], sigma=1)
pRatio2D_trainP[:,0,:]=gaussian_filter(pRatio2D_trainP[:,0,:], sigma=1)
pRatio2D_trainN[:,1,:]=gaussian_filter(pRatio2D_trainN[:,1,:], sigma=2)
pRatio2D_trainP[:,1,:]=gaussian_filter(pRatio2D_trainP[:,1,:], sigma=2)

pRatio2D_trainT[:,0,:]=gaussian_filter(pRatio2D_trainT[:,0,:], sigma=1)
pRatio2D_trainT[:,1,:]=gaussian_filter(pRatio2D_trainT[:,1,:], sigma=1)
#stop
ypLnoSlope=[]
for i,x1 in enumerate(x_val):
    ibzd=int(x1[-1]-130)
    irng=int(x1[4])
    irng=min(14,irng)
    ibzd=min(46,ibzd)
    #if x1[5]>0.052:
    if(x1[7]==1):
        ypL.append(x1[-2]/pRatio2D_trainT[ibzd,0,irng])
    else:
        ypL.append(x1[-2]/pRatio2D_trainT[ibzd,1,irng])
    #else:
    #    if(x1[7]==1):   
    #        ypL.append(x1[-2]/pRatio2D_trainN[ibzd,0,irng])
    #    else:
    #        ypL.append(x1[-2]/pRatio2D_trainN[ibzd,1,irng])
    if(ypL[-1]!=ypL[-1]):
        stop
    ypLnoSlope.append(x1[-2]/pRatio2D_train[ibzd,irng])
    if ypLnoSlope[-1]>300:
        ypLnoSlope[-1]=300
    if ypL[-1]>300:
        ypL[-1]=300
    #x_val[i,-2]=ypL[-1]
    

#x_val=scaler.transform(x_val)
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=15,weights='distance')
neigh.fit(x_train[:,0:4], y_train[:]-np.array(x_train[:,-2]))
#ypL2=neigh.predict(x_val)

plt.figure()
plt.pcolormesh(130+np.arange(46),153+np.arange(15),np.array(gaussian_filter(pRatio2D_train,2)).T,\
               cmap='gist_earth');
plt.title('Ratio of precipitation aloft to \nnear surface precipitation')
plt.ylim(167,153)
plt.xlabel('Zero degree bin')
plt.ylabel('Range bin')
plt.title("Ocean both types")
plt.colorbar()
plt.savefig('correctionTableCMBLand.png')

plt.figure()
plt.pcolormesh(130+np.arange(46),153+np.arange(15),pRatio2D_trainT[:,1,:].T,\
               cmap='gist_earth');
plt.title('Ratio of precipitation aloft to \nnear surface precipitation')
plt.ylim(167,153)
plt.xlabel('Zero degree bin')
plt.ylabel('Range bin')
plt.title('Ocean Convective')
plt.colorbar()
plt.savefig('correctionTableCvCMBLand.png')

plt.figure()
plt.pcolormesh(130+np.arange(46),153+np.arange(15),pRatio2D_trainT[:,0,:].T,\
               cmap='gist_earth');
plt.title('Ratio of precipitation aloft to \nnear surface precipitation')
plt.ylim(167,153)
plt.xlabel('Zero degree bin')
plt.ylabel('Range bin')
plt.title('Ocean Stratiform')
plt.colorbar()
plt.savefig('correctionTableStCMBLand.png')





fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.hist2d((y_val[:]),np.array(ypL[:]),bins=np.exp(-2+np.arange(25)*0.2),cmin=500,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('VPP prediction (mm/h)')
plt.title("Ocean both types")
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('correctSfcPrecipCMBLand.png')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
a1=np.nonzero(x_val[:,7]==2)
plt.hist2d((y_val[:])[a1],np.array(ypL[:])[a1],bins=np.exp(-2+np.arange(25)*0.2),cmin=500,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.title("Ocean Convective")
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('VPP prediction (mm/h)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('correctConvSfcPrecipCMBLand.png')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
a1=np.nonzero(x_val[:,7]==1)
plt.hist2d((y_val[:])[a1],x_val[:,-2][a1],bins=np.exp(-2+np.arange(25)*0.2),cmin=500,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.title("Ocean Stratiform")
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('Persistence-based prediction (mm/h)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('unCorrectStratSfcPrecipCMBLand.png')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
a1=np.nonzero(x_val[:,7]==1)
plt.hist2d((y_val[:])[a1],np.array(ypL[:])[a1],bins=np.exp(-2+np.arange(25)*0.2),cmin=500,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.title("Ocean Stratiform")
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('VPP prediction (mm/h)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('correctStratSfcPrecipCMBLand.png')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
a1=np.nonzero(x_val[:,7]==2)
plt.hist2d((y_val[:])[a1],x_val[:,-2][a1],bins=np.exp(-2+np.arange(25)*0.2),cmin=500,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.title("Ocean Convective")
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('Persistence-based prediction (mm/h)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('unCorrectConvSfcPrecipCMBLand.png')
#plt.savefig('precipitationRatioCMB.png)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#plt.hist2d((y_val[:]),(ypL2[:]),bins=np.exp(-2+np.arange(25)*0.2),cmin=100,cmap='jet')
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_aspect('equal')
#plt.xlabel('True precipitation rate (mm)')
#plt.ylabel('Predicted precipitation rate (mm)')
#c=plt.colorbar()
#c.ax.set_title('Counts')
#plt.savefig('correctSfcPrecip2CMB.png')

import xarray as xr

p1=xr.DataArray(pRatio2D_trainN,dims=['nFL','n2','nbin'])
p2=xr.DataArray(pRatio2D_trainP,dims=['nFL','n2','nbin'])
p3=xr.DataArray(pRatio2D_train2,dims=['nFL','nbin'])
p4=xr.DataArray(pRatio2D_trainT,dims=['nFL','n2','nbin'])
d=xr.Dataset({"nTable":p1,"pTable":p2,"aTable":p3,"cvstTable":p4})
d.to_netcdf("oceanTables.nc")