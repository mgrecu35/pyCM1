#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 11:16:42 2021

@author: mgrecu
"""
from netCDF4 import Dataset
from minisom import MiniSom
import numpy as np
fh=Dataset("simZ_Neg2.nc")
zKu=fh["zku"][:,:,:]
zKum=fh["zku_att"][:,:,:]
zKum[zKum<0]=0
a=np.nonzero(zKu[0,:,:]>42)
n2=1
n1=10
som_c = MiniSom(n1,n2,80,sigma=0.25,learning_rate=0.125, random_seed=0)
zKuL=[]
rwc=fh['rwc'][:,:,:]
swc=fh['swc'][:,:,:]
gwc=fh['gwc'][:,:,:]
ncr=fh['rwc'][:,:,:]
ncs=fh['ncs'][:,:,:]
ncg=fh['ncg'][:,:,:]
h1L=[]
for i in range(len(a[0])):
    zKuL.append(zKum[:,a[0][i],a[1][i]])
    h1=[]
    h1.extend(rwc[0:15,a[0][i],a[1][i]])
    h1.extend(gwc[7:40,a[0][i],a[1][i]])
    h1.extend(swc[20:50,a[0][i],a[1][i]])
    h1.extend(ncr[0:15,a[0][i],a[1][i]])
    h1.extend(ncg[7:40,a[0][i],a[1][i]])
    h1.extend(ncs[20:50,a[0][i],a[1][i]])
    h1L.append(h1)


h1L=np.array(h1L)
from sklearn import preprocessing
scaler  = preprocessing.StandardScaler()
scaler_hwc = scaler.fit(h1L)

hwc = scaler_hwc.transform(h1L)


from sklearn.decomposition import PCA


hpca = PCA()
hpca.fit(hwc)
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5,5))
ax.plot(hpca.explained_variance_ratio_[0:10]*100)
ax.plot(hpca.explained_variance_ratio_[0:10]*100,'ro')

zKuL=np.array(zKuL)
zKuL[zKuL!=zKuL]=0
som_c.random_weights_init(zKuL[:,:80])
som_c.train_random(zKuL[:,:80],1500) # training with 100 iterations
print("\n...ready!")

classAvc=np.zeros((n1,n2,80),float)
countc=np.zeros((n1,n2),float)

for it in range(zKuL.shape[0]):
    win=som_c.winner(zKuL[it,:80])
    classAvc[win[0],win[1],:]+=zKuL[it,:80]
    countc[win[0],win[1]]+=1
plt.figure()
labels=[]   
for i in range(5):
    plt.plot(classAvc[i,0,:]/countc[i,0],np.arange(80)*.25)
    labels.append('class %2i'%i)
plt.legend(labels)
plt.ylim(0,15)
plt.figure()   
for i in range(5,8):
    plt.plot(classAvc[i,0,:]/countc[i,0],np.arange(80)*.25)
plt.ylim(0,15)    
plt.figure()   
for i in range(8,10):
    plt.plot(classAvc[i,0,:]/countc[i,0],np.arange(80)*.25)
plt.ylim(0,15)