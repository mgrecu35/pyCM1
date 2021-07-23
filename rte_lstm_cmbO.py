#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:54:13 2021

@author: mgrecu
"""

import pickle
import xarray as xr
import numpy as np
for i in range(0,-8):
    d=pickle.load(open("Data/precipProfilesDPR_2018%2.2i_NH_v2L.pklz"%(1+i),"rb"))
    #tb=xr.DataArray(d["tb"],dims=["nt","ntb"])
    stormTop=xr.DataArray(np.array(d["TBBPR"])[:,3],dims=["nt"])
    pType=xr.DataArray(np.array(d["TBBPR"])[:,0],dims=["nt"])
    #stop
    wc=xr.DataArray(d["pWatCont_cmb"],dims=["nt","nwc"])
    pRate=xr.DataArray(d["pRate_cmb"],dims=["nt","nwc"])
    pRateDPR=xr.DataArray(d["pRate"],dims=["nt","nz"])
    zm=xr.DataArray(d["zm"],dims=["nt","nz","n2"])
    bcL=xr.DataArray(d["bcL"],dims=["nt"])
    bzdL=xr.DataArray(d["bzdL"],dims=["nt"])
    dSet=xr.Dataset({"wc":wc,"bcL":bcL,"bzdL":bzdL,"pRate":pRate,"pRateDPR":pRateDPR, \
                     "zm":zm,"pType":pType,"stormTop":stormTop})
    dSet.to_netcdf("Data/tb2018%2.2i_NH_L.nc"%(1+i))
    print("precipProfilesDPR_2018%2.2i_NH_v2L.pklz"%(1+i))
    
from netCDF4 import Dataset
xL=[]
yL=[]
from numba import jit
@jit(nopython=False)
def getAvgProf(bzdL,pTypeL,stormTop,ptype,pRateDPR,pRateCMB,bcL,c1,mProf):
    a=np.nonzero(bcL>167)
    pRateCMB2=pRateDPR.copy()*0
    pRateCMB[pRateCMB<0]=0
    for j in a[0]:
        if pRateCMB[j,32]>=0 and pRateCMB[j,33]<-1:
            pRateCMB[j,33]=pRateCMB[j,32]
        if pRateCMB[j,32]>=0 and pRateCMB[j,34]<-1:
            pRateCMB[j,34]=pRateCMB[j,32]
        pRateCMB2[j,0:68]=np.interp(np.arange(68),2*np.arange(35),pRateCMB[j,0:35])
    #pRateDPR=pRateCMB2
    for ibzd in range(130,176):
        b=np.nonzero(bzdL[a]==ibzd)
        c=np.nonzero(pTypeL[a][b]==1)
        #print(len(c[0]))
        c1[ibzd-130]=len(c[0])     
        if len(c[0])>0:
            mProf[ibzd-130,:,0]=pRateDPR[a[0][b][c],20:67].sum(axis=0)
        c=np.nonzero(pTypeL[a][b]==2)
        d=np.nonzero(stormTop[a][b][c]<ibzd+8)
        if len(d[0])>0:
            mProf[ibzd-130,:,1]=pRateDPR[a[0][b][c][d],20:67].sum(axis=0)
        d=np.nonzero(stormTop[a][b][c]>=ibzd+8)
        if len(d[0])>0:
            mProf[ibzd-130,:,2]=pRateDPR[a[0][b][c][d],20:67].sum(axis=0)
    #return mProf
import numpy as np
c1T=np.zeros((46),int)
mProfT=np.zeros((46,47,3),float)
for i in range(4,8):
    print(i)
    fh=Dataset("Data/tb2018%2.2i_SO.nc"%(i))
    #tb=fh["tb"][:,:]
    zm=fh["zm"][:,:,:]
    pType=fh["pType"][:]
    stormTop=fh["stormTop"][:]
    zm[zm<0]=0
    bcL=fh["bcL"][:]
    bzdL=fh["bzdL"][:]
    pRateDPR=fh["pRateDPR"][:,:]
    pRateCMB=fh["pRate"][:,:]
    pRateCMB2=pRateDPR.copy()*0
    a=np.nonzero(zm[:,67,0]>10)
    b=np.nonzero(bcL[a]>167)
    ptype=2
    c1=np.zeros((46),int)
    mProf=np.zeros((46,47,3),float)
    pRateDPR[pRateDPR<0]=0
    getAvgProf(bzdL,pType,stormTop,ptype,pRateDPR,pRateCMB,bcL,c1,mProf)
    c1T+=c1
    mProfT+=mProf
    continue
    stop
    for j in a[0]:
        if pRateCMB[j,32]>=0 and pRateCMB[j,33]<-1:
            pRateCMB[j,33]=pRateCMB[j,32]
        if pRateCMB[j,32]>=0 and pRateCMB[j,34]<-1:
            pRateCMB[j,34]=pRateCMB[j,32]
        pRateCMB2[j,0:68]=np.interp(np.arange(68),2*np.arange(35),pRateCMB[j,0:35])
    #stop
    #pRateDPR=pRateCMB2
    for irec in a[0][b]:#range(bzdL.shape[0]):
        if(bzdL[irec]>=128 and bcL[irec]>=167 and bzdL[irec]<176 and zm[irec,67,0]>10):
            if bzdL[irec]<136:
                k=5
            else:
                k=2
            r1=pRateDPR[irec,47:52]
            r1[r1<0]=0
            slopeR=np.polyfit(np.arange(5),r1,1)          
            for it in range(k):
                x1=[]
                x1.extend([-99.9, -99.9, -99.9, -99.9])
                
                #x1[x1<=0]=0
                ir=142+int(np.random.random()*25)
                irc=min(ir,bcL[irec])
                dn=irc-bzdL[irec]
                x1.append(irc-142)
                x1.append(slopeR[0])
                x1.append(stormTop[irec])
                x1.append(pType[irec])  
                z1=zm[irec,bzdL[irec]-100-25:bzdL[irec]-100,0].copy()
                if dn<0:
                    z1[dn:]=-10
                iwc=(10**(0.1*0.8*z1)).sum()
                x1.append(dn)
                x1.append(iwc)
                x1.append(pRateDPR[irec,irc-100])
                x1.append(bzdL[irec])
                xL.append(x1)
                yL.append(pRateDPR[irec,67])
            #yL.append(tb[irec,5:11])
    print(len(xL))
    

for i in range(46):
    mProfT[i,:,0]/=mProfT[i,-1,0]
    mProfT[i,:,1]/=mProfT[i,-1,1]
    mProfT[i,:,2]/=mProfT[i,-1,2]
import matplotlib.pyplot as plt

for i in range(3):
    plt.figure()
    plt.pcolormesh(130+np.arange(46),120+np.arange(47),np.array(mProfT[:,:,i]).T,cmap='gist_earth',vmax=1.1);
    plt.ylim(166,140)   
    plt.xlim(134,175)
    plt.colorbar()

stop   
import tensorflow as tf     
#from tf.keras.preprocessing.sequences import pad_sequences       
def dmodel(ndims=13):
    inp = tf.keras.layers.Input(shape=(ndims,))
    out1 = tf.keras.layers.Dense(16,activation='relu')(inp)
    out1 = tf.keras.layers.Dropout(0.1)(out1)
    out2 = tf.keras.layers.Dense(16,activation='relu')(out1)
    out2 = tf.keras.layers.Dropout(0.1)(out2)
    out = tf.keras.layers.Dense(1)(out2)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

def lstm_model(ndims=1):
    ntimes=None
    inp = tf.keras.layers.Input(shape=(ntimes,ndims,))
    out1 = tf.keras.layers.LSTM(12, return_sequences=True)(inp)
    out1 = tf.keras.layers.LSTM(6, recurrent_activation='sigmoid',return_sequences=True)(out1)
    #out1 = tf.keras.layers.LSTM(6, return_sequences=True)(out1)
    out = tf.keras.layers.LSTM(2, return_sequences=False)(out1)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

#def lstm_model

cModel=dmodel(9)
#cModel=lstm_model(1)
from sklearn.preprocessing import StandardScaler
import numpy as np
xL=np.array(xL)
yL=np.array(yL)
xLu=xL.copy()
yLu=yL.copy()

import xarray as xr
xLux=xr.DataArray(xLu)
yLux=xr.DataArray(yLu)
dset=xr.Dataset({"X":xLux,"y":yLux})
dset.to_netcdf("trainFeaturesLandComb.nc")
stop
yL[yL<0]=0
scaler = StandardScaler()
scalerY = StandardScaler()
xL=scaler.fit_transform(xL)
yL=scalerY.fit_transform(yL[:,np.newaxis])

r=np.random.random(yL.shape[0])
a=np.nonzero(r>0.5)
b=np.nonzero(r<0.5)
x_train=xL[a[0],:]
y_train=yL[a[0],0]
x_val=xL[b[0],:]
y_val=yL[b[0],0]

from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=50,weights='distance')
#neigh.fit(x_train, y_train[:])
from sklearn.cluster import KMeans

#kmeans=KMeans(n_clusters=50, random_state=10).fit(x_train)

#rmean=np.zeros((50),float)
#for i in range(50):
#    a=np.nonzero(kmeans.labels_=i)
#    rmean[i]=y_train[a].mean(axis=0)
#stop

#stop
cModel.compile(loss='mse', \
               optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])



history = cModel.fit(x_train, y_train, batch_size=64,epochs=50,\
                     validation_data=(x_val, y_val))
