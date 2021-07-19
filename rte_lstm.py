#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:54:13 2021

@author: mgrecu
"""

import pickle
import xarray as xr

for i in range(0,-1):
    d=pickle.load(open("precipProfilesDPR_2018%2.2i_SH_v2O.pklz"%(4+i),"rb"))
    tb=xr.DataArray(d["tb"],dims=["nt","ntb"])
    wc=xr.DataArray(d["pWatCont_cmb"],dims=["nt","nwc"])
    pRate=xr.DataArray(d["pRate_cmb"],dims=["nt","nwc"])
    pRateDPR=xr.DataArray(d["pRate"],dims=["nt","nz"])
    zm=xr.DataArray(d["zm"],dims=["nt","nz","n2"])
    bcL=xr.DataArray(d["bcL"],dims=["nt"])
    bzdL=xr.DataArray(d["bzdL"],dims=["nt"])
    dSet=xr.Dataset({"tb":tb,"wc":wc,"bcL":bcL,"bzdL":bzdL,"pRate":pRate,"pRateDPR":pRateDPR, "zm":zm})
    dSet.to_netcdf("tb2018%2.2i_SO.nc"%(4+i))
    print("precipProfilesDPR_2018%2.2i_SH_v2O.pklz"%(4+i))
from netCDF4 import Dataset
xL=[]
yL=[]
import numpy as np
for i in range(4):
    fh=Dataset("tb2018%2.2i_SO.nc"%(4+i))
    tb=fh["tb"][:,:]
    zm=fh["zm"][:,:,:]
    zm[zm<0]=0
    bcL=fh["bcL"][:]
    bzdL=fh["bzdL"][:]
    pRateDPR=fh["pRateDPR"][:,:]
    for irec in range(bzdL.shape[0]):
        if(bzdL[irec]>132 and bcL[irec]>=167 and bzdL[irec]<167 and zm[irec,67,0]>10 and tb[irec,5:9].min()>0):
            x1=[]
            #z1=zm[irec,10:67,0]
            #z1[z1<0]=0
            x1.extend(tb[irec,5:9])
            #x1=
            z1=zm[irec,bzdL[irec]-100-25:bzdL[irec]-100,0]
           
            #print(zm[irec,bzdL[irec]-100-25:bzdL[irec]-100,0])
            #print(iwc)
            #x1.append(iwc)
            #x1[x1<=0]=0
            ir=152+int(np.random.random()*15)
            irc=min(ir,bcL[irec])
            dn=irc-bzdL[irec]
            #print(zm[irec,bzdL[irec]-100:bzdL[irec]-100+dn,0],dn)
            #if dn>=15:
            #    x2=list(zm[irec,bzdL[irec]-100:bzdL[irec]-100+dn,0][0:15])
            #else:
            #    if dn>=0:
            #        x2=list(zm[irec,bzdL[irec]-100:bzdL[irec]-100+dn,0])
            #        x2.extend([-10 for k in range(15-dn)])
            #    else:
            #        x2=[-10 for k in range(15)]
            #        x1[dn:]=-10
                    #print(x1)
                    #stop
            #x1=list(x1)
            #x1.extend(x2)
            x1.append(irc-152)
            if dn<0:
                z1[dn:]=-10
            iwc=(10**(0.1*0.8*z1)).sum()
            x1.append(dn)
            x1.append(iwc)
            x1.append(pRateDPR[irec,irc-100])
            x1.append(bzdL[irec])
            #x1.extend(pRateDPR[irec,40:67])
            xL.append(x1)
            #stop
            #if max(x1)>53:
            #    stopsx1
            yL.append(pRateDPR[irec,67])
            #yL.append(tb[irec,5:11])
   
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
dset.to_netcdf("trainFeaturesOcean15.nc")
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
