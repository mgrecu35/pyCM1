#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:54:13 2021

@author: mgrecu
"""

import pickle
import xarray as xr
import numpy as np
for i in range(4,-12):
    d=pickle.load(open("Data/precipProfilesDPR_2018%2.2i_NH_v2O.pklz"%(i),"rb"))
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
    dSet.to_netcdf("Data/tb2018%2.2i_NH_v2O.nc"%(i))
    print("precipProfilesDPR_2018%2.2i_NH_v2O.pklz"%(i))
#stop   
from netCDF4 import Dataset
xL=[]
yL=[]
from numba import jit
@jit(nopython=False)
def getAvgProf(bzdL,pTypeL,stormTop,ptype,pRateDPR,pRateCMB,bcL,c1,mProf):
    a=np.nonzero(bcL>167)
    pRateCMB2=pRateDPR.copy()*0
    pRateCMB[:,0:33][pRateCMB[:,0:33]<0]=0
    for j in a[0]:
        if pRateCMB[j,32]>=0 and pRateCMB[j,33]<0:
            pRateCMB[j,33]=pRateCMB[j,32]
        if pRateCMB[j,32]>=0 and pRateCMB[j,34]<0:
            pRateCMB[j,34]=pRateCMB[j,32]
       
        pRateCMB2[j,0:69]=np.interp(np.arange(69),2*np.arange(35),pRateCMB[j,0:35])
        if stormTop[j]>100:
            pRateCMB2[j,0:stormTop[j]-100:]=0
    pRateDPR_1=pRateCMB2
    for ibzd in range(130,176):
        b=np.nonzero(bzdL[a]==ibzd)
        c=np.nonzero(pTypeL[a][b]==1)
        #print(len(c[0]))
        c1[ibzd-130]=len(c[0])     
        if len(c[0])>0:
            mProf[ibzd-130,:,0]=pRateDPR_1[a[0][b][c],20:68].sum(axis=0)
        c=np.nonzero(pTypeL[a][b]==2)
        d=np.nonzero(stormTop[a][b][c]<ibzd+8)
        if len(d[0])>0:
            mProf[ibzd-130,:,1]=pRateDPR_1[a[0][b][c][d],20:68].sum(axis=0)
        d=np.nonzero(stormTop[a][b][c]>=ibzd+8)
        if len(d[0])>0:
            mProf[ibzd-130,:,2]=pRateDPR_1[a[0][b][c][d],20:68].sum(axis=0)
            
@jit(nopython=False)
def get2dhist(bzdL,pTypeL,stormTop,ptype,pRateDPR,pRateCMB,bcL,cHist,cHist2,sumS,sumS2,mProfT,\
              c1,mProf,systDiff):
    a=np.nonzero(bcL>166)
    pRateCMB2=pRateDPR.copy()*0
    pRateCMB[:,0:33][pRateCMB[:,0:33]<0]=0
    for j in a[0]:
        if pRateCMB[j,32]>=0 and pRateCMB[j,33]<0:
            pRateCMB[j,33]=pRateCMB[j,32]
        if pRateCMB[j,32]>=0 and pRateCMB[j,34]<0:
            pRateCMB[j,34]=pRateCMB[j,32]
        #pRateCMB[j,pRateCMB[j,:]<0]=0
        pRateCMB2[j,0:69]=np.interp(np.arange(69),2*np.arange(35),pRateCMB[j,0:35])
        if stormTop[j]>100:
            pRateCMB2[j,0:stormTop[j]-100:]=0
    pRateDPR_1=pRateCMB2.copy()
    for i in a[0]:
        if bzdL[i]<134:
            continue
        if bcL[i]<=167:
            continue
        if pRateDPR_1[i,66]>=0:
            irng=int(np.random.random()*18)+150        
            i1=int(np.log(pRateDPR_1[i,66]+1e-9)/0.25)+2
            i2=int(np.log(pRateDPR_1[i,irng-100]+1e-9)/0.25)+2
            ibzd=bzdL[i]
            if ibzd>=176:
                continue
            #print(ibzd,irng-120)
            #while(pRateDPR_1_1[i,irng-100]<=0) and irng>140:
            #    irng-=1
            if pRateDPR_1[i,irng-100]<0.:
                continue
            if pTypeL[i]==1:
                correctRatio=mProfT[ibzd-130,irng-120,0]
                mProf[ibzd-130,irng-120,0]+=pRateDPR_1[i,irng-100]
                c1[ibzd-130,irng-120,0]+=1
                systDiff[0,0]+=pRateDPR_1[i,67]
                systDiff[0,1]+=pRateDPR_1[i,irng-100]
            else:
                #continue
                if stormTop[i]>=ibzd+8:
                    correctRatio=mProfT[ibzd-130,irng-120,2]
                    mProf[ibzd-130,irng-120,2]+=pRateDPR_1[i,irng-100]
                    c1[ibzd-130,irng-120,2]+=1
                    systDiff[2,0]+=pRateDPR_1[i,67]
                    systDiff[2,1]+=pRateDPR_1[i,irng-100]
                else:
                    correctRatio=mProfT[ibzd-130,irng-120,1]
                    mProf[ibzd-130,irng-120,1]+=pRateDPR_1[i,irng-100]
                    c1[ibzd-130,irng-120,1]+=1
                    systDiff[1,0]+=pRateDPR_1[i,67]
                    systDiff[1,1]+=pRateDPR_1[i,irng-100]
           
            if correctRatio!=correctRatio:
                correctRatio=1
            corrPRate=pRateDPR_1[i,irng-100]/(correctRatio+1e-4)
            if pTypeL[i]==1:
                systDiff[0,2]+=corrPRate
            else:
                if stormTop[i]>=ibzd+8:
                    systDiff[2,2]+=corrPRate
                else:
                    systDiff[1,2]+=corrPRate
            #print(corrPRate)
            sumS[0]+=(pRateDPR_1[i,67])
            sumS[1]+=(pRateDPR_1[i,irng-100])
            sumS[2]+=(pRateDPR_1[i,67])**2
            sumS[3]+=(pRateDPR_1[i,irng-100])**2
            sumS[4]+=(pRateDPR_1[i,irng-100])*pRateDPR_1[i,67]
            sumS[5]+=1
            i2=min(29,i2)
            i2=max(0,i2)
            i1=min(29,i1)
            i1=max(0,i1)
            i3=int(np.log(corrPRate+1e-9)/0.25)+2
            i3=min(29,i3)
            i3=max(0,i3)
            cHist[i1,i2]+=1
            cHist2[i1,i3]+=1
            sumS2[0]+=(pRateDPR_1[i,67])
            sumS2[1]+=(corrPRate)
            sumS2[2]+=(pRateDPR_1[i,67])**2
            sumS2[3]+=(corrPRate)**2
            sumS2[4]+=(corrPRate)*pRateDPR_1[i,67]
            sumS2[5]+=1
    #return mProf
import numpy as np
c1T=np.zeros((46),int)
mProfT=np.zeros((46,48,3),float)
mProfS=np.zeros((46,48,3),float)
cS=np.zeros((46,48,3),float)

cHist=np.zeros((30,30),float)

cHist2=np.zeros((30,30),float)
systDiff=np.zeros((3,3),float)

for i in range(1,-9):
    print(i)
    fh=Dataset("Data/tb2018%2.2i_NH_L.nc"%(i))
for i in range(4,12):
    print(i)
    fh=Dataset("Data/tb2018%2.2i_SO.nc"%(i))    #tb=fh["tb"][:,:]
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
    mProf=np.zeros((46,48,3),float)
    pRateDPR[pRateDPR<0]=0
    getAvgProf(bzdL,pType,stormTop,ptype,pRateDPR,pRateCMB,bcL,c1,mProf)
    c1T+=c1
    mProfT+=mProf
    continue
    
    

for i in range(46):
    mProfT[i,:,0]/=mProfT[i,-1,0]
    mProfT[i,:,1]/=mProfT[i,-1,1]
    mProfT[i,:,2]/=mProfT[i,-1,2]
  
from scipy.ndimage import gaussian_filter
for i in range(3): 
    mProfT[i,:,i]=gaussian_filter(mProfT[i,:,i], sigma=0.5)
sumS=np.zeros((8),float)
sumS2=np.zeros((8),float)
for i in range(1,-9):
    print(i)
    fh=Dataset("Data/tb2018%2.2i_NH_L.nc"%(i))
for i in range(4,12):
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
    #getAvgProf(bzdL,pType,stormTop,ptype,pRateDPR,pRateCMB,bcL,c1,mProf)
    #for k in range(15):
    get2dhist(bzdL,pType,stormTop,ptype,pRateDPR,pRateCMB,bcL,cHist,cHist2,sumS,sumS2,mProfT,\
              cS,mProfS,systDiff)
    #c1T+=c1
    #mProfT+=mProf
    continue 

import matplotlib.pyplot as plt
st=['stratiform','convective','shallow']
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

for i in range(3):
    plt.figure()
    plt.pcolormesh(130+np.arange(46),120+np.arange(48),np.array(mProfT[:,:,i]).T,cmap='gist_earth',vmax=1.1);
    plt.ylim(166,120)   
    plt.xlim(134,175)
    if i==2:
        plt.xlim(134,160)
    plt.xlabel('Zero degree bin')
    plt.ylabel('Range bin')
    plt.colorbar()
    plt.title("Ocean %s"%st[i])
    #plt.colorbar()
    plt.tight_layout()
    plt.savefig('correctionTableDPR_Ocean_%s.png'%st[i])
mProfS/=(cS+1e-9)
for i in range(46):
    mProfS[i,:,0]/=mProfS[i,-1,0]
    mProfS[i,:,1]/=mProfS[i,-1,1]
    mProfS[i,:,2]/=mProfS[i,-1,2]
        
for i in range(-3):
    plt.figure()
    plt.pcolormesh(130+np.arange(46),120+np.arange(48),np.array(mProfS[:,:,i]).T,cmap='gist_earth',vmax=1.1);
    plt.ylim(166,120)   
    plt.xlim(134,175)
    plt.xlabel('Zero degree bin')
    plt.ylabel('Range bin')
    plt.colorbar()
    plt.title("Ocean %s"%st[i])
    #plt.colorbar()
    plt.savefig('resampleD_DPR_%s_Ocean.png'%st[i])

#stop
fig=plt.figure()
ax=plt.subplot(111)
w=0.1
ax.bar(np.arange(3)-w,(systDiff[:,1]-systDiff[:,0])/systDiff[:,0], 2*w, label='uncorrected')
ax.bar(np.arange(3)+w,(systDiff[:,2]-systDiff[:,0])/systDiff[:,0], 2*w, label='corrected')
ax.plot([-0.25,2.25],[0,0],color='black',linewidth=0.5)
#ax.set_xticklabels(['St',' ','Cv',' ','Sh'])
#print(ax.get_xtickla)
ax.set_xticklabels(['','St','','Cv','','Sh',''])
plt.xlim(-0.25,2.25)
ax.legend()
plt.title('Ocean')
plt.tight_layout()
plt.savefig('biasesOcean.png')
import xarray as xr

p1=xr.DataArray(mProfT,dims=['nFL','nbin','nt'])
d=xr.Dataset({"cvstTable":p1})
d.to_netcdf("OceanTablesNH.nc")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x=np.exp(-2+np.arange(30)*0.25)
y=np.exp(-2+np.arange(30)*0.25)
plt.pcolormesh(x,y,
               cHist.T,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm/h)')
plt.ylabel('Persistence-based prediction (mm/h)')
plt.title('Ocean')
plt.colorbar()
plt.tight_layout()
plt.savefig('OceanPersistenceHist2d.png')


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x=np.exp(-2+np.arange(30)*0.25)
y=np.exp(-2+np.arange(30)*0.25)
plt.pcolormesh(x,y,
               cHist2.T,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm/h)')
plt.title('Ocean')
plt.ylabel('VPP prediction (mm/h)')
plt.colorbar()
plt.tight_layout()
plt.savefig('OceanVPP_Hist2d.png')


def corrCoef(sumS):
    x1=sumS[0]/sumS[5]
    y1=sumS[1]/sumS[5]
    sx=sumS[2]-sumS[5]*x1**2
    sy=sumS[3]-sumS[5]*y1**2
    sxy=sumS[4]-sumS[5]*x1*y1
    print(x1,y1)
    c=sxy/np.sqrt(sx*sy)
    return c
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
dset.to_netcdf("trainFeaturesOceanComb.nc")
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
