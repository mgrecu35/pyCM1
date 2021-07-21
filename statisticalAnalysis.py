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
plt.hist2d((y[:]),(X[:,-2]),bins=np.exp(-2+np.arange(25)*0.2),cmin=100,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('Nadir precipitation rate (mm)')
plt.ylabel('Assumed off nadir precipitation rate (mm)')
c=plt.colorbar()
c.ax.set_title('Counts')
plt.savefig('unCorrectSfcPrecip.png')
from sklearn.preprocessing import StandardScaler

yL=y.copy()
xL=X.copy()
yL[yL<0.01]=0.01
yL=np.log(yL)
scaler = StandardScaler()
scalerY = StandardScaler()
xL=scaler.fit_transform(xL)
yL=scalerY.fit_transform(yL[:,np.newaxis])
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
y_train=yL[a[0],0]
x_val=xL[b[0],:]
y_val=yL[b[0],0]
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

cModel=dmodel(9)

cModel.compile(loss='mse', \
               optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])


history = cModel.fit(x_train, y_train, batch_size=64,epochs=50,\
                     validation_data=(x_val, y_val))
    
    
yp=cModel.predict(x_val)[:,:]
#pRate_pred=yL.inverse_transform(yp)
from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=15,weights='distance')
#neigh.fit(x_train, y_train[:])
#plt.figure()
#pltyp2=neigh.predict(x_val)[:]
yp3=cModel.predict(x_val)[:,0]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#plt.hist2d((pRate_pred[:,0]),scalerY.inverse_transform(y_val),bins=np.exp(-2+np.arange(25)*0.2),cmin=1000,cmap='jet')
plt.hist2d(y_val[:]*scalerY.scale_+scalerY.mean_,yp3[:]*scalerY.scale_+scalerY.mean_,\
           bins=(-2+np.arange(25)*0.2),cmin=100,cmap='jet')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm)')
plt.ylabel('Predicted precipitation rate (mm)')


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#plt.hist2d((pRate_pred[:,0]),scalerY.inverse_transform(y_val),bins=np.exp(-2+np.arange(25)*0.2),cmin=1000,cmap='jet')
plt.hist2d(y_val[:]*scalerY.scale_+scalerY.mean_,x_val[:,-2]*scaler.scale_[-2]+scaler.mean_[-2],\
           bins=np.exp(-2+np.arange(25)*0.2),cmin=100,cmap='jet')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_aspect('equal')
plt.xlabel('True precipitation rate (mm)')
plt.ylabel('Predicted precipitation rate (mm)')
