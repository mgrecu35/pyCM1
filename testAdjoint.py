from numpy import *
import scipy.linalg as ln
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator,LogFormatter
from matplotlib.backends.backend_pdf import PdfPages

def Eddington1D(k,a,g,T,dz):
    n=k.shape[0]-1
    A=zeros((2*n+3,2*n+3),float)
    B=zeros((2*n+3),float)
    dz=0.25
    for i in range(0,n+1):
        A[2*i+1,2*i+1]=+3*k[i]*(1-a[i])*dz
        A[2*i+1,2*i+2]=1
        A[2*i+1,2*i]=-1
        B[2*i+1]=3*k[i]*(1-a[i])*T[i]*dz
        if(2*i+3<2*n+3):
            km=0.5*(k[i]+k[i+1])
            am=0.5*(a[i]+a[i+1])
            gm=0.5*(g[i]+g[i+1])
            A[2*i+2,2*i+2]=km*(1-am*gm)*dz
            A[2*i+2,2*i+3]=1
            A[2*i+2,2*i+1]=-1
            B[2*i+2]=0.
    A[2*n+2,2*n+1]=1.
    A[2*n+2,2*n+2]=-1./3
    A[2*n+2,2*n]=-1./3
    B[2*n+2]=2.7
    A[0,0]=(2-eps)/3./eps
    A[0,1]=1.
    A[0,2]=(2-eps)/3./eps
    B[0]=292
    I01p=dot(linalg.pinv(A),B)
    #I01p=ln.solve_banded((2,2), A, B)
    Tb=(I01p[2*n]+I01p[2*n+2])/2*cos(53./180.*pi)+I01p[2*n+1]
    return Tb,A,B, I01p

eps=0.9
dz=0.25
#k=dat[:,0]
#a=dat[:,1]
#g=dat[:,2]
#T=dat[:,3]
f=19.
ts=300
#from radtran import *
import radtran as rt
umu=cos(53/180.*pi)
import pickle
kext3d,salb3d,asym3d,t2,height=pickle.load(open('scattFields.neoguri.pickle','rb'), encoding='bytes')

n1=kext3d.shape[2]
levs=[0.1,0.2,0.4,0.8,1.6,3.2,6.4,15]
plt.figure()
cs=plt.contourf(arange(n1),height[:-1,5,0],kext3d[:,5,:,4],levels=levs,norm=LogNorm())
plt.xlim((0,n1-1))
plt.colorbar(cs)
plt.xlabel('Grid Cell')
plt.ylabel('Height')

plt.title('Extinction Coeff (km-1)')
#plt.savefig(pp,orientation='portrait',papertype='landscape',format='pdf')
plt.savefig('extinct.tiff')
plt.figure()
levs=arange(13)*0.0675+0.2
levs[-1]=1.
cs=plt.contourf(arange(n1),height[:-1,5,0],salb3d[:,5,:,4],levels=levs)
plt.xlim((0,n1-1))
plt.colorbar(cs)
plt.xlabel('Grid Cell')
plt.ylabel('Height')
plt.title('Scattering albedo')
plt.savefig('scatt.tiff')
#plt.savefig(pp,orientation='portrait',papertype='landscape',format='pdf')
plt.figure()
levs=arange(11)*0.07+0.2
cs=plt.contourf(arange(n1),height[:-1,5,0],asym3d[:,5,:,4],levels=levs)
plt.xlim((0,n1-1))
plt.colorbar(cs)

plt.xlabel('Grid Cell')
plt.ylabel('Height')
plt.title('Asymmetry factor')
plt.savefig('asymmetry.tiff')

dx=0.25
dz=0.25
dx=2.
def getSlantProp(a,hm,ny,nx,dx):
    asl=[]
    t1=tan(53./180*pi)
    for k in range(a.shape[0]):
        dnx=int((20-hm[k])*t1/dx)
        #dnx=0
        asl.append(a[k,ny,nx+dnx,4])
    #print asl
    asl=array(asl)
    aint=interp(arange(80)*0.25+.125,hm,asl)
    return aint,asl
t1L=[]
t2L=[]
kb2d=zeros((100,80),float)
kb2d_fd=zeros((100,80),float)
for nx in range(25,125):
    ny=4
    hm=0.5*height[:-1,ny,nx]+0.5*height[1:,ny,nx]
    k,k2=getSlantProp(kext3d,hm,ny,nx,dx)
    a,a2=getSlantProp(salb3d,hm,ny,nx,dx)
    g,g2=getSlantProp(asym3d,hm,ny,nx,dx)

    T=interp(arange(80)*0.25+.125,hm,t2[:,ny,nx])
    #stop
    eps=0.9
    emis=0.9
    ebar=0.9
    umu=cos(53/180.*pi)
    nlyr=k.shape[0]-1
    lyrhgt=0.25*arange(nlyr+1)
    fisot=2.7
    Ts=T[0]
    gp=g/(1+g)
    kp=(1-a*g**2)*k
    ap=(1-g**2)*a/(1-a*g**2)
    tb = rt.radtran(umu,nlyr,Ts,T,lyrhgt,k[:-1],a[:-1],g[:-1],fisot,emis,ebar)
    abig,b = rt.seteddington1d(k[:],a[:],g[:],T,eps,dz,Ts)
    incang=53.
    i01p=linalg.solve(abig,b)
    tb2 = rt.tbf90(i01p,T,incang,k[:],a[:],g[:],eps,dz)
    tbb=1.0
    lam2=i01p
    i01pb,kb,ab,gb = rt.tbf90_b(tb2,tbb,i01p,T,incang,k,a,g,eps,dz)
    print(i01pb)
    y=0.
    yb=1.
    lam1=i01pb
    lam11=dot(i01pb,linalg.inv(abig))
    kb1,ab1,gb1 = rt.suml1al2_b(y,yb,lam11,lam2,k,a,g,T,eps,dz)
    y=0.
    yb=1
    kb2,ab2 = rt.suml1b_b(y,yb,lam11,k,a,g,T,eps,dz)
    print(tb,tb2)
    t1L.append(tb)
    t2L.append(tb2)
    kb2d[nx-25,:]=ab-ab1+ab2
    n=80
    for i in range(n):
        ap=a.copy()
        ap[i]=ap[i]+0.01
        abig,b = rt.seteddington1d(k[:],ap[:],g[:],T,eps,dz,Ts)
        incang=53.
        i01p=linalg.solve(abig,b)
        tb21 = rt.tbf90(i01p,T,incang,k[:],ap[:],g[:],eps,dz)      
        kb2d_fd[nx-25,i]=(tb21-tb2)/0.01
plt.figure()
plt.plot(t1L)
plt.plot(t2L)
plt.figure()

plt.subplot(211)
plt.pcolormesh(kb2d[:,::-1].T,cmap='RdBu')

plt.subplot(212)
plt.pcolormesh(kb2d_fd[:,::-1].T,cmap='RdBu')
