import cv2
import os,sys
import time
import numpy as np
import scipy as sc
from pylab import *
from scipy import signal
from numpy.linalg import inv
from scipy.linalg import sqrtm 
from scipy.ndimage import filters
from scipy.signal import convolve2d
from sklearn.neighbors import KNeighborsClassifier


# import function

from mydiscgaussfft import *
from dxmsk import *
from conv2 import *
from crop2 import *
from extend2 import *
from maxsupression import *
from drawellipse_cov import *


img=[[1, 2], 
   [4, 5]];

a = [ [1.3333,    2.0000,   91.0000, 123.6667,   131.3333,  121.3333,  155.3333,  106.6667,    2.3333,    1.3333],
	    [4.6667,    2.0000,   95.0000, 133.6667 ,  140.3333,  129.3333,  166.3333,  129.6667,    2.0000,    1.3333],
	    [1.0000,    1.3333,   84.6667,  130.3333,  130.6667 , 112.6667,  149.6667,  132.0000,    2.0000,    1.0000],
	    [1.0000,    1.3333,   89.6667,  140.3333,  129.6667,   99.6667,  131.6667,  131.0000,    2.0000,    1.0000],
    	[0.3333,    4.0000,   89.6667,  147.0000,  131.6667,   95.6667,  116.6667,  124.0000,    2.0000,    1.0000],
    	[0.3333,    1.3333,   79.3333,  146.0000,  138.6667,  108.6667,  115.3333,  119.6667,    1.3333,    1.0000],
    	[2.0000,    7.0000,   72.6667,  147.3333,  156.6667,  140.3333,  132.6667,  126.3333,    1.3333,    1.0000],
    	[0.3333,         0,   51.6667,  130.0000,  153.6667,  146.6667,  129.6667,  113.0000,    1.3333,    1.0000],
    	[0.3333,    0.3333,   57.6667,  112.0000,  135.3333,  116.6667,  121.6667,  110.0000,    1.3333,    0.3333],
    	[4.3333,    4.3333,   88.6667,  157.0000,  177.3333,  161.6667,  169.3333,  148.0000,    1.0000,    0.3333]]

#print(img)
#print(sum(img,0))
#img = cv2.imread('frame1.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
height = np.size(img, 0)
width = np.size(img, 1)

sxl2 = 4
sxi2 = 2*sxl2
kparam = 0.04
npoints = 40

print(4)
L = mydiscgaussfft(extend2(img,4,4),sxl2)
print(5)
#print(np.size(L))

dxmask = dxmsk()
#print(dxmask)

lx = conv2(L,np.rot90(dxmask), mode='same')
Lx = crop2(lx,4,4)
Lx=Lx*2
#print(Lx)

dxmaskt=dxmask.transpose()
ly = conv2(L,np.rot90(dxmaskt), mode='same')
Ly = crop2(ly,4,4)
Ly=Ly*(2)
#print(Ly)


Lxm2=Lx*Lx
Lym2=Ly*Ly
Lxmy=Lx*Ly
#print(Lxmy)

Lxm2smooth=mydiscgaussfft(Lxm2,sxi2)
Lym2smooth=mydiscgaussfft(Lym2,sxi2)
Lxmysmooth=mydiscgaussfft(Lxmy,sxi2)
#print(Lxmysmooth)

detC=(Lxm2smooth*Lym2smooth)-(Lxmysmooth*Lxmysmooth)
trace2C=(Lxm2smooth+Lym2smooth)*(Lxm2smooth+Lym2smooth)
#print(trace2C)

cimg=detC-(kparam*trace2C)
#print(cimg)
#print(np.size(cimg,0))

#--------------------

position,Value = maxsupression(cimg)
#print(np.size(Value))
pos=[]
pxall=[]
pyall=[]

#print(position)
if np.size(position)>0:
    pxall=position[:,1]
    pyall=position[:,0]


#print(pxall)
#print(pyall)

sv=[]
si=[]

value = np.array(Value)
for i in range(0,np.size(value)):
	value[i]=-1*value[i]
#print(value)


sv = sorted(value,reverse=False)
si = [i[0] for i in sorted(enumerate(value), key=lambda x:x[1])]
#np.savetxt('CheckSet.txt',si)
#print(sv)
#print(np.size(si))

hight = np.size(si)
#print(hight)

px1 = si[0:min(npoints,hight)]
py1 = si[0:min(npoints,hight)]
szy = np.size(pyall)
szx = np.size(pxall)
px = []
py = []	
pxx = []
pyy = []	
val = []

#print(px1)
#print(py1)
        
for i in range(0,np.size(px1)):
	point = px1[i]-1	
	px.append(pxall[point])

for i in range(0,np.size(py1)):
	point = py1[i]-1	
	py.append(pyall[point])
#print(py)

for i in range(0,np.size(px)):
    	pxx.append(px[np.size(px)-i-1])

for i in range(0,np.size(py)):
	pyy.append(py[np.size(py)-i-1])
px = pxx
py = pyy

val = sv[0:min(npoints,hight)]
#print(np.size(val)
for i in range(0,np.size(val)):
	val[i]=-1*val[i]


#print(px)
#print(py)
#print(val)

shape = np.size(img,0)
#print(shape)

#ind=sub2ind(shape,py,px)
#print(ind)
ind = []
x=0

for i in range(0,np.size(px)):
	x = shape * (px[i]-1)
	x = x + py[i]
	ind.append(x) 	

#ind = px + py*np.size(img,1);

#print(ind)

c11 = []
c12 = []
c22 = []
	
for i in range(0,np.size(ind)):
   	point = ind[i]-1
	c=point/np.size(Lxm2smooth,0)
	r=point%np.size(Lxm2smooth,0)
	Lxm2smooth = Lxm2smooth.transpose()	
	c11.append([Lxm2smooth[c,r]])
#print(c11)

for i in range(0,np.size(ind)):
	point = ind[i]-1
	c=point/np.size(Lxmysmooth,0)
	r=point%np.size(Lxmysmooth,0)
	Lxmysmooth = Lxmysmooth.transpose()		
	c12.extend([Lxmysmooth[c,r]])	
#print(c12)

for i in range(0,np.size(ind)):
	point = ind[i]-1
	c=point/np.size(Lym2smooth,0)
	r=point%np.size(Lym2smooth,0)
	Lym2smooth = Lym2smooth.transpose()		
	c22.extend([Lym2smooth[c,r]])		
#print(c22)
	
posinit = []
#del posinit[:]

posinit=[px,py,sxl2*np.ones(size(px)),c11,c12,c12,c22]
#print(posinit)
if 1 :
	bound=2
	insideind=find((px>bound)*(px<(shape-bound))*(py>bound)*(py<(shape-bound)))
	#print(insideind)
	#posinit=posinit[insideind,:]
epos = []
epos = posinit
fcol = [1,1,0]

x1 = np.array(epos[0])	
#print(x1[1])
x2 = np.array(epos[1])	
#print(x2)
x3 = np.array(epos[2])	
#print(x3)
x4 = np.array(epos[3])	
#print(x4)
x5 = np.array(epos[4])	
#print(x5)
x6 = np.array(epos[5])	
#print(x6)
x7 = np.array(epos[6])	
#print(np.size(x7))
	
for i in range(0,2):
	x0 = x1[i]
	y0 = x2[i]
	s0 = x3[i]
	Sigma = np.array([[x4[i] ,x5[i]] ,[x6[i] ,x7[i]]],np.float)
	Sigma = inv(Sigma)
	x = np.linalg.det(Sigma)
	if x<0: x=x*-1
	Sigma = Sigma/np.sqrt(x)
	#print(i)
	h = drawellipse_cov(Sigma*s0*10,[x0,y0])
	#print(h)
	#plt.figure()
	h = plt.plot(x0,y0,color = fcol)
	#plt.show()
	#_ = raw_input("Press [enter] to continue.")		
	#plt.close()