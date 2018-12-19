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
from maxsupressionn import *


def STIP(img):
	#img = cv2.imread('frame1.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height = np.size(img, 0)
	width = np.size(img, 1)

	sxl2 = 4
	sxi2 = 2*sxl2
	kparam = 0.04
	npoints = 40

	#L = mydiscgaussfft(extend2(img,4,4),sxl2)
	L = mydiscgaussfft(img,sxl2)
	#print(np.size(L))

	dxmask = dxmsk()
	#print(dxmask)

	lx = conv2(L,dxmask, mode='same')
	#height = np.size(lx, 0)
	#width = np.size(lx, 0)
	#print(lx)
	#print(height)
	#print(width)

	Lx = crop2(lx,4,4)
	Lx=Lx*2
	#print(Lx)

	dxmaskt=dxmask.transpose()
	ly = conv2(L,dxmaskt, mode='same')
	Ly = crop2(ly,4,4)
	Ly=Ly*(2)
	#print(Ly)


	Lxm2=Lx*Lx
	Lym2=Ly*Ly
	Lxmy=Lx*Ly
	#print(Lxm2)

	Lxm2smooth=mydiscgaussfft(Lxm2,sxi2)
	Lym2smooth=mydiscgaussfft(Lym2,sxi2)
	Lxmysmooth=mydiscgaussfft(Lxmy,sxi2)
	#print(Lxm2smooth)

	detC=(Lxm2smooth*Lym2smooth)-(Lxmysmooth*Lxmysmooth)
	trace2C=(Lxm2smooth+Lym2smooth)*(Lxm2smooth+Lym2smooth)
	#print(trace2C)

	cimg=detC-(kparam*trace2C)
	#print(cimg[0])
	#print(np.size(cimg,0))

	Value = maxsupressionn(cimg)
	#print(np.size(Value))
	

	sv=[]
	si=[]

	value = np.array(Value)
	for i in range(0,np.size(value)):
		value[i]=-1*value[i]
	#print(value)


	sv = sorted(value,reverse=False)
	si = [i[0] for i in sorted(enumerate(value), key=lambda x:x[1])]
	
	hight = np.size(si)
	#print(hight)

	

	val = sv[0:min(npoints,hight)]
	#print(np.size(val)
	for i in range(0,np.size(val)):
		val[i]=-1*val[i]


	return val,cimg









