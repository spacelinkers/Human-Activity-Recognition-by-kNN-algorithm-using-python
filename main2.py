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




dxmaska=[]

def dxmsk():
	dxmaska=np.array([[0, 0, 0,  0, 0, 0, 0, 0, 0], 
		[0, 0, 0,  0, 0, 0, 0, 0, 0],  
		[0, 0, 0,  0, 0, 0, 0, 0, 0], 
		[0, 0, 0,  0, 0, 0, 0, 0, 0],  
		[0, 0, 0, -0.5, 0, 0.5, 0, 0, 0], 
		[0, 0, 0,  0, 0, 0, 0, 0, 0], 
		[0, 0, 0,  0, 0, 0, 0, 0, 0],  
		[0, 0, 0,  0, 0, 0, 0, 0, 0], 
		[0, 0, 0,  0, 0, 0, 0, 0, 0]])
	return dxmaska

def showellipticfeatures(epos,fcol):
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
	
	for i in range(0,40):
		x0 = x1[i]
		y0 = x2[i]
		s0 = x3[i]
		Sigma = np.array([[x4[i] ,x5[i]] ,[x6[i] ,x7[i]]],np.float)
		Sigma = inv(Sigma)
		x = np.linalg.det(Sigma)
		if x<0: x=x*-1
		Sigma = Sigma/np.sqrt(x)
		#print(Sigma)
		h = drawellipse_cov(Sigma*s0*10,[x0,y0])
		
		#plt.figure()
		h = plt.plot(x0,y0,color = fcol)
		#plt.show()
		#_ = raw_input("Press [enter] to continue.")		
		#plt.close()
		
		

def drawellipse_cov(cov, poss):
	points = np.linspace(0,2*pi,100)
	#print(points)
	
	r1 = cos(points).transpose()
	#print(r1)
	r2 = sin(points).transpose()
	r3 = [r1,r2]
	r4 = np.array(r3)
	r5 = np.dot(sqrtm(cov),r4)
	ell = r5.transpose()
	#print(ell)
	
	ell[:,0]=ell[:,0]+poss[0]
	ell[:,1]=ell[:,1]+poss[1]
	
	#print(ell)
	
	h=plt.plot(ell[:,0],ell[:,1], lineWidth = 1.5)
	#plt.show()
	return h
	
'''
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
'''

def conv2(x, y, mode='same'):
    return signal.convolve2d(x, np.rot90(y), mode='same')

def pad(A, val):

	#A=np.array([[0,0,6],
	#	[8,0,0]])

	#val = float(-inf)


	sizr = np.size(A,0)
	sizc = np.size(A,1)+2

	addToCol=np.c_[ val*np.ones(sizr),A, val*np.ones(sizr) ] 

	padmatrix=np.r_[ val*np.ones((1,sizc)),addToCol, val*np.ones((1,sizc)) ]  
		    
	#print(padmatrix)
	return padmatrix

Value = []


def maxsupression(A):
	#A=np.array([[1,0,6],
	#	[8,0,0]])

	del Value[:]
	framed = np.zeros(((np.size(A,0)+2),(np.size(A,1)+2)))
	sizx = np.size(A,0)+1
	sizy = np.size(A,1)+1
	framed[1:sizx,1:sizy]=A
	A=framed
	#print(A)

	Shifts = np.zeros((prod(size(A)),9))
	#print(Shifts)

	x=np.eye(9)
	for i in range(0,9):
		y = x[:,i]
		X = pad( A, -inf)
		#print(X)

		z = y.transpose()
		Y = np.reshape( z,( 3, 3))
		#print(Y)
	
		Shift = signal.convolve2d(X,Y, mode='valid')
		#Shift = conv2(X,Y, mode='valid')
		#print(Shift)
	

		fi = find(z)
		#print(fi)

		z1 = np.size(Shift,0)
		z2 = np.size(Shift,1)
		#print(z1,z2,z3)
		l=0
		for j in range(0,z2):
			for k in range(0,z1):
				Shifts[l,fi] = Shift[k,j]
				#print(Shifts)
				l=l+1


		#Shifts[:,fi] = Shift[:]
		#print(Shifts[0,0])

	#print(Shifts)


	Anms = np.ones((size(A)))
	#print(np.size(Anms))

	for shiftind in range(0,4):
	  	  Anms = np.uint64(Anms) & np.uint64(Shifts[ :, 4] >= Shifts[ :, shiftind])
	#print(Anms)

	for shiftind in range(5,9):
		  Anms = np.uint64(Anms) & np.uint64(Shifts[ :, 4] > Shifts[ :, shiftind])

	#print(Anms)

	locmaxind = find(Anms)
	#print(locmaxind)

	#xx=np.meshgrid(A,A)
	#print(xx)
	ysize = np.size(A,1)
	xsize = np.size(A,0)
	Col=np.zeros((xsize,ysize))
	Row=np.zeros((xsize,ysize))	


	for i in range(0,xsize):
		for j in range(0,ysize):
			Col[i][j]=j+1;
			Row[i][j]=i+1;

	#print(Col)
	#print(Row)


	#Pos = [Row(locmaxind) Col(locmaxind)]
	#print(Pos)

	length = np.size(locmaxind)
	#print(length)
	
	pos = [[] for i in range(length)]
	
	
	for i in range(0,length):
		point = locmaxind[i]
		#print(point)
		c=point/xsize
		r=point%xsize
		#print(c,r)
	
		pos[i] = ([Row[c,r-1],Col[c,r-1]])	
		#print(pos)	
	
		Value.extend([A[c,r-1]])

	#print(np.size(pos))
	#print(np.size(Value))
	

	Anms = np.reshape( Anms, (np.size( A, 0), np.size( A, 1))) * A
	#print(Anms)

	pos1 = np.array(pos)
	pos = pos1-1
	position = []
	del position[:]
	#print(pos)
	position = pos
	#print(np.size(position))

	Anms=Anms[1:(np.size(Anms,0)-1),1:(np.size(Anms,1)-1)]
	#print(np.size(Anms))
	return position




cresult = []

def crop2(data,ny,nx):
	ysize = np.size(data, 0)
	xsize = np.size(data, 1)
	#print(ysize)
	#print(xsize)

	cresult=data[ny:ysize-ny-1,nx:xsize-nx-1];
    	return cresult

result = []

def extend2(data,ny,nx):
	ysize = np.size(data, 0)
	xsize = np.size(data, 1)
	#print(ysize,xsize)
		
	newxsize = xsize+2*nx
	newysize = ysize+2*ny
			
	result = np.zeros((newysize,newxsize))
		
	result[ny:ysize+ny,nx:xsize+nx]=data
		
		
	for x in range(0,nx): 
		result[:,x]=result[:,nx]
		

	for x in range(xsize+nx,newxsize):
		result[:,x]=result[:,xsize+nx-1]
		

	for y in range(0,ny):
		result[y,:]=result[ny,:]
		

	for y in range(ysize+ny,newysize):
		result[y,:]=result[ysize+ny-1,:] 
	#print(result)
		
	return result

pixels = []

def mydiscgaussfft(inpic, sigma2):
	ftransform = np.fft.fft(inpic, 256, 0) 
	ftransform = np.fft.fft(ftransform, 256, 1) 
	xsize = np.size(ftransform, 0)
	ysize = np.size(ftransform, 1)
	
	#print(xsize)
	#print(ysize)
	
	#x, y= meshgrid[0 : xsize-1, 0 : ysize-1]
	#print(x)
	
	x=np.zeros(shape=(xsize,ysize))
	y=np.zeros(shape=(xsize,ysize))	

	for i in range(0,xsize):
		for j in range(0,ysize):
			x[i][j]=j;
			y[i][j]=i;
	#print(x)
	#print(y)

	a = cos(2 * pi*(x / xsize))
	b = cos(2 * pi*(y / ysize))
	c = exp(sigma2 * ( a + b - 2))
	p = c.transpose()
	d = p*ftransform
	pixels = np.fft.ifft(d, 264, 0)
	pixels = real(np.fft.ifft(pixels,264,1))
	#print(pixels)

	return pixels

#val = []


def STIP(img):
	#img = cv2.imread('frame1.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height = np.size(img, 0)
	width = np.size(img, 1)

	sxl2 = 4
	sxi2 = 2*sxl2
	kparam = 0.04
	npoints = 40

	L = mydiscgaussfft(extend2(img,4,4),sxl2)
	#print(L)

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
	print(cimg[0])

	position = maxsupression(cimg)
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
	np.savetxt('CheckSet.txt',si)
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
	val = []

	#print(px1)
	#print(py1)
        
	for i in range(0,np.size(px1)):
		point = px1[i]-1	
		px.append(pxall[point])

	for i in range(0,np.size(py1)):
		point = py1[i]-1	
		py.append(pyall[point])

	

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
		x = shape * (px[i])
		x = x + (py[i]+1)
		ind.append(x) 
	#print(np.size(ind))

	c11 = []
	c12 = []
	c22 = []
	
	for i in range(0,np.size(ind)):
		point = ind[i]-1
		c=point/np.size(Lxm2smooth,0)
		r=point%np.size(Lxm2smooth,0)	
		c11.append([Lxm2smooth[c,r-1]])
	#print(c11)

	for i in range(0,np.size(ind)):
		point = ind[i]-1
		c=point/np.size(Lxmysmooth,0)
		r=point%np.size(Lxmysmooth,0)	
		c12.extend([Lxmysmooth[c,r-1]])	
	#print(c12)

	for i in range(0,np.size(ind)):
		point = ind[i]-1
		c=point/np.size(Lym2smooth,0)
		np.int(c)
		r=point%np.size(Lym2smooth,0)
		np.int(r)	
		c22.extend([Lym2smooth[c,r-1]])		
	#print(c22)

	posinit = []
	#del posinit[:]

	posinit=[px,py,sxl2*np.ones(size(px)),c11,c12,c12,c22]
	#print(np.size(posinit[0]))

	return val,posinit


def il_rgb2gray(rimage):
	x = rimage.ndim
	#print(x)
	#print(rimage)
	if x<3:
		gray = rimage
	else:
		gray = sum(rimage,2)/np.size(rimage,2) 
		#print(gray)
	return gray


#-------------------------------#
#------------ main -------------#
#------------------------------


#--- Load Videos and Frame Conversion ----#

from Tkinter import Tk
from tkFileDialog import askopenfilename


Tk().withdraw() 
vid = askopenfilename()
cap = cv2.VideoCapture(vid)


success,image = cap.read()
os.system("rm Frame/*.jpg")
count = 1
Frame = 1

while success:
	success,image = cap.read()
	cv2.imwrite("Frame/frame%d.jpg" % count, image)     
	if count == Frame:                     
    		break
	count += 1


		#RESIZE#

for i in range(1,Frame+1):
	rimage = cv2.imread("Frame/frame%d.jpg" % i)  
	image = cv2.resize(rimage,(256,256))
	cv2.imwrite("Frame/frame%d.jpg" % i, image)



#--- Load Videos and Frame Conversion END----#


#------------ Processing Frame --------------#

		#RGB2GRAY#

#Valinit = [[0] * 39] *400
Valinit = [[] for i in range(Frame)]
#y = [[] for i in range(90)]

for i in range(1,Frame+1):
	rimage = cv2.imread("Frame/frame%d.jpg" % i)
	image = il_rgb2gray(double(rimage))  
	#image = cv2.cvtColor(rimage, cv2.COLOR_RGB2GRAY)
	cv2.imwrite("FrameG/frame%d.jpg" % i, image)
 

	print("Process No: ",i)
	valin,posinit = STIP(image)
	#print(np.size(valin))	
	
	Valinit[i-1] = valin
	#print(Valinit)
	

	plt.imshow(rimage)	
	showellipticfeatures(posinit,[1 ,1 ,0])
	#print(np.size(posinit[0]))
	plt.draw()
	plt.axis('off')
	plt.pause(0.001)
	#plt.show()
	
plt.ion()
fig = plt.figure()
plt.show()	

#print(Valinit[0])
#print(Valinit[1])


#------------ Processing Frame END ----------#

#np.save('TrainingSet.npy',Valinit)
#print(y)

cap.release()
cv2.destroyAllWindows()
















