# import function
from include import*
from mydiscgaussfft import *
from dxmsk import *
from conv2 import *
from crop2 import *
from extend2 import *
from maxsupression import *


def STIPEx(img):
	#img = cv2.imread('frame1.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	height = np.size(img, 0)
	width = np.size(img, 1)

	sxl2 = 4
	sxi2 = 2*sxl2
	kparam = 0.04
	npoints = 40


	L = mydiscgaussfft(extend2(img,4,4),sxl2)

	L = mydiscgaussfft(L,sxl2)
	L = mydiscgaussfft(L,sxl2)
	L = mydiscgaussfft(L,sxl2)

	#np.savetxt('TrainingSet.txt',L)
	#print(L)
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
	#print(cimg[:,255])
	#print(cimg)

	#cimg = cv2.cornerHarris(img,2,3,0.04)
	#cv2.imshow('HumanCimg',cimg)
	#print(np.size(img))

	position,Value = maxsupression(cimg)
	#print(position)
	#print(Value)
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
		point = px1[i]	
		px.append(pxall[point])

	for i in range(0,np.size(py1)):
		point = py1[i]	
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
		x = shape * (py[i]-1)
		x = x + px[i]
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
	del posinit[:]

	#posinit=[px,py,sxl2*np.ones(size(px)),c11,c12,c12,c22]
	sx = sxl2*np.ones(size(px))
	#print(posinit)
	insideind = [[] for i in range(np.size(px))]
	npx = []
	npy = []
	nsx = []
	nc11 = []
	nc12 = []
	nc22 = []
	#del insideind[:]
	p = 0
	if 1 :
		bound=2
		for i in range(np.size(px)):
			if (px[i]>bound):
				x1 = 1
			else:
				x1 = 0
			if (px[i]<(shape-bound)):
				x2 = 1
			else:
				x2 = 0
			if (py[i]>bound):
				x3 = 1
			else:
				x3 = 0
			if (py[i]<(shape-bound)):
				x4 = 1
			else:
				x4 = 0
			insideind[p] = (x1*x2*x3*x4)
			p = p+1
		#print(insideind)

		for i in range(np.size(insideind)):
			if insideind[i] == 1:
				#print(i)
				npx.append(px[i])
				npy.append(py[i])
				nsx.append(sx[i])
				nc11.append(c11[i]) 
				nc12.append(c12[i])
				nc12.append(c12[i])
				nc22.append(c22[i])
		#posinit=posinit[insideind,:]
		
	posinit=[npx,npy,nsx,nc11,nc12,nc12,nc22]
	#print(posinit)

	return val,posinit,cimg









