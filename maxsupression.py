# import function
from include import*
from pad import *



def maxsupression(A):
	Values = []
	del Values[:]
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
		Y1 = np.reshape( z,( 3, 3))
		Y = Y1.transpose()
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
	#np.savetxt('TrainingSet.txt',Shifts)
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
			Row[i][j]=j+1;
			Col[i][j]=i+1;

	#print(Col)
	#print(Row)


	#Pos = [Row(locmaxind) Col(locmaxind)]
	#print(Pos)

	length = np.size(locmaxind)
	#print(length)

	pos = [[] for i in range(length)]

		
	for i in range(0,length):
		point = locmaxind[i]+1
		#print(point)
		c=point/xsize
		r=point%xsize
		#print(c,r)
		
		pos[i] = ([Row[c,r-1],Col[c,r-1]])	
		#print(pos)	
		a = int(Row[c,r-1]-1)
		b = int(Col[c,r-1]-1)

		Values.extend([A[a,b]])

	#print(pos)
	#print(Values)
		

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

	return position,Values
