# import function
from include import*
from drawellipse_cov import *




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
	
	for i in range(np.size(x1)):
		x0 = x1[i]
		y0 = x2[i]
		s0 = x3[i]
		Sigma = np.array([[x4[i] ,x5[i]] ,[x6[i] ,x7[i]]],np.float)
		Sigma = inv(Sigma)
		#x = np.linalg.det(Sigma)
		#A = np.array(Sigma.data).reshape([3,3])
		(sign,logdet) = np.linalg.slogdet(Sigma)
		x = sign*np.exp(logdet)
		#print(x)
		#if x<0: x=x*-1
		Sigma = Sigma/np.sqrt(x)
		#print(Sigma)
		if x>=0:
			h = drawellipse_cov(Sigma*s0*10,[x0,y0])
		#h = plt.plot(x0,y0,color = fcol)	
		#plt.close()
		
