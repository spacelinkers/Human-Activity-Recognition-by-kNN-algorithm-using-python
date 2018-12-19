# import function
from include import*


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
	#print(ell)
	ell[:,1]=ell[:,1]+poss[1]
	#print(ell)
	
	h=plt.plot(ell[:,0],ell[:,1], lineWidth = 1.5, color = 'w')
	#plt.show()
	#plt.close()
	return h
