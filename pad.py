# import function
from include import*

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



