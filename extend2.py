# import function
from include import*



def extend2(data,ny,nx):
	ysize = np.size(data, 0)
	xsize = np.size(data, 1)
	#print(ysize,xsize)
		
	newxsize = xsize+2*nx
	newysize = ysize+2*ny
	#print(newysize,newxsize) #(11, 12)
			
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



