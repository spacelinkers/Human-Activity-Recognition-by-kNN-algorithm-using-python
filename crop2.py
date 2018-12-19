# import function
from include import*


def crop2(data,ny,nx):
	ysize = np.size(data, 0)
	xsize = np.size(data, 1)
	#print(ysize)
	#print(xsize)

	cresult=data[ny:ysize-ny,nx:xsize-nx];
    	return cresult
