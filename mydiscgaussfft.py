# import function
from include import*

def mydiscgaussfft(inpic, sigma2):
	asize = np.size(inpic,0)
	ftransform = np.fft.fft(inpic, asize, 0) 
	ftransform = np.fft.fft(ftransform, asize, 1) 
	#print(ftransform)

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
	#print(p)
	pixels = np.fft.ifft(d, asize, 0)
	pixels = real(np.fft.ifft(pixels,asize, 1))
	#pixels = np.fft.ifft(d, 264, 0)
	#pixels = real(np.fft.ifft(pixels,264,1))
	#print(pixels)

	return pixels
