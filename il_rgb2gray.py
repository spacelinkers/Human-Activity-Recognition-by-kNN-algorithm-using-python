# import function
from include import*

def il_rgb2gray(rimage):
	x = rimage.ndim
	#print(rimage[:,:,0])
	#r,g,b = rimage[:,:,0],rimage[:,:,1],rimage[:,:,2]
	if x<3:
		gray = rimage
	else:
		gray = rimage.sum(axis=2)
		b = np.size(rimage,2)
		gray = gray/b
		#print(gray)
		#gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray
