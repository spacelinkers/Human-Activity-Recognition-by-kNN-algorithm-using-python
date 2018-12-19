# import function
from include import*
from showellipticfeatures import *
from il_rgb2gray import *
from STIPH import *
from STIP import *

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
Frame = 45

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


Valinit = [[] for i in range(Frame)]


for i in range(1,Frame+1):
	rimage = cv2.imread("Frame/frame%d.jpg" % i)
	#image = il_rgb2gray(double(rimage)) 
	image = cv2.cvtColor(rimage, cv2.COLOR_RGB2GRAY)
	#image = np.float32(image) 

	cv2.imwrite("FrameG/frame%d.jpg" % i, image)
 

	print("Process No: ",i)
	valin,posinit,dst = STIP(image)
	#dst = STIPH(image)
	#print(np.size(valin))	
	
	#Valinit[i-1] = valin
	#print(Valinit)
	#'''
	dst = cv2.dilate(dst,None)
	rimage[dst>0.01*dst.max()]=[0,0,255]

	cv2.imshow('dst',rimage)
	cv2.waitKey(1) 
	#'''
	
	'''
	plt.imshow(rimage)	
	showellipticfeatures(posinit,[1 ,1 ,0])
	#print(np.size(posinit[0]))
	plt.draw()
	plt.axis('off')
	plt.pause(0.001)
	'''
	
plt.ion()
fig = plt.figure()
plt.show()	


#------------ Processing Frame END ----------#

#np.save('TrainingSet.npy',Valinit)
#print(y)

cap.release()
cv2.destroyAllWindows()
















