# import function
from include import*
from showellipticfeatures import *
from il_rgb2gray import *
from STIPEx import *
from STIP import *


#-------------------------------#
#------------ main -------------#
#------------------------------


#--- Load Videos and Frame Conversion ----#


#---------------1st-----------------------#

os.system("rm Frame/*.jpg")
count = 1
Frame = 35
Frame1 = Frame
rang = 5

for t in range(0,rang):
	Tk().withdraw() 
	vid = askopenfilename()
	cap = cv2.VideoCapture(vid)

	success,image = cap.read()

	while success:
		success,image = cap.read()
		cv2.imwrite("Frame/frame%d.jpg" % count, image)     
		if count == Frame:                     
				break
		count += 1
	Frame+=Frame1

Frame-=Frame1

			#RESIZE#

for i in range(1,Frame+1):
	rimage = cv2.imread("Frame/frame%d.jpg" % i)  
	image = cv2.resize(rimage,(256,256))
	cv2.imwrite("Frame/frame%d.jpg" % i, image)
#y[i]=1
p=0
y = [[] for i in range(Frame)]

for j in range(1,rang+1):
	for i in range(Frame1):
			y[p]=int(j)
			p+=1

np.save('TrainingRowE2.npy',y)
np.savetxt('TrainingRowE2.txt',y)
#print(y)
#--- Load Videos and Frame Conversion END----#


#------------ Processing Frame --------------#


Valinit = [[] for i in range(Frame)]

for i in range(1,Frame+1):
	rimage = cv2.imread("Frame/frame%d.jpg" % i)  
	image = cv2.cvtColor(rimage, cv2.COLOR_RGB2GRAY)
	cv2.imwrite("Frame/frame%d.jpg" % i, image)

	print("Process No: ",i)
	valin,posinit,dst = STIPEx(image)
	#valin,dst = STIP(image)
	#dst = STIPH(image)
	#print(np.size(valin))	
	
	Valinit[i-1] = valin
	#print(Valinit)

    #'''
	dst = cv2.dilate(dst,None)
	rimage[dst>0.01*dst.max()]=[0,0,255]

	cv2.imshow('dst',rimage)
	cv2.waitKey(1) 
	#'''

    
	
		
	

plt.ion()
fig = plt.figure()
plt.show()

#------------ Processing Frame END ----------#


np.save('TrainingSetE2.npy',Valinit)
np.savetxt('TrainingSetE2.txt',Valinit)

#np.save('TrainingSet.npy',Valinit)
#np.savetxt('TrainingSet.txt',Valinit)

#d = np.loadtxt('TrainingSet.npy')
#print(d)


cap.release()
cv2.destroyAllWindows()


