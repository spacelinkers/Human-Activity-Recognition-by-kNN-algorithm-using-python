import cv2
import os,sys
import time
import numpy as np
import scipy as sc
from pylab import *
from scipy import signal
from numpy.linalg import inv
from scipy.linalg import sqrtm 
from scipy.ndimage import filters
from scipy.signal import convolve2d
from sklearn.neighbors import KNeighborsClassifier

# import function
from showellipticfeatures import *
from il_rgb2gray import *
from STIP import *

#-------------------------------#
#------------ main -------------#
#-------------------------------#


#--- Load Videos and Frame Conversion ----#

from Tkinter import Tk
from tkFileDialog import askopenfilename


#------------ Processing Frame --------------#

Frame = 10000

Valinit = [[] for i in range(Frame)]
y = [[] for i in range(240)]
for i in range(0,81):
    	y[i]=1

for i in range(81,161):
    	y[i]=2

for i in range(161,240):
    	y[i]=3
#print(y)

cont1 = 0
cont2 = 0
cont3 = 0
xx = 0
d = np.load('TrainingSet.npy')
knn = KNeighborsClassifier()


Tk().withdraw() 
vid = askopenfilename()
cap = cv2.VideoCapture(vid)
#vid = cv2.VideoCapture('brush_sudipta.mp4')
#vid = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    rimage = cv2.resize(frame,(256,256))
    image = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)


    valin,dst = STIP(image)
    Valinit = valin
    #print(valin)

    dst = cv2.dilate(dst,None)
    rimage[dst>0.01*dst.max()]=[0,0,255]

    d = np.load('TrainingSet.npy')
    knn = KNeighborsClassifier()
    
    md = knn.fit(d,y)
    #print(md)
    ans = knn.predict(Valinit)
    #print(ans)
    ans1 = knn.predict_proba(Valinit)
    #print(ans1)

    xx = xx+1
    #print(ans)
    for i in range(0,np.size(ans)):
        if ans[i]==1:
            cont1+=1
        elif ans[i]==2:
                cont2+=1
        else:
            cont3+=1
    ct1 = cont1*100/xx
    ct2 = cont2*100/xx
    ct3 = cont3*100/xx
    #print(cont1)
    #print(cont2)
    #print(cont3)
    #print(xx)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if ct1>50:
        print 'Hands-Up'
        cv2.putText(rimage,'Hands-Up',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)
    elif ct2>50:
        print 'Brushing'
        cv2.putText(rimage,'Brushing',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)
    elif ct3>50:
        print 'Up-Down'
        cv2.putText(rimage,'Up-Down',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)
    else:
        print 'Not Recognized'
        cv2.putText(rimage,'Not Recognized',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)

    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('Human Activity Recognition',rimage,)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
	
plt.ion()
fig = plt.figure()
plt.show()	


#------------ Processing Frame END ----------#

cap.release()
cv2.destroyAllWindows()

















