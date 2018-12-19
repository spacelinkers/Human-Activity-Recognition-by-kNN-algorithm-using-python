# import function
from include import*
from showellipticfeatures import *
from il_rgb2gray import *
from STIPEx import *
from STIP import *
from plot import *



#-------------------------------#
#------------ main -------------#
#-------------------------------#


#--- Load Videos and Frame Conversion ----#



cont1 = 0
cont2 = 0
cont3 = 0
cont4 = 0
cont5 = 0
xx = 0
anss = []
d = np.load('TrainingSetE2.npy')
y = np.load('TrainingRowE2.npy')
#print(y)
knn = KNeighborsClassifier()

Tk().withdraw() 
vid = askopenfilename()
cap = cv2.VideoCapture(vid)
#vid = cv2.VideoCapture(0)

#------------ Processing Frame --------------#

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    rimage = cv2.resize(frame,(256,256))
    #print(rimage[:,:,0])
    #print(rimage[:,:,1])
    #print(rimage[:,:,2])
    #image = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)
    image = il_rgb2gray(double(rimage))
    #print(image)
    '''
    image =[[133.6667,  129.6667,  149.6667,  137.6667,  145.6667,  146.6667,  114.0000,  136.0000,  121.0000,  120.0000],
            [146.6667,  134.6667,  152.6667,  152.6667,  153.6667,  169.6667,  155.0000,  136.0000,  123.0000,  126.0000],
            [149.6667,  149.6667,  160.6667,  152.6667,  114.6667,  122.6667,  150.0000,  119.0000,  103.0000,  106.0000],
            [143.6667,  166.6667,  165.6667,  158.6667,  104.6667,   77.6667,  112.0000,   90.0000,  109.0000,  103.0000],
            [156.6667,  190.6667,  175.6667,  188.6667,  162.6667,  105.6667,  112.0000,  100.0000,   99.0000,   81.0000],
            [115.6667,  152.6667,  135.6667,  139.6667,  113.6667,   55.6667,   71.0000,   91.0000,   86.0000,   64.0000],
            [100.0000,  126.0000,  124.0000,  110.0000,   58.0000,   22.0000,   61.6667,   89.6667,   90.6667,   75.6667],
            [ 76.0000,   83.0000,  100.0000,   92.0000,   36.0000,    24.0000,   55.6667,   45.6667,   55.6667,  48.6667],
            [ 42.0000,   47.0000,   48.0000,   46.0000,   46.0000,   50.0000,   52.6667,   51.6667,   55.6667,   50.6667],
            [ 57.0000,   62.0000,   63.0000,   60.0000,   60.0000,   64.0000,   65.6667,   64.6667,   56.6667,   50.6667]]
    '''
    valin,posinit,dst = STIPEx(image)
    Valinit = [[] for i in range(np.size(valin))]
    #valini,dst = STIP(image)
    Valinit = valin
    #print(posinit)
    #print(Valinit)
    #np.savetxt('TrainingSet.txt',Valinit)


    #dst = cv2.dilate(dst,None)
    #rimage[dst>0.01*dst.max()]=[0,255,255]
    #print(dst)
    

    
    #--------------PLOT-----------------#

    change_plot(rimage,posinit)
    
    #-------------------PLOT CLOSE------------#
    
    
    #--------Circle Draw---------#
    '''
    x1 = np.array(posinit[0])	
    x2 = np.array(posinit[1])	
    x0 = [[0 for col in range(2)]for row in range(np.size(x1))]
    for i in range(np.size(x1)):
        x0[i][0] = int(x1[i])
        x0[i][1] = int(x2[i])
    for point in x0:
        cv2.circle(rimage,(point[0],point[1]),5,(0,0,225),-1)
    '''

    #d = np.load('TrainingSet.npy')
    #knn = KNeighborsClassifier()
    
    md = knn.fit(d,y)
    #print(md)
    ans = knn.predict(Valinit)
    anss.append(ans[0])
    print(ans)
    ans1 = knn.predict_proba(Valinit)
    #print(ans1)
    y1 = [2]
    #ans2 = accuracy_score(y1,ans)
    #print(ans2)
    ans3 = knn.score(Valinit,y1)
    #print(ans3)

    xx = xx+1
    #print(xx)
    for i in range(0,np.size(ans)):
        if ans[i]==1:
            cont1+=1
        elif ans[i]==2:
            cont2+=1
        elif ans[i]==3:
            cont3+=1
        elif ans[i]==4:
            cont4+=1
        else:
            cont5+=1
    ct1 = cont1*100/xx
    ct2 = cont2*100/xx
    ct3 = cont3*100/xx
    ct4 = cont4*100/xx
    ct5 = cont5*100/xx

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
    elif ct4>50:
        print 'Walking'
        cv2.putText(rimage,'Walking',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)
    elif ct5>50:
        print 'Running'
        cv2.putText(rimage,'Running',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)
    else:
        print 'Not Recognized'
        cv2.putText(rimage,'Not Recognized',(10,250), font, .5, (200,255,155), 1, cv2.LINE_AA)

    if(xx==60):
        print(anss)
        break
    print(ans)
    print(xx)
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow('Human Activity Recognition',rimage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
	
plt.ion()
fig = plt.figure()
plt.show()	


#------------ Processing Frame END ----------#

cap.release()
cv2.destroyAllWindows()
