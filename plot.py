# import function
from include import*
from showellipticfeatures import *

def change_plot(rimage,posinit):
    #fig = plt.figure()
    plt.suptitle('Corner Points', fontsize=14, fontweight='bold')
    #ax1 = fig.add_subplot(1,1,1)

    plt.cla()
    plt.imshow(rimage,showellipticfeatures(posinit,[1 ,1 ,0]))  
    plt.axis('off')
    plt.pause(0.001)

    #ani = animation.FuncAnimation(h, animate, interval=50, frames=100)
    #plt.tight_layout()
    #plt.show()

    #plt.close()