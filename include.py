import cv2
import os,sys
import time
import numpy as np
import scipy as sc
from pylab import *
from scipy import signal
from numpy.linalg import inv
from scipy.linalg import sqrtm 
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy.signal import convolve2d
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from Tkinter import Tk
from tkFileDialog import askopenfilename
