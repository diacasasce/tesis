from MLP import *
import cv2
import numpy as np
inp=np.array([[1,0],[0,1],[0,0],[1,1]])
out=np.array([[1],[1],[0],[0]])
red=mlp(2,np.array([20]),1)