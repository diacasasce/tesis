from matplotlib import pyplot as plt
import numpy as np
import cv2
from time import sleep
import picamera
from funciones import *

camera=picamera.PiCamera()
camera.vflip=True
camera.hflip=True
camera.sharpness = 100
camera.contrast = 100
#inicializacion de variables
# acomodacion de camara
camera.start_preview()
sleep(10)
camera.stop_preview()
#<<<<<<<<<<<<<<<<<< inicio calibracion >>>>>>>>>>>>>>>>>
name='cd1.jpg'
camera.capture(name,0)
img= cv2.imread(name)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('inicial.jpg',gray)
gray=correccion_luz(gray)
cv2.imwrite('correccionluz.jpg',gray);
bin1=define_tablero(gray)
cv2.imwrite('definetablero.jpg',bin1);
Mask=no_fondo(bin1,img)
ot= cv2.cvtColor(Mask,cv2.COLOR_GRAY2BGR)
plt.imshow(Mask),plt.show()
corners = cv2.goodFeaturesToTrack(Mask,30,0.01,10)
corners = np.int0(corners)
crd=[]
for i in corners:
    x,y = i.ravel()
    print((x,y))
    #crd.append((x,y))
    cv2.circle(img,(x,y),5,(255),-1)
        
plt.imshow(img),plt.show()
    