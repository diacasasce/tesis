# LIBRERIAS
import numpy as np
import cv2
from time import sleep
import picamera
import funciones as fun
#inicializacion de variables
# acomodacion de camara
#camera.start_preview()
#sleep(1)
#camera.stop_preview()
#<<<<<<<<<<<<<<<<<< inicio calibracion >>>>>>>>>>>>>>>>>
inp='';
while(inp.upper()!='Y'):
    i=1
    name='cal1.jpg'
#    camera.capture(name,0)
    Im= cv2.imread(name)
#    cv2.imshow('imagen',Im)
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
    warp=fun.encuadre(gray,Im)
    (war,(xf,yf))=fun.calibracion(Im)
    #cv2.imshow('CAL',war)
    cv2.waitKey(0);
    print(xf)
    print(yf)

    #inp=input('la cuadricula esta bien ubicada? y/n  ')
    inp='y'
    cv2.destroyAllWindows()
#<<<<<<<<<<<<<<<<<< fin calibracion >>>>>>>>>>>>>>>>>    

print('/----------------------/')
print(xf)
print(yf)
print('/----------------------/')

#<<<<<<<<<<<<<<<<<< reconocimiento  >>>>>>>>>>>>>>>>>
# nececesito umbral adaptativo
reina=cv2.imread('imagenes/reina.png',0)
(w,h)=reina.shape
inp='';
i=2
MIN_MATCH_COUNT=10
#surf=cv2.xfeatures2d.SURF_create(400)
#kp1, des1=surf.detectAndCompute(reina,None)
#print (len(kp1))

#img2=cv2.drawKeypoints(reina,kp1,None,(0,200,0),4)
#cv2.imshow('fg',img2)
name='rec1.jpg'
    ##   camera.capture(name,0)
Im= cv2.imread(name)
cv2.imshow('imagen',Im)
gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY)
warp=fun.encuadre(gray,Im)
grw=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
#thresh de otsu
ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('h',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(0,8):
    for j in range(0,8):
        name='tile'+str(i)+'-'+str(j)+'.jpg'
        Cp=th[xf[i]+17:xf[i+1]+17,yf[j]-18:yf[j+1]-18]
        Im= cv2.imwrite(name,Cp)
        print(name)
        

