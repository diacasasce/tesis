# LIBRERIAS
import numpy as np
import cv2
from time import sleep
import picamera
#FUNCIONES PROPIAS
def correccion_luz(img,cl=2.5,lgs=(8,8)) :
    clahe=cv2.createCLAHE(clipLimit=cl,tileGridSize=lgs)
    Ime=cv2.equalizeHist(Im)
    Imc=clahe.apply(Ime)
    return Imc;
def detect_border(Im,th1=80,th2=200):
    ker=np.matrix([[1,1,-1],[1,1,-1],[-1,-1,-1]])
    Imd=cv2.filter2D(Im,-1,ker)
    Imd=cv2.filter2D(Imd,-1,ker)
    ker=np.matrix([[-1,-1,-1],[-1,1,1],[-1,1,1]])
    Imf=cv2.filter2D(Im,-1,ker)
    Imf=cv2.filter2D(Imf,-1,ker)
    Imt=Imf+Imd
    Imt=correccion_luz(Imt)
    ker=np.matrix([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]])
    Imh=cv2.filter2D(Imf+Imd,-1,ker)
    ker=np.matrix([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]])
    Imh=cv2.filter2D(Imh,-1,ker)
    ker=np.matrix([[1,1,-1],[1,0,-1],[-1,-1,-1]])
    Imd=cv2.filter2D(Imh,-1,ker)
    Imd=cv2.filter2D(Imd,-1,ker)
    ker=np.matrix([[-1,-1,-1],[-1,0,1],[-1,1,1]])
    Imf=cv2.filter2D(Imh,-1,ker)
    Imf=cv2.filter2D(Imf,-1,ker)
    Imt=Imf+Imh
    kex=np.matrix([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.uint8)
    key=np.matrix([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],dtype=np.uint8)
    idl=cv2.dilate(Imt,kex,iterations=1)
    idl=cv2.dilate(idl,key,iterations=1)
    idl=cv2.dilate(idl,kex,iterations=1)
    idl=cv2.dilate(idl,key,iterations=1)
    can=cv2.Canny(idl,th1,th2)
    Mi=np.amax(Imt)
    Imt=Mi-Imt
    kex=np.matrix([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,2,2,3,2,2,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],dtype=np.uint8)
    key=np.matrix([[0,0,1,0,0],[0,0,1,0,0],[0,0,2,0,0],[0,0,2,0,0],[0,0,2,0,0]],dtype=np.uint8)
    idl=cv2.erode(Imt,kex,iterations=2)
    idl=cv2.erode(idl,key,iterations=2)
    idl=cv2.dilate(idl,key,iterations=3)
    cv2.imshow('blu7',idl)
    can=cv2.Canny(idl,th1,th2)
    cv2.imshow('blue8',can)
    return Im;
#inicializacion de vatiables
camera=picamera.PiCamera()
camera.vflip=True
#camera.brightness=55
camera.start_preview()
sleep(1)
camera.stop_preview()
camera.capture('cap2.jpg',0)
Im= cv2.imread('cap2.jpg',0)
cv2.imshow('esquinas',Im)
Img=correccion_luz(Im)
cv2.imshow('esquias',Img)
Img=detect_border(Img,100,200)
cv2.imwrite('bordes.jpg',Img)
cv2.imshow('E',Img)
cv2.waitKey(0)
cv2.destroyAllWindows()
