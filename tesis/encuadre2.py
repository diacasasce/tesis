# LIBRERIAS
import numpy as np
import cv2
from time import sleep
import picamera
#FUNCIONES PROPIAS
def correccion_luz(img,cl=2.5,lgs=(8,8)) :
    clahe=cv2.createCLAHE(clipLimit=cl,tileGridSize=lgs)
    Ime=cv2.equalizeHist(img)
    Imc=clahe.apply(Ime)
    ker=np.matrix([[0,-1,0],[-1,5,-1],[0,-1,0]])
    Imc=cv2.filter2D(Imc,-1,ker)
    return Imc;
def define_tablero(gray):
    gray=cv2.GaussianBlur(gray,(5,5),0)
    _, bin=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    M=np.amax(bin)
    bin=cv2.dilate(bin,None)
    cv2.imshow('esquinas',bin)
    bin=M-bin
    bin=cv2.dilate(bin,None)
    bin=cv2.dilate(bin,None)
    bin=cv2.dilate(bin,None)
    cv2.imshow('esquin',bin)
    bin=cv2.erode(bin,None)
    cv2.imshow('esqui',bin)
    M=np.amax(bin)
    bin1=M-bin
    bing=bin1
    cv2.imshow('esqu',bin1)
    cv2.imshow('esq',bing)
    return bin1;
#inicializacion de variables
camera=picamera.PiCamera()
camera.vflip=True
camera.brightness=55
camera.start_preview()
sleep(5)
camera.stop_preview()
camera.capture('cap2.jpg',0)
Im= cv2.imread('cap2.jpg')
cv2.imshow('imagen',Im)
gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
gray=correccion_luz(gray)
cv2.imshow('ima',gray)
bin1=define_tablero(gray)
bin,contours,hier=cv2.findContours(bin1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print(range(len(contours)))
for i in range(len(contours)):
        rc=cv2.minAreaRect(contours[i])
        are=cv2.contourArea(contours[i])
        print(are)
        box=cv2.boxPoints(rc)
        for p in box:
            pt=(p[0],p[1])
            print (p)
            cv2.circle(Im,pt,10,(200,0,0),0)        
cv2.imshow('detect',Im)
cv2.waitKey(0)
cv2.destroyAllWindows()
