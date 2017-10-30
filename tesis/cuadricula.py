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
    return Imc;

#inicializacion de variables
camera=picamera.PiCamera()
camera.vflip=True
#camera.brightness=55
camera.start_preview()
sleep(10)
camera.stop_preview()
camera.capture('cap2.jpg',0)
Im= cv2.imread('cap2.jpg',0)
cv2.imshow('esquinas',Im)
cv2.circle(Im,(280,160),10,(255,255,255),1)
cv2.circle(Im,(1045,130),10,(255,255,255),1)
cv2.circle(Im,(1225,905),10,(255,255,255),1)
cv2.circle(Im,(65,905),10,(255,255,255),1)
tl=(280,160)
tr=(1045,130)
br=(1225,905)
bl=(65,905)
pts=np.array((tl,tr,br,bl),dtype=np.float32)
dst=np.array(((0,0),(900,0),(900,900),(0,900)),dtype=np.float32)
M=cv2.getPerspectiveTransform(pts,dst)
warp=cv2.warpPerspective(Im,M,(880,900))
cv2.imshow('im1',warp)
ret2,th2=cv2.threshold(warp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('im1',th2)
can=cv2.Canny(th2,200,400)
cv2.imshow('im2',can)
can3=np.float32(can)
crn=cv2.cornerHarris(can3,10,5,0.18)
crn=cv2.dilate(crn,None)
key=np.matrix([[1,1,1],[1,0,1],[1,1,1]],dtype=np.uint8)
idl=cv2.erode(crn,key,iterations=2)
cv2.imshow('cuadros',crn)
warp[crn>0.01*crn.max()]=[255]
IMG=warp[50:900, 20:900]
cv2.imshow('esquinas',IMG)


cv2.waitKey(0)
cv2.destroyAllWindows()
