import numpy as np
import cv2
from time import sleep
import picamera
from funciones import *

#inicializacion de variables
camera=picamera.PiCamera()
camera.vflip=True
camera.brightness=50

#inicializacion de variables
 #acomodacion de camara

#camera.start_preview()
#sleep(30)
#camera.stop_preview()
#<<<<<<<<<<<<<<<<<< inicio calibracion >>>>>>>>>>>>>>>>>
inp=''
i=0
while(inp.upper()!='Y'):
    i=i+1
    name='caln'+str(i)+'.jpg'
    camera.capture(name,0)
    Im= cv2.imread(name)
    cv2.imshow('imagen',Im)
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
    cv2.imwrite('inicial.jpg',gray)
    gray=correccion_luz(gray)
    cv2.imwrite('correccionluz.jpg',gray);
    bin1=define_tablero(gray)
    cv2.imwrite('definetablero.jpg',bin1);
    Mask=no_fondo(bin1,Im)
    _,cont,_=cv2.findContours(Mask.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    print(cont)
    cv2.imshow('no_fondo.jpg',Mask)
    cv2.imwrite('no_fondo.jpg',Mask)
    corners = cv2.goodFeaturesToTrack(Mask,20,0.01,10)
    corners = np.int0(corners)
    crd=[]
    rg=[[[200, 300],[100, 200]],[[780 , 880],[60, 160]],[[80 , 180],[660 , 760]],[[920 , 1020],[680 , 780]]]
    crn=0*Mask;
    for i in corners:
        x,y = i.ravel()
        if (rg[0][0][0]<x<rg[0][0][1] and rg[0][1][0]<y<rg[0][1][1])or (rg[1][0][0]<x<rg[1][0][1] and rg[1][1][0]<y<rg[1][1][1])or (rg[2][0][0]<x<rg[2][0][1] and rg[2][1][0]<y<rg[2][1][1])or (rg[3][0][0]<x<rg[3][0][1] and rg[3][1][0]<y<rg[3][1][1]):
            print((x,y))
            cv2.circle(crn,(x,y),5,(255),-1)
            crd.append((x,y))
    coor=ordena(crd)
    cv2.imwrite('esquinas.jpg',crn)
    cv2.imshow('esquinas.jpg',crn)
    warp=perspectiva(Im,coor)
    print(warp.shape)
    (r,c,d)=warp.shape
    MR=cv2.getRotationMatrix2D((c/2,r/2),-90,1)
    warp=cv2.warpAffine(warp,MR,(c,r))
    cv2.imwrite('final.jpg',warp)
    cv2.imshow('final.jpg',warp)
    cv2.waitKey(0);
#    print(xf)
#    print(yf)
    #inp=input('la cuadricula esta bien ubicada? y/n  ')
    inp='y'
    cv2.destroyAllWindows()
    
