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
    cv2.imshow('CAL',war)
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
reina=cv2.imread('imagenes/peon.png',0)
hum=cv2.HuMoments(cv2.moments(reina)).flatten()
print(hum)
print('--')
#reina=cv2.imread('tile1-4.jpg',0)
cv2.imshow('fge',reina)
(w,h)=reina.shape
inp='';
i=2
MIN_MATCH_COUNT=10
surf=cv2.xfeatures2d.SURF_create(1000)
kp1, des1=surf.detectAndCompute(reina,None)
print (len(kp1))

img2=cv2.drawKeypoints(reina,kp1,None,(0,200,0),4)
cv2.imshow('fg',img2)
while(inp.upper()!='Y'):
    name='rec'+str(i)+'.jpg'
##   camera.capture(name,0)
    Im= cv2.imread(name)
    cv2.imshow('imagen',Im)
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY)
    warp=fun.encuadre(gray,Im)
    grw=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    #thresh de otsu
    ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#    th=th[0:xf[2]+17,yf[0]-17:yf[4]-17]
    sth=th.shape
#    th=cv2.resize(th,(2*sth[1],2*sth[0]))
    sth=th.shape
    back=np.ones(sth)*255
    
    _,cont,_=cv2.findContours(th.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    print(len(cont))
    q=0
    cc=[]
    humm=[]
    for c in cont:
        ar=cv2.contourArea(c)
        if ar>1000 and ar<2500:
            q+=1
            hum=cv2.HuMoments(cv2.moments(c)).flatten()
            c,radius = cv2.minEnclosingCircle(c) 
            ihu=[]
            for h in range(len(hum)):
                nh=1/hum[h]
                nh=round(float(nh),2)
                ihu.append(nh)
            if 4 < ihu[0] and ihu[0]<3000:
                if 60 < ihu[1]:
                    cc.append(c)
                    humm.append(ihu)
    print('--')
    print(q)
    print('*****')
    print(humm)
    print('*****')
    icon=cv2.drawContours(back,cc,-1,(00,120,120))
    cv2.imshow('fg1co',icon)    
    kp2, des2=surf.detectAndCompute(th,None)
    img3=cv2.drawKeypoints(th,kp2,None,(200,0,0),4)
    cv2.imshow('fg1',img3)
    bf = cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]
    for m,n in matches:
        if m.distance<2*n.distance:
            good.append([m])
    img4=cv2.drawMatchesKnn(reina,kp1,th,kp2,good,th.copy(),flags=2)
    cv2.imshow('fg2',img4)
    cv2.imwrite('c-peon.jpg',img4)
    cv2.waitKey(0)
    inp=input('FINALIZAR RECONOCIMIENTO? y/n  ')
    cv2.destroyAllWindows()
    i+=1
cv2.imshow('h',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
