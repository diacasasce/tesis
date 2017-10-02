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
    bin=M-bin
    bin=cv2.dilate(bin,None)
    bin=cv2.dilate(bin,None)
    bin=cv2.dilate(bin,None)
    bin=cv2.erode(bin,None)
    M=np.amax(bin)
    bin1=M-bin
    return bin1;
def no_fondo(bin1):
    bin,contours,hier=cv2.findContours(bin1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.ones(Im.shape[:2],dtype='uint8')*255
    print(range(len(contours)))
    for c in contours:
        if no_vaa(c,300000):
            cv2.drawContours(mask,[c],-1,0,-1)
    Mask=np.amax(mask)- mask
    return Mask

def no_va(c):
    ar=cv2.arcLength(c,True)
    app=cv2.approxPolyDP(c,0.02*ar,True)
    print(ar)
    return not len(app)==4;
    
def no_vaa(c,area):
    ar=cv2.contourArea(c)
    chk=(ar<area)
    return not chk;
def blob_coor(bin1):
    bin1=np.uint8(bin1)
    bin,contours,hier=cv2.findContours(bin1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    print(range(len(contours)))
    coor=[]
    ra=range(len(contours))
    for i in ra:
        rc=cv2.minAreaRect(contours[i])
        box=cv2.boxPoints(rc)
        pt=(0,0)
        for p in box:
            pt=(p[0],p[1])
        coor.append(pt)
    while len(coor)<4:
        pt=(0,0)
        coor.append(pt)
    return coor
def ordena(coor):
    ra=range(len(coor))
    for i in ra:
        V=len(coor)-i
        for j in range(1,V):
            da2=coor[j-1]
            da2=da2[0]+da2[1]
            da1=coor[j]
            da1=da1[0]+da1[1]
            if da2>da1:
                ct=coor[j-1]
                coor[j-1]=coor[j]
                coor[j]=ct
    print(coor)
    return coor
def perspectiva(Im,coor):
    coor=ordena(coor)
    print(coor)
    tl=coor[0]
    tr=coor[1]
    bl=coor[2]
    br=coor[3]
    pts=np.array((tl,tr,br,bl),dtype=np.float32)
    dst=np.array(((0,0),(900,0),(900,900),(0,900)),dtype=np.float32)
    M=cv2.getPerspectiveTransform(pts,dst)
    warp=cv2.warpPerspective(Im,M,(880,900))
    return warp
#inicializacion de variables
camera=picamera.PiCamera()
camera.vflip=True
camera.brightness=55
camera.start_preview()
sleep(2)
camera.stop_preview()
camera.capture('cap2.jpg',0)
Im= cv2.imread('cap2.jpg')
cv2.imshow('imagen',Im)
gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
gray=correccion_luz(gray)
bin1=define_tablero(gray)
Mask=no_fondo(bin1)
can3=np.float32(Mask)
crn=cv2.cornerHarris(can3,10,3,0.18)
crn=cv2.dilate(crn,None)
crn=cv2.dilate(crn,None)
crn=cv2.dilate(crn,None)
c=crn>0.008*crn.max()
cv=crn*c
cv2.imshow('puntos',crn)
cv2.imshow('punto',cv)
coor=blob_coor(cv)
print(coor)
warp=perspectiva(Im,coor)
cv2.imshow('im1',warp)
print('---')
print(coor)
Im[crn>0.01*crn.max()]=[0,200,0]
cv2.imshow('esquinas',Im)
cv2.waitKey(0)
cv2.destroyAllWindows()
