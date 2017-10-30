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
def no_area(bin1,area):
    bin1=np.uint8(bin1)
    bin,contours,hier=cv2.findContours(bin1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.ones(bin1.shape[:2],dtype='uint8')*255
    print(range(len(contours)))
    for c in contours:
        if no_vaa(c,area):
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
    box=[]
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
def blob_coor(bin1):
    print('----------coordenadas ------ ')
    bin1=np.uint8(bin1)
    bin,contours,hier=cv2.findContours(bin1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    coor=[]
    ra=range(len(contours))
    for i in ra:
        rc=cv2.minAreaRect(contours[i])
        box=cv2.boxPoints(rc)
        x=int((np.amax(box[:,0])+np.amin(box[:,0]))/2)
        y=int((np.amax(box[:,1])+np.amin(box[:,1]))/2)
        pt=(x,y)
        print(pt)
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
    tl=coor[0]
    tr=coor[1]
    bl=coor[2]
    br=coor[3]
    pts=np.array((tl,tr,br,bl),dtype=np.float32)
    dst=np.array(((0,0),(900,0),(900,900),(0,900)),dtype=np.float32)
    M=cv2.getPerspectiveTransform(pts,dst)
    warp=cv2.warpPerspective(Im,M,(900,900))
    return warp
def anclas_cuad(warp):
    can=cv2.Canny(warp,100,400)
    cv2.imshow('can',can)
    can3=np.float32(can)
    crn=cv2.cornerHarris(can3,10,5,0.18)
    crn=cv2.dilate(crn,None)
    crn=cv2.dilate(crn,None)
    rn=no_area(crn,70)
    c=rn>0.01*rn.max()
    cv=rn*c
    coor=blob_coor(cv.copy())
    print(coor)
    war=warp
    war[rn>0.01*rn.max()]=[255]
    IMG=warp[0:900, 0:900]
    cv2.imshow('esquinas',IMG)
    return coor
def encuadre(gray):
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
    coor=blob_coor(cv)
    print('--')
    print(coor)
    print('--')
    warp=perspectiva(Im,coor)
    return warp
def cuad_coord(coor):
    ra=range(len(coor))
    x=[]
    y=[]
    for i in ra:
        d=coor[i]
        x=np.append(x,d[0])
        y=np.append(y,d[1])
    ra=range(len(x))
    for i in ra:
        v=len(x)-i
        for j in range(1,v):
            tmp=x[j-1]
            if tmp>x[j]:
                x[j-1]=x[j]
                x[j]=tmp
    ra=range(len(y))
    for i in ra:
        v=len(y)-i
        for j in range(1,v):
            tmp=y[j-1]
            if tmp>y[j]:
                y[j-1]=y[j]
                y[j]=tmp
    return (x,y)
def seg_coord(x,dif):
    xf=[x[0]]
    ln=len(x)
    k=1;
    print(len(xf))
    print(xf[0])
    for i in range(1,ln):
        l=len(xf)-1
        if np.absolute(x[i-1]-x[i])>dif:
            print(k)
            xf[l]//=k
            xf.append(x[i])
            k=1        
        else:
            k=k+1
            xf[l]+=x[i]
            print(xf[l])
    l=len(xf)-1
    xf[l]//=k
    return xf
def print_cuad(xf,yf,img):
    rx=range(len(xf))
    ry=range(len(yf))
    for i in rx:
        cv2.line(img,(int(xf[i]),int(yf[0])),(int(xf[i]),int(yf[len(yf)-1])),(0,200,0),2)
    for i in ry:
        cv2.line(img,(int(xf[0]),int(yf[i])),(int(xf[len(xf)-1]),int(yf[i])),(0,200,0),2)
    cv2.imshow('cuad',img)
    return
def calibracion(Im):
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
    warp=encuadre(gray)
    war=warp.copy()
    cv2.imshow('w',warp)
    coor=anclas_cuad(warp)
    (x,y)=cuad_coord(coor)
    dif=50
    xf=seg_coord(x,dif)
    dif=50
    yf=seg_coord(y,dif)
    print(xf)
    print(yf)
    lx=len(xf)
    ly=len(yf)
    print((lx,ly))
    print_cuad(xf,yf,war)
    cv2.imshow('w',war)
    return(war,(xf,yf))

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
#(warp,coor)=calibracion(Im)
gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
warp=encuadre(gray)
war=warp.copy()
cv2.waitKey(0)
cv2.imshow('w',warp)

cv2.destroyAllWindows()
