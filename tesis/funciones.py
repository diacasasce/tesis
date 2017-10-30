# LIBRERIAS
import numpy as np
import cv2
from time import sleep
import chess
from stockfish import Stockfish
#FUNCIONES
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
def no_fondo(bin1,Im):
    bin,contours,hier=cv2.findContours(bin1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.ones(Im.shape[:2],dtype='uint8')*255
    for c in contours:
        if no_vaa(c,300000):
            cv2.drawContours(mask,[c],-1,0,-1)
    Mask=np.amax(mask)- mask
    return Mask
def no_area(bin1,area):
    bin1=np.uint8(bin1)
    bin,contours,hier=cv2.findContours(bin1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.ones(bin1.shape[:2],dtype='uint8')*255
    for c in contours:
        if no_vaa(c,area):
            cv2.drawContours(mask,[c],-1,0,-1)
    Mask=np.amax(mask)- mask
    return Mask
def no_va(c):
    ar=cv2.arcLength(c,True)
    app=cv2.approxPolyDP(c,0.02*ar,True)
    return not len(app)==4;
    
def no_vaa(c,area):
    ar=cv2.contourArea(c)
    chk=(ar<area)
    return not chk;
def blob_coor(bin1):
    bin1=np.uint8(bin1)
    bin,contours,hier=cv2.findContours(bin1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    coor=[]
    ra=range(len(contours))
    for i in ra:
        rc=cv2.minAreaRect(contours[i])
        box=cv2.boxPoints(rc)
        pt=getCenter(box)
        coor.append(pt)
    while len(coor)<4:
        pt=(0,0)
        coor.append(pt)
    return coor
def getCenter(box):
    x=int((np.amax(box[:,0])+np.amin(box[:,0]))/2)
    y=int((np.amax(box[:,1])+np.amin(box[:,1]))/2)
    return (x,y)
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
    return coor

def perspectiva(Im,coor):
    coor=ordena(coor)
    print(coor)
    tl=coor[0]
    tr=coor[1]
    bl=coor[2]
    if len(coor)>4:
        br=coor[4]
    else:
        br=coor[3]
    pts=np.array((tl,tr,br,bl),dtype=np.float32)
    dst=np.array(((0,0),(900,0),(900,900),(0,900)),dtype=np.float32)
    M=cv2.getPerspectiveTransform(pts,dst)
    warp=cv2.warpPerspective(Im,M,(900,900))
    return warp
def anclas_cuad(warp):
    can=cv2.Canny(warp,100,400)
    can3=np.float32(can)
    crn=cv2.cornerHarris(can3,10,5,0.18)
    crn=cv2.dilate(crn,None)
    crn=cv2.dilate(crn,None)
    rn=no_area(crn,70)
    c=rn>0.01*rn.max()
    cv=rn*c
    coor=blob_coor(cv.copy())
    war=warp
    war[rn>0.01*rn.max()]=[255]
    IMG=warp[0:900, 0:900]
    return coor
def encuadre(gray,Im):
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
    corners = cv2.goodFeaturesToTrack(Mask,300,0.01,10)
    corners = np.int0(corners)
    crd=[]
    rg=[[[160, 220],[20, 80]],[[800 , 900],[40, 80]],[[10 , 80],[660 , 700]],[[920 , 1020],[640 , 780]]]
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
    for i in range(1,ln):
        l=len(xf)-1
        if np.absolute(x[i-1]-x[i])>dif:
            xf[l]//=k
            xf.append(x[i])
            k=1        
        else:
            k=k+1
            xf[l]+=x[i]
    l=len(xf)-1
    xf[l]//=k
    return xf
def print_cuad(xf,yf,im,color=(0,200,0)):
    img=im.copy()
    rx=range(len(xf))
    ry=range(len(yf))
    for i in rx:
        cv2.line(img,(int(xf[i]),int(yf[0])),(int(xf[i]),int(yf[len(yf)-1])),color,2)
    for i in ry:
        cv2.line(img,(int(xf[0]),int(yf[i])),(int(xf[len(xf)-1]),int(yf[i])),color,2)
    cv2.imshow('cuad',img)
    return 
def calibracion(Im):
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY);
    warp=encuadre(gray,Im)
    coor=anclas_cuad(warp)
    war=warp.copy()
    (x,y)=cuad_coord(coor)
    dif=50
    xf=seg_coord(x,dif)
    dif=50
    yf=seg_coord(y,dif)
    lx=len(xf)
    ly=len(yf)
    print_cuad(xf,yf,war)
    return(war,(xf,yf))
def ubicar(pin,div):
    for i in range(len(div)-1):
        if div[i]<=pin<=div[i+1]:
            return i
    return 0

# funcion de reconocimiento

def recon(warp,xf,yf):
    grw=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
#thresh de otsu
    ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    cv2.imshow('iman',th)
    sth=th.shape
    sth=th.shape
    back=np.ones(sth)*255
    # se extraen los contornos
    _,cont,_=cv2.findContours(th.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    print(len(cont))
    q=0
    cc=[]
    humm=[]
    icon=back
    
    board=np.ones((8,8))*0
    cboard=np.ones((8,8))*5
    for c in cont:
        ar=cv2.contourArea(c)
        if ar>1000 and ar<2500:
            q+=1
            ihu=[]
            hum=cv2.HuMoments(cv2.moments(c)).flatten()
            for h in range(len(hum)):
                nh=1/hum[h]
                nh=round(float(nh),2)
                ihu.append(nh)
            if 4<ihu[0]<50 and 60<ihu[1]:   
                cx,cy,w,h = cv2.boundingRect(c)
                cx+=w/2
                cx1=cx+(w/8)
                cx2=cx-(w/8)
                cy+=h/2
                cy1=cy+(h/8)
                cy2=cy-(h/8)
                cx=int(cx)
                cy=int(cy)
                px=ubicar(cx,xf)
                py=ubicar(cy,yf)
                pieza=0
                if 5.2 >ihu[0]:
                    if 90 >ihu[1]:
                        pieza+=2  # peon
                    elif 90 <ihu[1] <130:
                        pieza+=3 # torre
                    elif 130 <ihu[1]<500:
                        pieza+=4 # alfil
                    else:
                        pieza+=5 # reina
                else:
                    if 2000 >ihu[1]:
                        pieza+=6 # caballo
                    else: 
                        pieza+=7 # rey
                board[py,px]=pieza
                cl1=((th[cy1,cx]/255)+(th[cy1,cx1]/255)+(th[cy1,cx2]/255))/3
                cl=((th[cy,cx]/255)+(th[cy,cx1]/255)+(th[cy,cx2]/255))/3
                cl2=((th[cy2,cx]/255)+(th[cy2,cx1]/255)+(th[cy2,cx2]/255))/3
                cor=(cl+cl1+cl2)/3
                if cor>0.5:
                    color=1
                else:
                    color=0
                cboard[py,px]=color
                cc.append(c)
    icon=cv2.drawContours(icon,cc,-1,(00,120,120))
    cv2.imshow('cont',icon)
    #print_cuad(xf,yf,icon)
    return(q,board,cboard)

# funciones del tablero
def tran_chess(tab):
    fen=tab.fen().split()[0]
    lf=len(fen)
    rd=[]
    rc=[]
    bs=1;
    for i in range(len(fen)):
        val=fen[i]
        orv=ord(val.upper())
        if orv==47:
            nv=0
            nc=5
        else:
            if orv > 65:
                bs=1;
                if orv==80:
                    nv=2
                elif orv==82:
                    nv=3
                elif orv==78:
                    nv=6
                elif orv==66:
                    nv=4
                elif orv==81:
                    nv=5
                elif orv==75:
                    nv=7
                if val==val.upper():
                    nc=1;
                else:
                    nc=0
            else:
                bs=int(val)
            for j in range(bs):
                rd.append(float(nv))
                rc.append(float(nc))
    lb=np.asarray(rd).reshape(8,8)
    lc=np.asarray(rc).reshape(8,8)
    return(lb,lc)

# funciones del motor de ajedrez stockfish
def start():
    sf=Stockfish()
    board=chess.Board()
    return(board,sf)
def mover (mv,ucim,board,sf): # funcion para el movimiento del jugador
    ucim.append(mv)
    sf.set_position(ucim)
    board.push_uci(mv)
    print(tablero)
    return
def auto(ucim,board,sf): # funcion para el movimiento del aparato
    mv=sf.get_best_move()
    ucim.append(mv)
    sf.set_position(ucim)
    board.push_uci(mv)
    print(board)
    return