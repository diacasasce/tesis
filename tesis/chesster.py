import numpy as np
import cv2
from time import sleep
import picamera
import chess
from stockfish import Stockfish
from matplotlib import pyplot as plt

class Chesster:
    
    ## metodos
    
    def __init__(self,sharp=-1,contrast=-1,bright=-1):
        self.camara = picamera.PiCamera()
        self.camara.vflip=True
        self.camara.hflip=True
        
        if sharp>0:
            self.camara.sharpness = sharp
        if contrast>0:
            self.camara.contrast = contrast
        if bright>0:
            self.camara.brightness = bright
    
    def help(self):
        file = open('help.txt', 'r') 
        for line in file: 
            print(line), 
    
    def preview(self,s=5):
        self.camara.start_preview()
        sleep(s)
        self.camara.stop_preview()
    
    def new_game(self,col=1):
        # falta setear la correccion de coordenadas
        self.color=col
        self.motor=Stockfish()
        self.tablero=chess.Board()
    
    def captura(self,name='captura.jpg'):
        self.camara.capture(name,0)
        Im= cv2.imread(name)
        return Im
    
    def plt_show(self,imagen):
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()    
    
    def encuadre(self,im,CP=False):
        cr_l=self.correccion_luz(im)
##        self.plt_show(cr_l)
        df_t=self.define_tablero(cr_l)
##        self.plt_show(df_t)
        no_f=self.remove_area_men(df_t,100000)
        self.plt_show(no_f)
        if not(CP):
            corners = cv2.goodFeaturesToTrack(no_f,500,0.01,10)
            corners = np.int0(corners)
            x=[]
            y=[]
            crn=[]
            crn1=[]
            for i in corners:
                xi,yi = i.ravel()
                x.append(xi)
                y.append(yi)
                crn.append((xi,yi))
                xi1=1200-xi
                yi1=yi
                crn1.append((xi1,yi1))
    ##        print(np.amax(x))
            im_co=self.show_coor(im,crn)
##            self.plt_show(im_co)
            prd=np.asarray(x)*np.asarray(y)
            prd1=np.asarray(y)*((np.amax(x)*1.1)-np.asarray(x))
            tl=crn[np.argmin(prd)]
            print(tl)
            tr=crn[np.argmin(prd1)]
            print(tr)
            br=crn[np.argmax(prd)]
            print(br) 
            bl=crn[np.argmax(prd1)]
            print(bl)
            
            puntos=(tl,tr,br,bl)
            self.pun=puntos
            im_co=self.show_coor(im,puntos)
##            self.plt_show(i_mco)
        else:
            puntos=self.pun
        wrp=self.perspectiva(im,puntos)
        return wrp
    def perspectiva(self,Im,coor):
        pts=np.array(coor,dtype=np.float32)
        print(pts)
        dst=np.array(((0,0),(700,0),(700,700),(0,700)),dtype=np.float32)
        M=cv2.getPerspectiveTransform(pts,dst)
        warp=cv2.warpPerspective(Im,M,(700,700))
        return warp
    def correccion_luz(self,im,cl=2.5,lgs=(8,8)) :
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
        Ime=cv2.equalizeHist(gray)
        clahe=cv2.createCLAHE(clipLimit=cl,tileGridSize=lgs)
        Imc=clahe.apply(Ime)
        ker=np.matrix([[0,-1,0],[-1,5,-1],[0,-1,0]])
        Imc=cv2.filter2D(Imc,-1,ker)
        bin=self.define_tablero(Imc)
        return bin;    
    def define_tablero(self,gray):
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
        return bin1

    def remove_area_men(self,bin1,area):
        bin,contours,hier=cv2.findContours(np.uint8(bin1.copy()),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        mask=np.ones(bin1.shape[:2],dtype='uint8')*255
        for c in contours:
            ar=cv2.contourArea(c)
            if ar>area:
                cv2.drawContours(mask,[c],-1,0,-1)
        Mask=np.amax(mask)- mask
        return Mask
    
    def remove_area_may(self,bin1,area):
        bin,contours,hier=cv2.findContours(np.uint8(bin1.copy()),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        mask=np.ones(bin1.shape[:2],dtype='uint8')*255
        for c in contours:
            ar=cv2.contourArea(c)
            if ar<area:
                cv2.drawContours(mask,[c],-1,0,-1)
        Mask=np.amax(mask)- mask
        return Mask
    def calibracion(self,im):
        warp=self.encuadre(im)
        self.plt_show(warp)
        coor=self.anclas_cuad(warp)
        cord=self.blob_coor(coor)
        cord=self.ordena(cord)
        print(cord)
        tim=self.show_coor(warp,cord)
        self.plt_show(tim)
        cerd=self.rem_cluster(cord,80)
        (cx,cy)=self.cuad_coord(cerd)
        timy=self.show_coor(warp,cerd)
        self.plt_show(timy)
        dif=20
        lx=1
        self.xf=[]
        self.yf=[]
        while not(lx==9):
            if lx > 9:
                dif=dif+1
            else:
                dif=dif-1
            self.xf=self.seg_coord(cx,dif)
            lx=len(self.xf)
##            print(lx)
        ly=1
        dif=20
        while not(ly==9):
            if ly > 9:
                dif=dif+1
            else:
                dif=dif-1
            self.yf=self.seg_coord(cy,dif)
            ly=len(self.yf)
##            print(ly)
        self.xf=np.asarray(self.xf).astype(int)
        self.yf=np.asarray(self.yf).astype(int)
        self.Cuad=self.XY2Coor(self.xf,self.yf)
##        print(len(self.Cuad))
        im_c=self.print_cuad(warp)
        imco=self.show_coor(im_c,self.Cuad)
        self.plt_show(imco)
        return imco
        
    def anclas_cuad(self,warp):
##        self.plt_show(warp)
        gwr=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY);
##        cr_l=self.correccion_luz(warp,cl=2.5)
        cr_l=gwr
##        self.plt_show(cr_l)
        ret,th=cv2.threshold(np.uint8(cr_l),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##        self.plt_show(th)
        can=cv2.Canny(th,100,400)
        can3=np.float32(can)
        crn=cv2.cornerHarris(can3,10,5,0.18)
        crn=cv2.dilate(crn,None,iterations=1)
##        self.plt_show(crn)
        rn=self.remove_area_men(crn,70)
##        rn1=self.remove_area_men(crn,50) 
##        print(1)
##        self.plt_show(rn)
##        self.plt_show(rn1)
##        rn=rn1-rn
##        self.plt_show(rn)
##        self.plt_show(crn-rn)
        c=rn>0.2*rn.max()
        cv=self.remove_area_may(c*rn,190)
##        self.plt_show(cv)
        return cv
    def XY2Coor(self,x,y):
        coor=[]
        for i in x:
            for j in y:
                coor.append((i,j))
        return coor
    def show_coor(self,im,coor):
        im_co=im.copy()
        for i in coor:
            print(i)
            cv2.circle(im_co,i,5,(255),-1)
        return im_co

    def blob_coor(self,bin1):
        bin1=np.uint8(bin1)
        bin,contours,hier=cv2.findContours(bin1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        coor=[]
        ra=range(len(contours))
        for i in ra:
            rc=cv2.minAreaRect(contours[i])
            box=cv2.boxPoints(rc)
            pt=self.getCenter(box)
            coor.append(pt)
        return coor
    def getCenter(self,box):
        x=int((np.amax(box[:,0])+np.amin(box[:,0]))/2)
        y=int((np.amax(box[:,1])+np.amin(box[:,1]))/2)
        return (x,y)
    def cuad_coord(self,coor):
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
    def seg_coord(self,x,dif):
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
    def cv_show(self,im,win='window'):
        cv2.imshow(win,im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def set_bright(self,bright):
        self.camara.brightness=bright
    def print_cuad(self,im,color=(255,255,255)):
        img=im.copy()
        rx=range(len(self.xf))
        ry=range(len(self.yf))
        for i in rx:
            cv2.line(img,(self.xf[i],self.yf[0]),(self.xf[i],self.yf[len(self.yf)-1]),color,2)
        for i in ry:
            cv2.line(img,(self.xf[0],self.yf[i]),(self.xf[len(self.xf)-1],self.yf[i]),color,2)
        return img
    
    def recon(self,warp):
        grw=cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    #thresh de otsu
        ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##        self.plt_show(th)
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
            if 700<ar<1500:
                q+=1
                ihu=[]
                hum=cv2.HuMoments(cv2.moments(c)).flatten()                
                for h in range(len(hum)):
                    nh=1/hum[h]
                    nh=round(float(nh),2)
                    ihu.append(nh)
                print(ihu)
                if 4<ihu[0]<50 :   
                    cx,cy,w,h = cv2.boundingRect(c)
                    cx+=w/2
                    cx1=cx+(w/8)
                    cx2=cx-(w/8)
                    cy+=h/2
                    cy1=cy+(h/8)
                    cy2=cy-(h/8)
                    cx=int(cx)
                    cy=int(cy)
                    px=self.ubicar(cx,self.xf)
                    py=self.ubicar(cy,self.yf)
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
                        if 1500 >ihu[1]:
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
        print(humm)
        icon=cv2.drawContours(icon,cc,-1,(00,120,120))
        self.cv_show(icon)
        #print_cuad(xf,yf,icon)
        return(q,board,cboard)
        
    
    def ubicar(self,pin,div):
        for i in range(len(div)-1):
            if div[i]<=pin<=div[i+1]:
                return i
        return 0
    def ordena(self,coor):
        ctype=[('x',int),('y',int)]        
        ar=np.array(coor,dtype=ctype)
        sar=np.sort(ar)
        aro=[]
        for i in sar:
            aro.append((int(i['x']),int(i['y'])))
        return aro
    def rem_cluster(self,coor,dif):
        lc=len(coor)
        rst=[coor[0]]
        for i in range(1,lc):
            cr=coor[i]
            if self.dist(cr,rst[-1])>dif:
                rst.append(cr)
        return rst
    def dist(self,c1,c2):
        rst=np.sqrt(((c1[0]-c2[0])**2)+((c1[1]-c2[1])**2))
        return rst
    def get_board(self):
        im= self.captura()
        self.plt_show(im)
        en=self.encuadre(im)
        (q,b,c)=self.recon(en)
        return (q,b,c)
player=Chesster(bright=60,sharp=100,contrast=100)
im=player.captura()
#im=cv2.imread('captura.jpg')
player.plt_show(im)

## calibracion
imc=player.calibracion(im)
## deteccion
hum=[]
##(q,board,cboard)=player.get_board()
