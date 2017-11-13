import numpy as np
import cv2
from time import sleep
import picamera
from serCOM import *
from motor import Brain
import threading
from matplotlib import pyplot as plt
import MLP
class Chesster:
    
    ## metodos
    
    def __init__(self,sharp=-1,contrast=-1,bright=-1):
        self.camara = picamera.PiCamera()
        (self.arm,self.dA)=conectarA(to=10)
        self.posiciones=[]
        for i in range(0,64):
            self.posiciones.append(False)
        self.dic_x={1:"a",2:"b",3:"c",4:"d",5:"e",6:"f",7:"g",8:"h"}
        
        self.dic_p={(2,0):'p',(2,1):'P',(3,0):'r',(3,1):'R',(4,0):'b',(4,1):'B',(5,0):'q',(5,1):'Q',(6,0):'n',(6,1):'N',(7,0):'k',(7,1):'K'}
        self.k=0
        self.red=MLP.mlp(7,np.array([3]),1)
        if sharp>0:
            self.camara.sharpness = sharp
        if contrast>0:
            self.camara.contrast = contrast
        if bright>0:
            self.camara.brightness = bright
    def __del__(self):
        self.camara.close()
    def help(self):
        file = open('help.txt', 'r') 
        for line in file: 
            print(line), 

    def preview(self,s=5):
        self.camara.start_preview()
        sleep(s)
        self.camara.stop_preview()
    
    def new_game(self,col=True):
        if col:
            self.camara.vflip=True
            self.camara.hflip=True
        else:
            self.camara.vflip=False
            self.camara.hflip=False
            # falta setear la correccion de coordenadas
        self.color=col
        self.brain=Brain()
        self.capt=0
        
    
    def captura(self,name='captura.jpg'):
        if self.dA:
            send_ser('150;0;0;1;2;\r\n',self.arm)
            sleep(10)
            self.camara.capture(name,0)
            Im= cv2.imread(name)
            return Im
    
    def plt_show(self,imagen):
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()    
    
    def encuadre(self,im,CP=False,it=1):
        cr_l=self.correccion_luz(im)
##        self.cv_show(cr_l)
        df_t=self.define_tablero(cr_l,it)
##        self.cv_show(df_t)
        
        no_f=self.no_fondo(df_t)
##        self.cv_show(no_f)
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
            self.plt_show(im_co)
            prd=np.asarray(x)+np.asarray(y)
            prd1=np.asarray(y)+((np.amax(x)*1.5)-np.asarray(x))
            tl=crn[np.argmin(prd)]
##            print(tl)
            tr=crn[np.argmin(prd1)]
##            print(tr)
            br=crn[np.argmax(prd)]
##            print(br) 
            bl=crn[np.argmax(prd1)]
##            print(bl)
            puntos=(tl,tr,br,bl)
            self.pun=puntos
            print('enter show')
            im_co=self.show_coor(im,puntos)
            self.plt_show(im_co)
        else:
            puntos=self.pun
            
        wrp=self.perspectiva(im,puntos)
        return wrp
    def no_fondo(self,bin1):
        bin,contours,hier=cv2.findContours(np.uint8(bin1.copy()),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        mask=np.ones(bin1.shape[:2],dtype='uint8')*255
        ar1=0
        cf=0
        for c in contours:
            ar=cv2.contourArea(c)
            if ar>ar1:
                cf=c
                ar1=ar
            
        cv2.drawContours(mask,[cf],-1,0,-1)
        Mask=np.amax(mask)- mask
        return Mask
    def perspectiva(self,Im,coor):
        pts=np.array(coor,dtype=np.float32)
        print(pts)
        dst=np.array(((0,0),(840,0),(840,840),(0,840)),dtype=np.float32)
        M=cv2.getPerspectiveTransform(pts,dst)
        warp=cv2.warpPerspective(Im,M,(840,840))
        return warp
    def correccion_luz(self,im,cl=2.5,lgs=(8,8)) :
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
        Ime=cv2.equalizeHist(gray)
##        self.cv_show(Ime)
        clahe=cv2.createCLAHE(clipLimit=cl,tileGridSize=lgs)
        Imc=clahe.apply(Ime)
##        self.cv_show(Imc)
        ker=np.matrix([[0,-1,0],[-1,5,-1],[0,-1,0]])
        Imc=cv2.filter2D(Imc,-1,ker)
##        self.cv_show(Imc)
        bin=self.define_tablero(Imc)
        return bin;    
    def define_tablero(self,gray,it=3):
        gray=cv2.GaussianBlur(gray,(5,5),0)
        _, bin=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        M=np.amax(bin)
        bin=cv2.dilate(bin,None)
        bin=M-bin
        bin=cv2.dilate(bin,None,iterations=it)
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
    
    def goNext(self):
        self.k=self.k+1
    def calibracion(self,im):
        warp=self.encuadre(im)
##        self.plt_show(warp)
        coor=self.anclas_cuad(warp)
        cord=self.blob_coor(coor)
        cord=self.ordena(cord)
##        print(cord)
        tim=self.show_coor(warp,cord)
##        self.plt_show(tim)
        cerd=self.rem_cluster(cord,80)
        (cx,cy)=self.cuad_coord(cerd)
        timy=self.show_coor(warp,cerd)
##        self.plt_show(timy)
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
        self.D_ccx={"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}
        self.D_ccy={"1":0,"2":1,"3":2,"4":3,"5":4,"6":5,"7":6,"8":7}
        self.ccx=[]
        self.ccy=[]
        for i in range(1,len(self.xf)):
            self.ccx.append((self.xf[i]+self.xf[i-1])/2)
            self.ccy.append((self.yf[i]+self.yf[i-1])/2)
##        print(len(self.Cuad))
        im_c=self.print_cuad(warp)
        imco=self.show_coor(im_c,self.Cuad)
##        self.plt_show(imco)
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
##        print('go')
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
        self.plt_show(th)
        sth=th.shape
        sth=th.shape
        back=np.ones(sth)*255
        # se extraen los contornos
        _,cont,_=cv2.findContours(th.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        print(len(cont))
        q=0
        cc=[]
        humm=[]
        icon=self.print_cuad(back)
        ico=icon
        board=np.ones((8,8))*0
        cboard=np.ones((8,8))*5
        for c in cont:
            ar=cv2.contourArea(c)
            if 700<ar<5000:
                q+=1
                ihu=[]
                hum=cv2.HuMoments(cv2.moments(c)).flatten()                
                for h in range(len(hum)):
                    nh=1/hum[h]
                    nh=round(float(nh),2)
                    ihu.append(nh)
                
                if 4<ihu[0]<6:
                    print((ar,ihu))
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
##                    if ihu[1]>2000: #rey
                    if 400 >=ihu[1]: # peon - torre
                        if ihu[4]>0:
                            if ihu[2]<5000:
                                pieza+=2  # peon
                            else:
                                pieza+=3 # torre
                        else:
                            pieza+=4 # alfil
                    elif 400<ihu[1]<2000:
                        if 5000>ihu[2]:
                            pieza+=5 # reina
                            
                        else:
                            pieza+=6 # caballo
                    elif ihu[1]>2000: 
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
##                    ico=cv2.drawContours(ico,cc,-1,(00,120,120))
                    print(pieza)
##                    self.plt_show(ico)
                    
        icon=cv2.drawContours(icon,cc,-1,(00,120,120))
        icc=self.print_cuad(icon,color=(0,0,0))
        #print_cuad(xf,yf,icon)
        return(q,board,cboard,icc)
        
    
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
    def get_board(self,it,CP=False):
        im= self.captura('cap1.jpg')
        en=self.encuadre(im,CP,it)
        (q,b,c,icc)=self.recon(en)
        return (q,b,c,icc)
    def ch2eng(self,b,c):
        fb=b.flatten()
        fc=c.flatten()
        pz=[]
        for i in range(len(fb)):
            if fb[i]>0 and fc[i]<5:
                x=(i%8)+1
                y=8-(i//8)
                pz.append((self.dic_p[(fb[i],fc[i])],(self.dic_x[x],y)))
        print(pz)

    def get_data(self,na='data',ti=0,CP=False):
        name='CP'+na+'.jpg'
        if ti <1:
            im= self.captura(name)
            self.cv_show(im)
            print('in encuadre')
            en=self.encuadre(im,CP=CP)
            print('out encuadre')
            self.cv_show(en)
            sen='en'+na+'.jpg'
            cv2.imwrite(sen,en)
            grw=cv2.cvtColor(en,cv2.COLOR_BGR2GRAY)
        #thresh de 8otsu
            ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            self.cv_show(th)
            sth=th.shape
            sth=th.shape
        else:
            im =cv2.imread(name)
            grw=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #thresh de 8otsu
            ret,th=cv2.threshold(grw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            self.cv_show(th)
            sth=th.shape
            
        
        back=np.ones(sth)*255
##        # se extraen los contornos
        _,cont,_=cv2.findContours(th.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        print(len(cont))
        q=0
        cc=[]
        humm=[]
        icon=self.print_cuad(back)
        ico=icon
        board=np.ones((8,8))*0
        cboard=np.ones((8,8))*5
        inp=[]
        out=[]
        for c in cont:
            ar=cv2.contourArea(c)
            
            if 700<ar<5000:
                q+=1
                ihu=[]
                hum=cv2.HuMoments(cv2.moments(c)).flatten()                
                for h in range(len(hum)):
                    nh=1/hum[h]
                    nh=round(float(nh),2)
                    ihu.append(nh)
                if 4<ihu[0]<6:
                    print (1)
                    cc.append(c)
                    ico=cv2.drawContours(ico,c,-1,(00,120,120))
                    self.plt_show(ico)
                    pie=float(input('? '))
                    print(ihu,pie)
                    inp.append(ihu)
                    out.append([pie])
                    er=self.red.backward(np.array([ihu]),pie)
                    print(er)
        return (inp,out)
##        icon=cv2.drawContours(icon,c,-1,(00,120,120))
##        icc=self.print_cuad(icon,color=(0,0,0))
##        self.cv_show(icon)
##        wn='dt'+na+'.jpg'
##        cv2.imwrite(wn,icon)
    
        
    def to_move(self,mv):
        return(self.brain.to_move(mv))
    def go_auto(self):
        mv=self.brain.auto()
        print(mv)
        self.goto(mv)
    def goto(self,mv):
        enr=self.brain.enroque(self.to_move(mv))
        print('enroque',enr)
        cap=self.brain.capturo(self.to_move(mv))
        print('captura',cap)
        if cap[0]:
            y1=-round(((self.ccx[self.D_ccx[mv[2]]])/2)-210)
            x1=round(((self.ccy[self.D_ccy[mv[3]]])/2)-25)
            self.mv2ser(x1,y1)
            yc=275
            
            xc=50*(self.capt%5)+25
            self.capt=self.capt+1
            self.mv2ser(xc,yc)
        if enr[0]:
            if enr[1]:
                my="hf"
            else:
                my="ad"
            y=-round(((self.ccx[self.D_ccx[my[0]]])/2)-210)
            x=round(((self.ccy[self.D_ccy[mv[1]]])/2)-25)
            self.mv2ser(x,y)
    ##        sleep(8)
            print(x,y)
            y1=-round(((self.ccx[self.D_ccx[my[1]]])/2)-210)
            x1=round(((self.ccy[self.D_ccy[mv[3]]])/2)-25)
            self.mv2ser(x1,y1)
            
            
        y=-round(((self.ccx[self.D_ccx[mv[0]]])/2)-210)
        x=round(((self.ccy[self.D_ccy[mv[1]]])/2)-25)
        self.mv2ser(x,y)
##        sleep(8)
        print(x,y)
        y1=-round(((self.ccx[self.D_ccx[mv[2]]])/2)-210)
        x1=round(((self.ccy[self.D_ccy[mv[3]]])/2)-25)
        self.mv2ser(x1,y1)
##        sleep(8)
        print(x,y)
        self.brain.mover_uci(mv)
        self.mv2ser(-100,0)
    def human(self):
        
        mv=input ("jugada? ")
        rt=self.brain.legal(mv)
        while not(rt):
            mv=input (" ERROR Corrija su jugada? ")
            rt=self.brain.legal(mv)
        self.brain.mover_uci(mv)
        
    def mv2ser(self,x,y):
        st=str(x)+";"+str(y)+";0;1;1;\r\n"
        send_ser(st,self.arm)