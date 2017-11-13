import numpy as np
import cv2
from time import sleep
import picamera
import chess
from stockfish import Stockfish
import threading
from matplotlib import pyplot as plt
import MLP
class Chesster:
    
    ## metodos
    
    def __init__(self,sharp=-1,contrast=-1,bright=-1):
        self.camara = picamera.PiCamera()
##        self.camara.vflip=True
##        self.camara.hflip=True
        self.dic_x={1:"a",2:"b",3:"c",4:"d",5:"e",6:"f",7:"g",8:"h"}
        self.dic_p={(2,0):'p',(2,1):'P',(3,0):'r',(3,1):'R',(4,0):'b',(4,1):'B',(5,0):'q',(5,1):'Q',(6,0):'n',(6,1):'N',(7,0):'k',(7,1):'K'}
        self.k=0
        self.red=mlp(2,np.array([20]),1)
        if sharp>0:
            self.camara.sharpness = sharp
        if contrast>0:
            self.camara.contrast = contrast
        if bright>0:
            self.camara.brightness = bright
    def __del__(self):
        self.camara.close()
     
    def captura(self,name='captura.jpg'):
        self.camara.capture(name,0)
        Im= cv2.imread(name)
        return Im
    
    def plt_show(self,imagen):
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()
    def cv_show(self,im,win='window'):
        cv2.imshow(win,im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
        dst=np.array(((0,0),(700,0),(700,700),(0,700)),dtype=np.float32)
        M=cv2.getPerspectiveTransform(pts,dst)
        warp=cv2.warpPerspective(Im,M,(700,700))
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
##            print(np.amax(x))
            print (crn)
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
            print(puntos)
        else:
            puntos=self.pun
            
        wrp=self.perspectiva(im,puntos)
        return wrp