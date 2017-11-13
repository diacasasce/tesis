import sys
import threading
import cv2
from PyQt4 import QtGui,QtCore
from serCOM import *
from time import sleep
from chesster import Chesster

class Window(QtGui.QMainWindow):
    def __init__(self,):
        super(Window,self).__init__()
        self.busy=False
        self.dA=False
        self.dB=False
        self.on=False
        self.w=800
        self.h=480
        self.im_queue=0
##        titulo
        self.label('Chess-ter',size=(self.w,60),frame="white")
##        info secuencias
        self.label('Controles',size=(150,30),pos=(25,60))
        fr1=self.frame(size=(150,370) ,pos=(25,90))
        btt1=self.btn('Nuevo Juego',self.iniciar,size=(120,50),pos=(40,100))
        btt2=self.btn('Calibrar',self.calibrar,size=(120,50),pos=(40,150))
        btt3=self.btn('Jugar',self.jugar,size=(120,50),pos=(40,200))
        btt4=self.btn('Finalizar',self.rep,size=(120,50),pos=(40,250))
        btt5=self.btn('Salir',self.fin,size=(120,50),pos=(40,300))
        self.label('Imagen',size=(350,30),pos=(225,70))
        self.imv=self.label(size=(350,350) ,pos=(225,100),frame='-')
        self.label('Representacion',size=(150,30),pos=(625,70))
        
        
        self.setGeometry(50,50,self.w,self.h)
        self.setWindowTitle('Chess-ter 1')
        self.conecta()
##        self.showFullScreen()
        self.show()
        
   
        
    def frame(self,size=0,pos=0):
        frm=QtGui.QFrame(self)
        if size!=0:
            frm.resize(size[0],size[1])
        if pos!=0: 
            frm.move(pos[0],pos[1])
        frm.setFrameShape(QtGui.QFrame.Box)
        frm.setFrameShadow(QtGui.QFrame.Sunken)
        return frm
    def label(self,txt=0,size=0,pos=0,frame=0):
        if frame!=0: 
            lfr=self.frame(size,pos)
            if frame!='-':
                stl="QFrame { background-color:"+frame+"}"
                lfr.setStyleSheet(stl)
        if txt!=0:
            lbl=QtGui.QLabel(txt,self)
        else:
            lbl=QtGui.QLabel(self)
        if size!=0:
            lbl.resize(size[0],size[1])
        if pos!=0: 
            lbl.move(pos[0],pos[1])
        return lbl
        
    def btn(self,name,funct,size=0,pos=0):
        btn=QtGui.QPushButton(name,self)
        btn.clicked.connect(funct)
        if size!=0:
            btn.resize(size[0],size[1])
        if pos!=0:
            btn.move(pos[0],pos[1])
        return btn
    def combo(self,items,size=0,pos=0):
        cbx=QtGui.QComboBox(self)
        if type(items)==tuple:
            for i in range(len(items)):
                cbx.addItem(items[i])
        else:
            cbx.addItem(items[i])
        if size!=0:
            cbx.resize(size[0],size[1])
        if pos!=0:
            cbx.move(pos[0],pos[1])
        cbx.activated[str].connect(self.com_fun)
        return cbx
    def conecta(self):
        print('Conectar')
        if not(self.dA):
            (self.SerA,self.dA)=conectarA()
            if self.dA:
                print('Ca')
                Ca=threading.Thread(name='sa',target=self.conA)
                Ca.start()
            
        if not(self.dB):
            (self.SerB,self.dB)=conectarB()
            if self.dB:
                print('cb')
                Cb=threading.Thread(name='sb',target=self.conB)
                Cb.start()
    def conB(self):
        self.dB=on_B()
        while self.dB==True:
            self.lMb.setText("activo")
            self.fMb.setStyleSheet("QFrame { background-color: Green }" ) 
            self.dB=on_B()
        self.lMb.setText("inactivo")
        self.fMb.setStyleSheet("QFrame { background-color: Red }" )
    def conA(self):
        self.dA=on_A()
        while self.dA==True:
            self.lMa.setText("activo")
            self.fMa.setStyleSheet("QFrame { background-color: Green }" ) 
            self.dA=on_A()
        self.lMa.setText("inactivo")
        self.fMa.setStyleSheet("QFrame { background-color: Red }" )
    def print_im(self,imc):
        sh=imc.shape
        h=sh[0]
        w=sh[1]
        if len(sh)==3:
            c=sh[2]
        bpl=3*w
        qimc=QtGui.QImage(imc.data,w,h,bpl,QtGui.QImage.Format_RGB888)
        qpix=QtGui.QPixmap()
        qpix.convertFromImage(qimc)
        self.imv.setPixmap(qpix.scaledToHeight(350))
    def iniciar(self):
        if not(self.on):
            self.player=Chesster(bright=55,sharp=100,contrast=100)
##            self.player.preview(2)
    
    def calibrar(self): 
        print('calibrar')
        im=self.player.captura()
        imc=self.player.calibracion(im)
        print(imc.shape)
        self.im_queue=imc
        self.print_im(imc)
    def jugar(self):
        print('recon')
        im=self.player.captura()
        (q,b,c,imc)=self.player.get_board(1)
        print(imc.shape)
        print(b)
        print(q)
        self.im_queue=imc
        self.print_im(cv2.merge((imc,imc,imc))) 
    def rep(self):
        del self.player
        print("rep")
    def fin(self):
        sys.exit(self)

def main():
    app=QtGui.QApplication(sys.argv)
    GUI=Window()
    sys.exit(app.exec_())
    
    
gui=threading.Thread(name='gui',target=main)
gui.start()
