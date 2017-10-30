 # LIBRERIAS
import numpy as np
import cv2
from time import sleep
import picamera
import funciones as fun

camera=picamera.PiCamera()
camera.vflip=True
camera.sharpness = 100
camera.contrast = 0
#inicializacion de variables
# acomodacion de camara
#camera.start_preview()
#sleep(0)
#camera.stop_preview()
#<<<<<<<<<<<<<<<<<< inicio calibracion >>>>>>>>>>>>>>>>>
inp='';
while(inp.upper()!='Y'):
    i=1
    name='caln1.jpg'
    camera.capture(name,0)
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
i=2
inp=''
board1=np.zeros((8,8))
cboard1=board1
trn=1
k=0;
inp='';
while(inp.upper()!='Y'):
    #### ------ Captura de imagen    
    name='rec'+str(i)+'.jpg'
    ##   camera.capture(name,0)
    Im= cv2.imread(name)
    #    cv2.imshow('imagen',Im)
    gray=cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY)
    warp=fun.encuadre(gray,Im)

        
        #### ------ imagen capturada y procesada para reconocimiento
        ## grw  = la imagen   
       
    k+=1
    if trn==0:
       trn=1
    else:
       trn=0
    if k>1:
       cboard1=cboard
       board1=board
    cv2.imshow('tablero',warp)
    (q,board,cboard)=fun.recon(warp,xf,yf)
        
    print('--')
    print(q)
    print('*****')
    print(board)
    print('*****')
    print(cboard)
    print('*****')
        
    if k>1:
        cambio=np.equal(board.flatten(),board1.flatten());
        cambio=1-(1*cambio);
        change=cambio.nonzero();
        change=change[0]
        print(change)
        print(change % 8) #columna
        print(change // 8) # fila
            
        print('*****')
        turno=1*(np.equal(cboard,trn))
        turno=turno.flatten()
        otr=(not(trn))*1
        oturno=1*(np.equal(cboard,otr))
        oturno=oturno.flatten()
        print(turno)
        print((change[0],change[1]))
        j1=(turno[change[0]],turno[change[1]])
        j2=(oturno[change[0]],oturno[change[1]])
        print(j1)# orden 1 es la nueva 0 la vieja
        print(j2)
        print((j1[0]*j2[0],j1[1]*j2[1]))# si hay transferencia alguno tiene que ser =1
        print('*****')    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import chess
tab=chess.Board()
# traduccion fen a array
(lb,lc)=fun.tran_chess(tab)
print(lb)