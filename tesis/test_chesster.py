import numpy as np
import cv2
from time import sleep
import picamera
import chess
from stockfish import Stockfish
import threading
from matplotlib import pyplot as plt
from chesster import Chesster
##import train-data
player=Chesster(bright=40,sharp=100,contrast=100)

##inicio de juego
im=player.captura('br1.jpg')
##im=cv2.imread('captura.jpg')
## calibracion
imc=player.calibracion(im)
player.plt_show(imc)
##inp=np.load('inp.npy')
##out=np.load('out.npy')