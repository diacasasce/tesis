import numpy as np
import cv2
from time import sleep
import picamera
import chess
from stockfish import Stockfish
import threading
from matplotlib import pyplot as plt
from chesster import Chesster
import serCOM as sC
##import train-data
player=Chesster(bright=45,sharp=100,contrast=100)
##inicio de juego

player.new_game(False)
im=player.captura('br1.jpg')
#im=cv2.imread('br1.jpg')
player.plt_show(im)
## calibracion

imc=player.calibracion(im)
input('go?')

while not(player.brain.is_over()):
    player.go_auto()
    if not(player.brain.is_over()):
        player.human()
        
player.mv2ser(-100,0)
##player.plt_show(imc)

