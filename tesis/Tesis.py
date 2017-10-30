import numpy as np
import cv2
from time import sleep
import picamera
import funciones as fun
import chess
from stockfish import Stockfish

class Chesster:
    
    ## metodos
    def __init__(self, col,sharp=50,contrast=50,bright=50):
        self.camara = picamera.PiCamera()
        self.camara.vflip=True
        self.camara.hflip=True
        self.camara.sharpness = sharp
        self.camara.contrast = contrast
        self.camara.brightness = bright
        self.color=col
        self.motor=Stockfish()
        self.tablero=chess.Board()
        
    def preview(s=5):
        self.camera.start_preview()
        sleep(s)
        self.camera.stop_preview()
player