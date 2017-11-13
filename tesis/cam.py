import picamera
from funciones import *
from time import sleep

camera=picamera.PiCamera()
##camera.vflip=True
##camera.hflip=True
camera.sharpness = 100
camera.contrast = 100

camera.start_preview()
sleep(30)
camera.stop_preview()