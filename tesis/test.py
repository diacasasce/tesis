import numpy as np
import cv2
from time import sleep
import picamera
camera=picamera.PiCamera()
camera.vflip=True
camera.brightness=55
clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
camera.start_preview()
sleep(3)
camera.stop_preview()
camera.capture('cap1.jpg',0)
Im= cv2.imread('cap1.jpg',0)
crn=cv2.goodFeaturesToTrack(Im,100,0.5,300)
crn=np.int0(crn)
for crn in crn:
     x,y=crn.ravel()
     Ims=cv2.circle(Im,(x,y),5,255,-1)
cv2.imshow('esqui',Ims)
can=cv2.Canny(Im,60,100)
sy=cv2.Sobel(can,cv2.CV_8U,1,0,ksize=5)
sx=cv2.Sobel(can,cv2.CV_8U,0,1,ksize=5)
so=sx+sy
can=so
cv2.imshow('ls',so)
sb=cv2.GaussianBlur(so,(7,7),1)
cv2.imshow('lb',sb)
sbg=np.float32(so)
crn=cv2.goodFeaturesToTrack(sbg,100,0.01,10)
#crn=cv2.cornerHarris(gr,3,1,1)
crn=np.int0(crn)
for crn in crn:
     x,y=crn.ravel()
     cv2.circle(Im,(x,y),5,255,-1)
     cv2.circle(sb,(x,y),5,255,-1)
cv2.imshow('esquinas',Im)
cv2.imshow('lbe',sb)

#print(np.amax(can))
#cv2.circle(eq,(300,180),10,(255,255,255),1)
#cv2.circle(eq,(1000,210),10,(255,255,255),1)
#cv2.circle(eq,(1255,800),10,(255,255,255),1)
#cv2.circle(eq,(120,820),10,(255,255,255),1)
tl=(300,180)
tr=(1000,210)
br=(1255,800)
bl=(120,820)
#pts=np.array((tl,tr,br,bl),dtype=np.float32)
#dst=np.array(((0,0),(700,0),(700,700),(0,700)),dtype=np.float32)
#M=cv2.getPerspectiveTransform(pts,dst)
#warp=cv2.warpPerspective(Im,M,(900,800))
#cv2.imshow('im',la)
#cv2.imshow('im1',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

