import numpy as np
import cv2
from time import sleep
class mlp:
    def __init__(self,inp=-1,lay=-1,out=-1,tmpm=-1,mp=0):
        if tmpm<0:
##            np.random.seed(0)
            self.mpm=[]
            h=len(lay)
            l=(4+np.random.rand(inp,lay[0]))/10
            self.mpm.append(l)
            for i in range(1,h):
                l=(4+np.random.rand(lay[i-1],lay[i]))/10
                self.mpm.append(l)
            l=(4+np.random.rand(lay[h-1],out))/10
            self.mpm.append(l)
        else:
            self.mpm=mp
    def train(self,inp,out,ta,th,ep=100):
        erm=1
        k=0
        while (th<erm)==True or (k<ep)==true:
            erm=0
            ir=len(inp)
            for i in range(ir):
                (er)=self.backward(np.array([inp[i]]),out[i],ta)
                if np.abs(er)>erm:
                    erm=np.abs(er)
                    
            k=k+1
            if k%100==0:
                np.save('mpm',self.mpm)
            print(erm)
##            print(th)
##            print(th<erm)
##            print('=')
    def backward(self,inp,out,ta=0.5):
        bmpm=self.mpm
        h=len(self.mpm)
        fwi,fwo,fd=self.forward(inp)
        for j in range(0,h):
            i=h-j-1
            if i== h-1:
                er=out-fwo[i]
                erm=er
            else:
                er=np.dot(self.mpm[h-j],Dk.transpose())
            Dk=fwo[i]*(1-fwo[i])*er.transpose()
            self.mpm[i]=self.mpm[i]+(ta*np.dot(fwi[i].transpose(),Dk))
        return erm[0][0]
    def forward(self,inp):
        ohl=[]
        inl=[]
        h=len(self.mpm)
        z=np.dot(inp,self.mpm[0])
        out=1/(1+np.exp(-z))
        ohl.append(out)
        inl.append(inp)
        for i in range(1,h):
            inl.append(out)
##            print('**')
##            print(out)
##            print(self.mpm[i])
            z=np.dot(out,self.mpm[i])
            out=1/(1+np.exp(-z))
            ohl.append(out)
        return (inl,ohl,out)
##    
##inp=np.array([[1,0],[0,1],[0,0],[1,1]])
##out=np.array([[1],[1],[0],[0]])
##tmp=[np.array([[ 0.74856387,  0.76629502,  0.8850826 ,  0.62223291,  1.43796271,
##         3.96249789,  1.36342753,  0.84616755,  1.06261907,  0.9622986 ,
##         1.22543524,  0.85206104,  5.16471445,  1.32145597,  0.57929458,
##        -0.27700252,  0.44894949,  1.11997725,  5.08188136,  1.02904826],
##       [ 0.96224203,  0.94630195,  0.82184915,  1.10065024,  0.31837072,
##         3.99005018,  0.38413806,  0.86175214,  0.65157157,  0.75296003,
##         0.50244078,  0.85823874,  5.19664839,  0.41889021,  1.14877159,
##         2.20540342,  1.29127436,  0.60004435,  5.11365131,  0.68454176]]),np.array([[ -1.66549004],
##       [ -1.29230526],
##       [ -1.53169105],
##       [ -2.01468532],
##       [ -2.03828191],
##       [  5.00631703],
##       [ -2.16730589],
##       [ -1.60717013],
##       [ -1.66711323],
##       [ -2.01511939],
##       [ -1.77359077],
##       [ -1.74893403],
##       [ 10.12310877],
##       [ -1.94763642],
##       [ -2.20035083],
##       [ -2.50946155],
##       [ -2.01212002],
##       [ -1.81476804],
##       [  9.66036961],
##       [ -1.73055929]])]
##red=mlp(tmpm=1,mp=tmp)
##red.train(inp,out,0.5,0.5)