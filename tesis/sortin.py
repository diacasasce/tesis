import numpy as np
ctype=[('x',int),('y',int)]
a=[(1,2),(1,5),(1,3),(3,4),(3,2),(2,1)]
ar=np.array(a,dtype=ctype)
sar=np.sort(ar)
aro=[]
for i in sar:
    aro.append((int(i['x']),int(i['y'])))
##def bubbleXY(data):