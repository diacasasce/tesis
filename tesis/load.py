import numpy as np
O=np.load('o.npy')
I=np.load('fi.npy')
inp=[]
for i in range(len(I)):
    for j in range(len(I[i])):
##        print(I[i][j])
        inp.append(I[i][j])
out=[]
for i in range(len(O)):
    for j in range(len(O[i])):
##        print(O[i][j])
        out.append(O[i][j])