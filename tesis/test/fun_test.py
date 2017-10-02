import numpy as np

div=[0,4,8,12,16,20,24,28,32,36,40]
pin=35

# empieza la funcion
pout=50
for i in range(len(div)-1):
    if div[i]<=pin<=div[i+1]:
        pout=i
print (pout)
