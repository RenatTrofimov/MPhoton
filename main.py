from numba import njit, prange, vectorize, float64
from numba import jit
import numpy as np
import time

import matplotlib.pyplot as plt
####################################################################
@njit(parallel=True)
def calculation(G, data, previousLayer, curentLayer, nextLayer):
    ht = data[0]
    hx = data[1]
    hy = data[2]
    hz = data[3]
    F = data[4]
    p = data[5]
    ht2 = ht**2
    hx2 = hx*hx
    hy2 = hy*hy
    hz2 = hz*hz
    for i in prange(1,curentLayer.shape[0]-1):
        for j in prange(1,curentLayer.shape[1]-1):
            for k in prange(1,curentLayer.shape[2]-1):
                suming = 0.0
                cur = curentLayer[i,j,k]
                prev = previousLayer[i,j,k]
                cur2 = cur+cur
                for r in prange(G.shape[0]):
                    suming += G[r]*np.sin((r+1)*cur)
                next = (cur2 - prev)/ht2
                + ((curentLayer[i+1,j,k]-cur2+curentLayer[i-1,j,k])/(hx2)
                + (curentLayer[i,j+1,k]-cur2+curentLayer[i,j-1,k])/(hy2)
                + (curentLayer[i,j,k+1]-cur2+curentLayer[i,j,k-1])/(hz2)
                - F*((cur-prev)/ht)**int(2*p-1)
                - suming)
                nextLayer[i,j,k] = next*ht2
    #return np.copy(curentLayer), np.copy(nextLayer)
@njit(parallel=True)
def initLayers(initData, data, previousLayer,curentLayer):
    ht = data[0]
    hx = data[1]
    hy = data[2]
    hz = data[3]
    kx = initData[0]
    ky = initData[1]
    kz = initData[2]
    b = initData[3]
    u = initData[4]
    gamma = initData[5]
    
    for i in prange(1,curentLayer.shape[0]-1):
        for j in prange(1,curentLayer.shape[1]-1):
            for k in prange(1,curentLayer.shape[2]-1):
                previousLayer[i,j,k] = np.exp(-(hz*(k-0.5*kz)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))
                curentLayer[i,j,k] = np.exp(-(hz*(k-0.5*kz-u*ht)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))

###########################################################################
initData = np.array([
    300.0,
    300.0,
    600.0,
    0.03,
    0.93,
    np.sqrt(1-0.93**2)
], dtype=np.float64)
###########################################################################
data = np.array([0.01,
                0.06,
                0.06,
                0.06,
                0.0001,
                2.0
], dtype=np.float64)

G = np.array([
                0.910946143,
                0.327382235,
                0.184638392,
                0.118116261,
                0.080225077,
                0.056415084,
                0.040572248,
                0.029635502,
                0.021892511
], dtype=np.float64)
Layer = np.zeros((3, int(initData[0]), int(initData[1]), int(initData[2])), dtype=np.float64)

##############################################################################


initLayers(initData, data, Layer[0,:,:,:], Layer[1,:,:,:])
for k in range(9):
    for i in range(50):
        start = time.perf_counter()
        calculation(G, data,  Layer[(0+i)%3,:,:,:],  Layer[(1+i)%3,:,:,:],  Layer[(2+i)%3,:,:,:])
        #initLayers.parallel_diagnostics(level=4)
        end = time.perf_counter()
        #print("Elapsed (after compilation) = {}s".format((end - start)))
    
    Data = (Layer[(2+50-1)%3,:,:,:] - Layer[(1+50-1)%3,:,:,:])/data[0]
    with open(f"{(k+1)*50}_1.txt", 'w') as outfile:
   
        outfile.write('# Array shape: {0}\n'.format(Data.shape))
        
        for data_slice in Data:

            np.savetxt(outfile, data_slice, fmt='%-7.2f')

##################################################################################################

fig, ax = plt.subplots()

ax.imshow(Layer[(1000)%3,:,int(initData[1]/2),:])

plt.show()