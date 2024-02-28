from numba import njit, prange
from numba import jit
import numpy as np
import time

import matplotlib.pyplot as plt
####################################################################
@njit(parallel=True)
def calculation(G, data, previousLayer,curentLayer, nextLayer):
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
                cur2 = 2.0*cur
                for r in prange(G.shape[0]):
                    suming += G[r]*np.sin((r+1)*cur)
                nextLayer[i,j,k] = cur2
                - previousLayer[i,j,k]
                + ht2*(curentLayer[i+1,j,k]-cur2-curentLayer[i-1,j,k])/(hx2)
                + ht2*(curentLayer[i,j+1,k]-cur2-curentLayer[i,j-1,k])/(hy2)
                + ht2*(curentLayer[i,j,k+1]-cur2-curentLayer[i,j,k-1])/(hz2)
                - ht2*F*((cur-previousLayer[i,j,k])/ht)**(2*p-1)
                - ht2*suming
    return nextLayer
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
                previousLayer[i,k,j] = np.exp(-(hz*(k-0.5*kz)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))
    
    for i in prange(1,curentLayer.shape[0]-1):
        for j in prange(1,curentLayer.shape[1]-1):
            for k in prange(1,curentLayer.shape[2]-1):
                curentLayer[i,k,j] = np.exp(-(hz*(k-0.5*kz-u*ht)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))
    
    return previousLayer, curentLayer

###########################################################################
initData = np.array([
    300.0,
    300.0,
    600.0,
    0.03,
    0.93,
    np.sqrt(1-0.93**2)
])
data = np.array([0.01,
                0.06,
                0.06,
                0.06,
                0.0001,
                2.0])
G = np.array([
                0.910946143,
                0.327382235,
                0.184638392,
                0.118116261,
                0.080225077,
                0.056415084,
                0.040572248,
                0.029635502,
                0.021892511])
curentLayer = np.arange(int(initData[0]*initData[1]*initData[2])).reshape(int(initData[0]), int(initData[1]), int(initData[2]))
previousLayer = np.arange(int(initData[0]*initData[1]*initData[2])).reshape(int(initData[0]), int(initData[1]), int(initData[2]))

##############################################################################


previousLayer, curentLayer = initLayers(initData, data, previousLayer, curentLayer)
dummyZerosArray = np.zeros((int(initData[0]), int(initData[1]), int(initData[2])))
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
plt.show()
for i in range(450):
    start = time.perf_counter()
    previousLayer = calculation(G, data, previousLayer,curentLayer, dummyZerosArray)
    #initLayers.parallel_diagnostics(level=4)
    end = time.perf_counter()
    previousLayer, curentLayer = curentLayer, previousLayer
    print("Elapsed (after compilation) = {}s".format((end - start)))