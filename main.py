<<<<<<< HEAD
from numba import njit, prange, vectorize, float64
=======
from numba import njit, prange
>>>>>>> main
from numba import jit
import numpy as np
import time

<<<<<<< HEAD
import matplotlib.pyplot as plt
####################################################################


@njit(parallel=True)

=======
@njit(parallel=True)
>>>>>>> main
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
<<<<<<< HEAD
    
=======
>>>>>>> main
    for i in prange(1,curentLayer.shape[0]-1):
        for j in prange(1,curentLayer.shape[1]-1):
            for k in prange(1,curentLayer.shape[2]-1):
                suming = 0.0
                cur = curentLayer[i,j,k]
<<<<<<< HEAD
                cur2 = cur+cur
                for r in prange(G.shape[0]):
                    suming += G[r]*np.sin((r+1)*cur)
                nextValue = (cur2 - previousLayer[i,j,k])/ht2
                + (curentLayer[i+1,j,k]-cur2+curentLayer[i-1,j,k])/(hx2)
                + (curentLayer[i,j+1,k]-cur2+curentLayer[i,j-1,k])/(hy2)
                + (curentLayer[i,j,k+1]-cur2+curentLayer[i,j,k-1])/(hz2)
                - F*((cur-previousLayer[i,j,k])/ht)**(2*p-1)
                - suming
                nextLayer[i,j,k] = nextValue*ht2
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
                previousLayer[i,j,k] = np.exp(-(hz*(k-0.5*kz)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))
                curentLayer[i,j,k] = np.exp(-(hz*(k-0.5*kz-u*ht)/gamma)**2) * np.exp(-b*(hx*(i-0.5*kx)**2)) * np.exp(-b*(hy*(j-0.5*ky)**2))
    
                
    
    return previousLayer, curentLayer

###########################################################################
initData = np.array([
    300.0,
    300.0,
    600.0,
    0.03,
    0.93,
    np.sqrt(1-0.93**2)
], dtype=np.float64)
=======
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
###########################################################################
curentLayer = np.arange(54000000).reshape(300, 300, 600)
previousLayer = np.arange(54000000).reshape(300, 300, 600)
>>>>>>> main
data = np.array([0.01,
                0.06,
                0.06,
                0.06,
                0.0001,
<<<<<<< HEAD
                2.0
], dtype=np.float64)
=======
                2.0])
>>>>>>> main
G = np.array([
                0.910946143,
                0.327382235,
                0.184638392,
                0.118116261,
                0.080225077,
                0.056415084,
                0.040572248,
                0.029635502,
<<<<<<< HEAD
                0.021892511
], dtype=np.float64)
curentLayer = np.zeros((int(initData[0]), int(initData[1]), int(initData[2])), dtype=np.float64)
previousLayer = np.zeros((int(initData[0]), int(initData[1]), int(initData[2])), dtype=np.float64)
dummyZerosArray = np.zeros((int(initData[0]), int(initData[1]), int(initData[2])), dtype=np.float64)
##############################################################################


previousLayer, curentLayer = initLayers(initData, data, previousLayer, curentLayer)


print(np.sum(previousLayer))
print(np.sum(curentLayer))
for i in range(450):
    start = time.perf_counter()
    previousLayer = calculation(G, data, previousLayer,curentLayer, dummyZerosArray)
    #initLayers.parallel_diagnostics(level=4)
    end = time.perf_counter()
    previousLayer, curentLayer = curentLayer, previousLayer
    print(np.sum(curentLayer))
    print("Elapsed (after compilation) = {}s".format((end - start)))


=======
                0.021892511])
print(data)
##############################################################################

for i in range(10):
    start = time.perf_counter()
    a = calculation(G, data, previousLayer,curentLayer, np.zeros((300, 300, 600)))
    #calculation.parallel_diagnostics(level=4)
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))
>>>>>>> main
