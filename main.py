from numba import njit, prange
from numba import jit
import numpy as np
import time

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
###########################################################################
curentLayer = np.arange(54000000).reshape(300, 300, 600)
previousLayer = np.arange(54000000).reshape(300, 300, 600)
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
print(data)
##############################################################################

for i in range(10):
    start = time.perf_counter()
    a = calculation(G, data, previousLayer,curentLayer, np.zeros((300, 300, 600)))
    #calculation.parallel_diagnostics(level=4)
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))